import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from src.guesser.engine.llmprovider import get_llm, unload_model
from src.guesser.engine.router import Router
from src.guesser.engine.query_translator import QueryTranslator
from src.millionaire_client.models import Question
from src.guesser.ingestion.loader import Loader
from src.models import ApproachType
from src.guesser.engine.utils.code_executor import execute_pot_code
from src.guesser.engine.utils.web_search import web_search
from src.guesser.engine.utils.math_classifier import MathClassifier
from src.guesser.engine.prompts import MCQ_PROMPT_MATHS_THEORY
from src.guesser.engine.configs import TRANSLATOR_MODEL, FALLBACK_INFERENCE_MODEL

class GuesserEngine:

    def __init__(self, llm_model="llama3.2", db_path:str="", embedding_model:str="", top_k=2, temperature=0.0, theme:str="", debug:bool=False, pre_load_models:list=None):
        self.llm_model = llm_model
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.theme = None
        self.approach_type = None
        self.debug = debug
        self.current_fallback_model = None

        # Initialize Query Translator
        self.translator = QueryTranslator(debug=self.debug)

        # Initialize Math Classifier (reuses translator-sized model, already warm in VRAM)
        self.math_classifier = MathClassifier(model_name=TRANSLATOR_MODEL, debug=self.debug)

        # Theme Cache stores: (retriever, prompt_template, approach_type, llm, current_theme_config, translator, is_rag, fallback_model)
        self.theme_cache = {} 
        
        # Pre-load requested models into VRAM with warm-up
        if pre_load_models:
            if self.debug:
                print(f"[DEBUG] Pre-loading models into VRAM: {pre_load_models}")
            for model in pre_load_models:
                llm = get_llm(model_name=model)
                try:
                    # Send a trivial request to force-load the model weights
                    llm.complete(" ") 
                    if self.debug:
                        print(f" [Ollama] Model '{model}' warmed up and resident in VRAM.")
                except Exception as e:
                    print(f" [Ollama] Warning: Could not warm up model '{model}': {e}")
            
            # Post-loading VRAM diagnostic
            self._check_vram_status()

        if theme:
            self.set_theme(theme)
        else:
            self.set_theme("science and nature")

    def _check_vram_status(self):
        """
        Runs 'ollama ps' and prints a report on model residency and processor usage.
        """
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
            output = result.stdout
            
            if self.debug:
                print("\n--- OLLAMA VRAM DIAGNOSTIC ---")
                print(output)
            
            lines = output.strip().split('\n')
            if len(lines) <= 1:
                print(" [Ollama] Warning: No models detected as resident in memory.")
                return

            cpu_models = []
            gpu_models = []
            
            for line in lines[1:]: # Skip header
                parts = line.split()
                if not parts: continue
                
                # Processor info is usually the 4th column in 'ollama ps'
                # But it can vary. We'll look for "CPU" or "GPU" in the string.
                model_name = parts[0]
                if "CPU" in line:
                    cpu_models.append(model_name)
                elif "GPU" in line:
                    gpu_models.append(model_name)
            
            if cpu_models:
                print("\n" + "!"*60)
                print(f" CRITICAL VRAM WARNING: The following models are running on CPU:")
                for m in cpu_models:
                    print(f"  - {m}")
                print("\n This will cause EXTREME slowness and likely TIMEOUTS.")
                print(" Your GPU VRAM is likely full. Consider using smaller models.")
                print("!"*60 + "\n")
            elif gpu_models:
                print(f" [Ollama] Success: {len(gpu_models)} models are successfully pinned to GPU VRAM.")
            
        except Exception as e:
            if self.debug:
                print(f" [Ollama] Could not run VRAM diagnostic: {e}")

    def set_theme(self, theme: str):
        """
        Dynamically switch the theme (index, prompt, model and approach) used by the engine.
        """
        if theme == self.theme:
            return

        if theme in self.theme_cache:
            self.retriever, self.prompt_template, self.approach_type, self.llm, self.current_theme_config, self.translator, self.is_rag, self.current_fallback_model = self.theme_cache[theme]
            self.theme = theme
            if self.debug:
                print(f"[DEBUG] Switched to cached theme: {theme}")
            return

        # Use Router to get the new configuration
        theme_config = Router(theme).route() 
        
        collection_name = theme_config.collection_name
        prompt_template = theme_config.prompt_template
        approach_type = theme_config.approach_type
        is_rag = theme_config.is_rag
        fallback_model = theme_config.fallback_model
        
        # Override theme model with instance model if provided
        model_name = self.llm_model if self.llm_model else theme_config.model_name

        # Determine stop sequences based on approach
        # POT needs multi-line, others benefit from early stopping
        # Exception: Math themes should ALWAYS allow multi-line for scratchpad reasoning
        is_math = "math" in theme.lower()
        stop_seq = None if (approach_type == ApproachType.POT or is_math) else ["\n", "\n\n"]

        # Get or create LLM for this specific theme
        llm = get_llm(
            model_name=model_name,
            temperature=theme_config.temperature,
            num_predict=theme_config.num_predict,
            stop=stop_seq,
            timeout=theme_config.timeout
        )

        # Pre-load fallback model if it's different to ensure it's in VRAM
        if fallback_model and fallback_model != model_name:
            if self.debug:
                print(f"[DEBUG] Pre-loading fallback model: {fallback_model}")
            get_llm(
                model_name=fallback_model,
                temperature=theme_config.temperature,
                num_predict=theme_config.num_predict,
                stop=stop_seq,
                timeout=theme_config.timeout
            )
        
        # Update Translator if needed
        translator = self.translator
        if theme_config.translator_model:
            if not self.translator or self.translator.model_name != theme_config.translator_model:
                translator = QueryTranslator(model_name=theme_config.translator_model, debug=self.debug)

        index = Loader(self.db_path, self.embedding_model).get_index(collection_name)
        retriever = index.as_retriever(similarity_top_k=theme_config.top_k)
        
        self.theme_cache[theme] = (retriever, prompt_template, approach_type, llm, theme_config, translator, is_rag, fallback_model)
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.approach_type = approach_type
        self.llm = llm
        self.current_theme_config = theme_config
        self.translator = translator
        self.theme = theme
        self.is_rag = is_rag
        self.current_fallback_model = fallback_model
        
        if self.debug:
            print(f"[DEBUG] Theme Set: {theme} | Approach: {approach_type} | RAG: {is_rag}")

    def _reciprocal_rank_fusion(self, results_list, k=60):
        node_scores = {}
        for nodes in results_list:
            for rank, node_with_score in enumerate(nodes):
                node_id = node_with_score.node.node_id
                if node_id not in node_scores:
                    node_scores[node_id] = {
                        "rrf_score": 0.0, 
                        "node": node_with_score.node,
                        "original_score": node_with_score.score
                    }
                else:
                    # Keep the highest original similarity score
                    node_scores[node_id]["original_score"] = max(node_scores[node_id]["original_score"], node_with_score.score)
                node_scores[node_id]["rrf_score"] += 1.0 / (k + rank + 1)
        
        sorted_nodes = sorted(node_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_nodes

    def generate_queries(self, question: Question):
        only_question = question.text
        
        options_list = []
        raw_options = []
        for index, option in enumerate(question.options):
            options_list.append(f"[{index}] {option.text}")
            raw_options.append(option.text)
        
        options_text = " ".join(options_list)
        question_and_options = f"Question: {only_question} Options: {options_text}"
        
        return only_question, question_and_options, options_text, raw_options

    def chat(self, prompt: str) -> dict:
        """
        Directly chat with the LLM using RAG context for debugging.
        """
        nodes = self.retriever.retrieve(prompt)
        
        # De-duplicate by content to avoid redundant information
        unique_contents = []
        seen_contents = set()
        for n in nodes:
            content = n.get_content().strip()
            if content not in seen_contents:
                unique_contents.append(content)
                seen_contents.add(content)
        
        context_str = "\n\n".join(unique_contents)
        
        final_prompt = f"Context:\n{context_str}\n\nUser Question: {prompt}\n\nAnswer:"
        response = self.llm.complete(final_prompt)
        
        return {
            "response": str(response).strip(),
            "context_chunks": unique_contents
        }

    def answer(self, question: Question):
        start_total = time.time()
        self.last_search_time = 0.0
        self.last_reasoning_time = 0.0

        q_only, q_full, options_str, options_raw = self.generate_queries(question)

        if self.debug:
            print(f"\n[DEBUG] Starting Inference for Level {getattr(question, 'level', 'unknown')}")
            print(f"[DEBUG] Question: {q_only[:100]}...")

        # Math hybrid routing: classify between 'calculation' (PoT) and 'theory' (direct).
        # Override approach/prompt/model just for this question, restored in finally.
        _orig_state = (self.approach_type, self.prompt_template, self.llm_model)
        is_math = "math" in self.theme.lower()
        if is_math and self.approach_type == ApproachType.POT:
            if self.debug:
                print(f"[DEBUG] Math theme detected -- running MathClassifier")
            category = self.math_classifier.classify(q_only)
            print(f" [MathClassifier] '{category}'")
            if category == "theory":
                self.approach_type = ApproachType.RAG
                self.prompt_template = MCQ_PROMPT_MATHS_THEORY
                self.llm_model = FALLBACK_INFERENCE_MODEL
                if self.debug:
                    print(f"[DEBUG] Math override applied:")
                    print(f"[DEBUG]   approach: POT -> RAG (no retrieval, prompt-only)")
                    print(f"[DEBUG]   prompt:   MCQ_PROMPT_MATHS_POT -> MCQ_PROMPT_MATHS_THEORY")
                    print(f"[DEBUG]   model:    {_orig_state[2]} -> {FALLBACK_INFERENCE_MODEL}")
            else:
                if self.debug:
                    print(f"[DEBUG] Math kept on PoT pipeline (no override)")

        try:
            context_str = ""
            final_nodes = []
        
            # Decide if retrieval is needed
            if self.approach_type == ApproachType.SEARCH:
                search_start = time.time()
                try:
                    print(f" [WebSearch] Searching for: {q_only[:80]}...")
                    context_str = web_search(
                        q_only,
                        max_results=self.current_theme_config.max_search_results,
                        debug=self.debug
                    )
                    if context_str:
                        print(f" [WebSearch] Got {len(context_str)} chars of context.")
                    else:
                        print(f" [WebSearch] No results. LLM will use internal knowledge.")
                finally:
                    self.last_search_time = time.time() - search_start
                    if self.debug:
                        print(f"[DEBUG] Web Search Finished in {self.last_search_time:.2f}s")

            elif self.is_rag:
                search_start = time.time()
                try:
                    # Use Query Translator to strip noise from the main question
                    translated_q = self.translator.translate(q_only)
                
                    if self.current_theme_config.two_vector_search:
                        if self.debug:
                            print(f"[DEBUG] Performing Two-Vector Search: '{translated_q}' AND full prompt")
                        # Hybrid multi-pass search (RRF) - Parallelized for speed
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            future_only = executor.submit(self.retriever.retrieve, translated_q)
                            future_full = executor.submit(self.retriever.retrieve, q_full)
                            nodes_only = future_only.result()
                            nodes_full = future_full.result()
                    
                        fused_items = self._reciprocal_rank_fusion([nodes_only, nodes_full])
                    else:
                        if self.debug:
                            print(f"[DEBUG] Performing Single-Vector Search: '{translated_q}'")
                        # Single clean search for speed
                        nodes = self.retriever.retrieve(translated_q)
                        fused_items = [
                            {
                                "rrf_score": 1.0 / (60 + i + 1), 
                                "node": n, 
                                "original_score": n.score
                            } 
                            for i, n in enumerate(nodes)
                        ]

                    # De-duplicate by content
                    seen_contents = set()
                    for item in fused_items:
                        content = item["node"].get_content().strip()
                        if content not in seen_contents:
                            final_nodes.append(item)
                            seen_contents.add(content)
                
                    # Filter by similarity threshold
                    threshold = self.current_theme_config.similarity_threshold
                    final_nodes = [item for item in final_nodes if item["original_score"] >= threshold]
                
                    final_nodes = final_nodes[:self.current_theme_config.top_k]
                    context_str = "\n\n".join([item["node"].get_content() for item in final_nodes])
                
                    if final_nodes:
                        print(f" [Retriever] Top {len(final_nodes)} chunks passing similarity threshold (>= {threshold}):")
                        for idx, item in enumerate(final_nodes):
                            print(f"   - Chunk {idx+1}: Similarity={item['original_score']:.4f} | RRF={item['rrf_score']:.4f}")
                            if self.debug:
                                print(f"     [CONTENT]: {item['node'].get_content()}")
                    else:
                        print(f" [Retriever] No chunks passed similarity threshold (>= {threshold}). Using LLM's internal knowledge.")
                finally:
                    self.last_search_time = time.time() - search_start
                    if self.debug:
                        print(f"[DEBUG] Search Phase Finished in {self.last_search_time:.2f}s")

            # --- DYNAMIC MODEL SWAPPING LOGIC ---
            # 1. Determine active vs inactive models
            # For non-RAG/non-SEARCH themes (like Math/POT), respect the theme's dedicated model_name.
            # For RAG/SEARCH themes, the instance-level self.llm_model can override.
            if not self.is_rag and self.approach_type != ApproachType.SEARCH:
                primary_model_name = self.current_theme_config.model_name
            else:
                primary_model_name = self.llm_model if self.llm_model else self.current_theme_config.model_name
            fallback_model_name = self.current_fallback_model
        
            # LOGIC:
            # - If it's a non-RAG theme (like Math), ALWAYS use the primary model.
            # - If it's a RAG theme and NO context was found, use the larger/fallback model for knowledge.
            # - If it's a RAG theme and context WAS found, use the smaller/primary model for efficiency.
        
            if not self.is_rag:
                active_model_name = primary_model_name
                inactive_model_name = fallback_model_name
            elif not final_nodes and fallback_model_name:
                active_model_name = fallback_model_name
                inactive_model_name = primary_model_name
                if self.debug:
                    print(f"[DEBUG] No context found. Switching to knowledge fallback: {active_model_name}")
            else:
                active_model_name = primary_model_name
                inactive_model_name = fallback_model_name

            # 2. Check if we need to swap
            current_loaded_model = getattr(self.llm, 'model', None)
            if current_loaded_model != active_model_name:
                # We no longer force unload to keep both in VRAM as requested

                # Load/Switch to the active model
                stop_seq = None if self.approach_type == ApproachType.POT else ["\n", "\n\n"]
                self.llm = get_llm(
                    model_name=active_model_name,
                    temperature=self.current_theme_config.temperature,
                    num_predict=self.current_theme_config.num_predict,
                    stop=stop_seq,
                    timeout=self.current_theme_config.timeout
                )
            # -----------------------------------

            if self.approach_type == ApproachType.DIRECT_LLM:
                final_prompt = f"You are an expert assistant. Answer the following multiple choice question.\n\nQUESTION:\n{q_only}\n\nOPTIONS:\n{options_str}\n\nOutput ONLY the correct option index (0, 1, 2, or 3).\nFinal Option Index: "
            elif self.approach_type == ApproachType.POT:
                # For POT, we can pass the raw options list to make code generation easier
                final_prompt = self.prompt_template.format(
                    context=context_str,
                    question=q_only,
                    options=options_str,
                    options_list=str(options_raw)
                )
            else:
                final_prompt = self.prompt_template.format(
                    context=context_str,
                    question=q_only,
                    options=options_str
                )
            
            if self.debug:
                actual_model = getattr(self.llm, 'model', active_model_name)
                print(f"[DEBUG] Reasoning Phase (Model: {actual_model})")
                print(f"[DEBUG] Final Prompt Sent to LLM:\n{'-'*60}\n{final_prompt}\n{'-'*60}")

            reasoning_start = time.time()
            final_answer = None
            raw_response = ""
            _POT_MAX_RETRIES = 2

            try:
                if self.approach_type == ApproachType.POT:
                    for attempt in range(_POT_MAX_RETRIES + 1):
                        try:
                            response = self.llm.complete(final_prompt)
                            raw_response = str(response).strip()

                            if self.debug:
                                print(f"[DEBUG] PoT Attempt {attempt + 1} raw response: {raw_response[:400]}...")

                            pot_result = execute_pot_code(raw_response, options=options_raw)
                            if pot_result is not None:
                                final_answer = str(pot_result)
                                print(f" [PoT] Attempt {attempt + 1} succeeded. Result: {final_answer}")
                                break
                            else:
                                if attempt < _POT_MAX_RETRIES:
                                    print(f" [PoT] Attempt {attempt + 1} execution failed. Retrying...")
                                else:
                                    print(f" [PoT] All {_POT_MAX_RETRIES + 1} attempts failed. Falling back to raw response.")
                                    final_answer = raw_response
                        except Exception as e:
                            if attempt < _POT_MAX_RETRIES:
                                print(f" [PoT] Attempt {attempt + 1} error: {e}. Retrying...")
                            else:
                                print(f" [PoT] All attempts exhausted with error: {e}")
                                final_answer = raw_response
                else:
                    response = self.llm.complete(final_prompt)
                    raw_response = str(response).strip()
                    if self.debug:
                        print(f"[DEBUG] Raw LLM Response: {raw_response[:500]}...")
                    final_answer = raw_response
            finally:
                self.last_reasoning_time = time.time() - reasoning_start
                if self.debug:
                    print(f"[DEBUG] Reasoning Phase Finished in {self.last_reasoning_time:.2f}s")

            if final_answer is None:
                final_answer = raw_response

            fonts = [item["node"].get_content() for item in final_nodes]
            resolution = [item["node"].metadata.get('answer', 'No Resolution Found') for item in final_nodes]

            total_elapsed = time.time() - start_total
            print(f"[{self.approach_type.upper()}] Total: {total_elapsed:.2f}s | Search: {self.last_search_time:.2f}s | Reasoning: {self.last_reasoning_time:.2f}s")

            return {
                "answer": final_answer,
                "sources": fonts,
                "resolution_method": resolution,
                "search_time": self.last_search_time,
                "reasoning_time": self.last_reasoning_time,
                "total_time": total_elapsed,
                "chunks_metadata": [
                    {"text": item["node"].get_content(), "similarity": item["original_score"], "rrf": item["rrf_score"]}
                    for item in final_nodes
                ]
            }
        finally:
            self.approach_type, self.prompt_template, self.llm_model = _orig_state
