from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pprint
import threading
import os
import re
import io
import time
from src.millionaire_client.models import Question, Option
from src.models import ExperimentConfig


class Guesser(ABC):     
    """
    Generic interface for all Guessers.
    """
    _whisper_models = {}
    _whisper_lock = threading.Lock()

    def __init__(self, config: ExperimentConfig, mode: str = "text", transcription_model: str = "tiny"):
        if not isinstance(config, ExperimentConfig):
            raise ValueError(f"config must be a {type(ExperimentConfig)}")
        self.config = config
        self.mode = mode
        self.transcription_model_size = transcription_model
        self.search_time: float = 0.0
        self.reasoning_time: float = 0.0
        self.transcription_time: float = 0.0

    def preload(self):
        """Preload models to avoid delays during the first question."""
        if self.mode == "speech":
            _ = self.whisper_model
            print(f"Whisper '{self.transcription_model_size}' model preloaded and ready.")

    @property
    def whisper_model(self):
        """Thread-safe lazy loading of the specified Whisper model."""
        size = self.transcription_model_size
        if size not in Guesser._whisper_models:
            with Guesser._whisper_lock:
                # Double-check pattern to handle concurrent threads
                if size not in Guesser._whisper_models:
                    try:
                        import whisper
                        print(f"Loading Whisper '{size}' model...")
                        Guesser._whisper_models[size] = whisper.load_model(size)
                    except ImportError:
                        print("Error: 'whisper' library not found. Please install it with 'pip install openai-whisper'.")
                        raise
        return Guesser._whisper_models[size]


    def print(self):
        pprint.pprint(self.config)

    @abstractmethod
    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        pass
    
    def format_question_for_llm(self, question: Question) -> str:
        """
        Format question
         """
        prompt_lines = [f"Question: {question.text}\n", "Options:"]
        
        for index, option in enumerate(question.options):
            prompt_lines.append(f"[{index}] {option.text}")
            
        return "\n".join(prompt_lines)

    def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio bytes using Whisper and apply basic post-processing."""
        import librosa
        import whisper
        import numpy as np

        try:
            # Improvement #1: In-memory processing using io.BytesIO to avoid disk I/O latency
            audio_np, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
            
            # Improvement #3: Silence trimming to minimize data passed to the model
            audio_np, _ = librosa.effects.trim(audio_np)
            
            # Check if audio is empty after trimming (prevents RuntimeError)
            if audio_np.size == 0:
                return ""

            # Improvement #2: Optimized decoding pipeline (Skip language detection)
            # We force English since the quiz questions are known to be in English
            audio_padded = whisper.pad_or_trim(audio_np)
            
            # Use the lock to ensure only one thread uses the Whisper model for inference at a time
            # Whisper's internal state (like KV caches) is not thread-safe.
            with Guesser._whisper_lock:
                mel = whisper.log_mel_spectrogram(audio_padded).to(self.whisper_model.device)
                options = whisper.DecodingOptions(language="en", fp16=False)
                result = whisper.decode(self.whisper_model, mel, options)
                text = result.text
            
            return self._post_process_text(text)
        except Exception as e:
            print(f"Warning: Transcription failed: {e}")
            return ""

    def _post_process_text(self, text: str) -> str:
        """Advanced filtering of speech artifacts, laughter, and false option prefixes."""
        
        text = text.strip()
        
        # 1. Remove repeated word/character patterns (3 or more) like "ha ha ha" or "C-C-C-C"
        # Matches a word followed by 2 or more repetitions of the same word (with space/hyphen)
        text = re.sub(r'(\b\w+)\b([- ]?\1\b){2,}', '', text, flags=re.IGNORECASE)
        
        # 2. Remove common laughter patterns not caught by the repetition regex
        text = re.sub(r'\b(ha|he|hi|ho|ah)\b', '', text, flags=re.IGNORECASE)

        # 3. Aggressive removal of "Option X" style prefixes at the START
        # Catches: "Option A", "options see", "Pop's in B", "Topsy and D", "Top-ion D", etc.
        # Logic: find variations of "Option", "Top*", "Pop*" at the start and remove everything up to the first A, B, C, or D.
        text = re.sub(r'^(options?|top\w*|pop\w*).*?\b([abcd]|see|sea)\b[\s,.;:?!-]*', '', text, flags=re.IGNORECASE)

        # 4. Remove word-based prefixes that might appear elsewhere or weren't caught by the start regex
        text = re.sub(r'\b(options?|topst?ion|topson|topption|pops)\b(\s+and)?\s*\w+[\s,.;:?!]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(topst?ion|topson|topption|pops)\w+\b', '', text, flags=re.IGNORECASE)
        
        # 5. Remove "Topsyn*" false transcriptions (removes only that word)
        text = re.sub(r'\btopsynd\w*\b', '', text, flags=re.IGNORECASE)

        # 6. Original filler regex (e.g., um, uhm, hmm)
        # Starts with [aoumh] and followed by one or more [hm], excluding "am"
        filler_regex = r'\b(?!am\b)[aoumh][hm]+\b'
        text = re.sub(filler_regex, '', text, flags=re.IGNORECASE)
        
        # 7. Whisper artifacts and final cleanup
        # Remove parenthetical or bracketed text, but try to preserve content if it wraps the whole thing
        # If the entire text is wrapped in () or [], strip them
        if (text.startswith('(') and text.endswith(')')) or (text.startswith('[') and text.endswith(']')):
            text = text[1:-1].strip()
            
        # Remove any remaining nested brackets/parentheses
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Clean up multiple spaces and strip
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove leading/trailing punctuation leftovers that might remain after prefix removal
        # Specifically targets "., !" and other punctuation at the start or end
        text = re.sub(r'^[\s,.;:?!-]+|[\s,.;:?!-]+$', '', text)
        
        return text

    def get_speech_question(self, game_session: Any) -> Question:
        """
        Process speech mode: fetch audio clips and transcribe them.
        Optimized to start transcription while fetching subsequent options.
        
        The 30s timer starts after the last option (D) is fetched from the server.
        """
        if game_session is None:
            raise ValueError("game_session is required for speech mode")

        print("Speech mode active: Fetching and transcribing audio...")
        self.transcription_time = 0.0
        
        # 1. Fetch question audio
        q_audio = game_session.fetch_audio_question()
        
        results = {}
        threads = []

        def transcription_task(key, audio_bytes):
            results[key] = self._transcribe_audio(audio_bytes)

        # Start transcribing question immediately while we fetch options
        t_q = threading.Thread(target=transcription_task, args=("q", q_audio))
        t_q.start()
        threads.append(t_q)

        options_ids = []
        
        # 2. Fetch options A, B, C, D sequentially (as required by the API)
        # We start transcribing each option as soon as its audio is received.
        for i in range(4):
            letter = chr(65 + i)
            opt_audio = game_session.fetch_audio_option_next()
            
            t_opt = threading.Thread(target=transcription_task, args=(f"opt_{i}", opt_audio))
            t_opt.start()
            threads.append(t_opt)
            
            # Preserve the option IDs from the game state
            if game_session.current_question and i < len(game_session.current_question.options):
                options_ids.append(game_session.current_question.options[i].id)
            else:
                options_ids.append(i)

        # The user wants to track transcription time AFTER the last option is pulled
        start_transcription_tracking = time.time()

        # Wait for all transcription threads to complete
        print("  Waiting for transcriptions to finish...")
        for t in threads:
            t.join()

        self.transcription_time = time.time() - start_transcription_tracking

        # Build the Question object with transcribed text for the LLM
        transcribed_options = []
        for i in range(4):
            text = results.get(f"opt_{i}", "")
            transcribed_options.append(Option(id=options_ids[i], text=text))

        return Question(
            id=game_session.current_question.id if game_session.current_question else 0,
            text=results.get("q", ""),
            options=transcribed_options,
            level=game_session.current_level
        )
