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
        """Filter speech artifacts, laughter, and false option prefixes using consolidated patterns."""
        
        text = text.strip()
        
        # 1. Remove bracketed text (Whisper artifacts) unless it wraps the entire string
        if (text.startswith('(') and text.endswith(')')) or (text.startswith('[') and text.endswith(']')):
            text = text[1:-1]
        text = re.sub(r'[\[\(].*?[\]\)]', '', text)

        # 2. Apply consolidated regex patterns sequentially
        patterns = [
            r'(\b\w+)\b([- ]?\1\b){2,}',                                         # Repeated words (3+ times)
            r'^(?:options?|[tp]op\w*).*?\b(?:[abcd]|see|sea)\b',                 # Start-of-sentence prefix mishears
            r'\b(?:options?|topst?ion|topson|topption|pops|topsynd)\w*\b(?:\s+(?:and\s+)?\w+)?', # Stray false options
            r'\b(?:(?!am\b)[aoumh][hm]+|ha|he|hi|ho|ah)\b'                       # Fillers (um, uhm) & laughter
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
        # 3. Clean up formatting: multiple spaces and leading/trailing punctuation
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'^[\s,.;:?!-]+|[\s,.;:?!-]+$', '', text).strip()

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
