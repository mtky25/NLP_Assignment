from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pprint
import threading
import os
import re
import io
from src.millionaire_client.models import Question, Option
from src.models import ExperimentConfig


class Guesser(ABC):     
    """
    Generic interface for all Guessers.
    """
    _whisper_model = None

    def __init__(self, config: ExperimentConfig, mode: str = "text"):
        if not isinstance(config, ExperimentConfig):
            raise ValueError(f"config must be a {type(ExperimentConfig)}")
        self.config = config
        self.mode = mode
        self.search_time: float = 0.0
        self.reasoning_time: float = 0.0

    @property
    def whisper_model(self):
        """Lazy loading of the Whisper model to avoid unnecessary overhead."""
        if Guesser._whisper_model is None:
            try:
                import whisper
                # Improvement #4: Use 'tiny' model for maximum speed in time-critical environments
                print("Loading Whisper 'tiny' model...")
                Guesser._whisper_model = whisper.load_model("tiny")
            except ImportError:
                print("Error: 'whisper' library not found. Please install it with 'pip install openai-whisper'.")
                raise
        return Guesser._whisper_model


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

        # Improvement #1: In-memory processing using io.BytesIO to avoid disk I/O latency
        audio_np, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # Improvement #3: Silence trimming to minimize data passed to the model
        audio_np, _ = librosa.effects.trim(audio_np)
        
        # Improvement #2: Optimized decoding pipeline (Skip language detection)
        # We force English since the quiz questions are known to be in English
        audio_padded = whisper.pad_or_trim(audio_np)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.whisper_model.device)
        
        options = whisper.DecodingOptions(language="en", fp16=False)
        result = whisper.decode(self.whisper_model, mel, options)
        
        return self._post_process_text(result.text)

    def _post_process_text(self, text: str) -> str:
        """Quick and easy filtering of speech fillers and artifacts."""
        # Regex for fillers: starts with [aoumh] and followed by one or more [hm]
        # We use a negative lookahead to avoid filtering the common word "am"
        filler_regex = r'\b(?!am\b)[aoumh][hm]+\b'
        text = re.sub(filler_regex, '', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(rf'\s+', ' ', text).strip()
        
        # Remove common Whisper artifacts like text in brackets or parentheses
        text = re.sub(rf'\[.*?\]', '', text)
        text = re.sub(rf'\(.*?\)', '', text)
        
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
            print(f"  Fetching option {letter} audio...")
            opt_audio = game_session.fetch_audio_option_next()
            
            t_opt = threading.Thread(target=transcription_task, args=(f"opt_{i}", opt_audio))
            t_opt.start()
            threads.append(t_opt)
            
            # Preserve the option IDs from the game state
            if game_session.current_question and i < len(game_session.current_question.options):
                options_ids.append(game_session.current_question.options[i].id)
            else:
                options_ids.append(i)

        # Wait for all transcription threads to complete
        print("  Waiting for transcriptions to finish...")
        for t in threads:
            t.join()

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
