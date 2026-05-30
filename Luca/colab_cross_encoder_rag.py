# !pip install torch sentence-transformers wikipedia-api requests librosa openai-whisper pandas openpyxl python-dotenv

import torch
import os
import time
import re
import io
import threading
import pprint
import dataclasses
import numpy as np
import pandas as pd
import requests
import librosa
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
from collections import defaultdict
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import wikipediaapi
from dotenv import load_dotenv

# ==============================================================================
# 1. MILLIONAIRE CLIENT (Models & Exceptions)
# ==============================================================================

class MillionaireError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
    def __str__(self):
        return f"[{self.status_code}] {self.message}" if self.status_code else self.message

class AuthenticationError(MillionaireError): pass
class GameError(MillionaireError): pass
class TimeoutError(MillionaireError): pass
class ValidationError(MillionaireError): pass
class NotFoundError(MillionaireError): pass
class ServerError(MillionaireError): pass
class RateLimitError(MillionaireError): pass

class GameStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class Option:
    id: int
    text: Optional[str] = None
    @classmethod
    def from_dict(cls, data: dict) -> "Option":
        return cls(id=data["id"], text=data.get("text"))

@dataclass
class Question:
    id: int
    text: Optional[str] = None
    options: List[Option] = field(default_factory=list)
    level: int = 0
    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(id=data["id"], text=data.get("text"),
                   options=[Option.from_dict(opt) for opt in data.get("options", [])],
                   level=data.get("level", 0))

@dataclass
class MoneyLevel:
    level: int
    amount: float
    @classmethod
    def from_dict(cls, data: dict) -> "MoneyLevel":
        return cls(level=data["level"], amount=data["amount"])

@dataclass
class Competition:
    id: int
    name: str
    description: Optional[str] = None
    max_levels: int = 15
    is_infinite: bool = False
    @classmethod
    def from_dict(cls, data: dict) -> "Competition":
        return cls(id=data["id"], name=data["name"], description=data.get("description"),
                   max_levels=data.get("maxLevels", 15), is_infinite=data.get("isInfinite", False))

@dataclass
class GameState:
    session_id: int
    competition: Competition
    status: GameStatus
    earned_amount: float
    current_level: int
    money_pyramid: List[MoneyLevel]
    question_deadline: Optional[datetime] = None
    question: Optional[Question] = None
    mode: str = "text"

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        deadline = data.get("questionDeadline")
        if deadline:
            try: deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except: deadline = None
        return cls(
            session_id=data.get("sessionId", data.get("id", 0)),
            competition=Competition.from_dict(data["competition"]),
            status=GameStatus(data.get("status", "in_progress")),
            earned_amount=data.get("earnedAmount", 0),
            current_level=data.get("currentLevel", 1),
            money_pyramid=[MoneyLevel.from_dict(ml) for ml in data.get("moneyPyramid", [])],
            question_deadline=deadline,
            question=Question.from_dict(data["question"]) if data.get("question") else None,
            mode=data.get("mode", "text"),
        )
    @property
    def in_progress(self) -> bool: return self.status == GameStatus.IN_PROGRESS
    @property
    def is_game_over(self) -> bool: return self.status in (GameStatus.COMPLETED, GameStatus.FAILED, GameStatus.TIMEOUT)

@dataclass
class AnswerResult:
    correct: Optional[bool] = None
    game_over: bool = False
    earned_amount: float = 0
    timed_out: bool = False
    status: Optional[str] = None
    current_level: Optional[int] = None
    question_deadline: Optional[datetime] = None
    question: Optional[Question] = None
    money_pyramid: List[MoneyLevel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerResult":
        deadline = data.get("questionDeadline")
        if deadline:
            try: deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except: deadline = None
        return cls(
            correct=data.get("correct"),
            game_over=data.get("gameOver", False),
            earned_amount=data.get("earnedAmount", 0),
            timed_out=data.get("timedOut", False),
            status=data.get("status"),
            current_level=data.get("currentLevel"),
            question_deadline=deadline,
            question=Question.from_dict(data["question"]) if data.get("question") else None,
            money_pyramid=[MoneyLevel.from_dict(ml) for ml in data.get("moneyPyramid", [])]
        )

# ==============================================================================
# 2. MILLIONAIRE CLIENT (Logic Modules)
# ==============================================================================

class BaseClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
    
    def set_auth_cookie(self, cookie_value: str):
        self._session.cookies.set("polimillionaire_auth", cookie_value)
    
    def is_authenticated(self) -> bool:
        return "polimillionaire_auth" in self._session.cookies
    
    def request(self, method: str, endpoint: str, data=None, params=None, auth_required=True, raw=False) -> Any:
        if auth_required and not self.is_authenticated():
            raise AuthenticationError("Authentication required.")
        url = urljoin(f"{self.base_url}/", endpoint.lstrip("/"))
        try:
            resp = self._session.request(method, url, json=data, params=params, timeout=self.timeout)
            if raw and resp.status_code < 400: return resp.content
            
            try: data = resp.json() if resp.text else {}
            except: data = {}
            
            if resp.status_code in (200, 201, 204): return data
            msg = data.get("message", data.get("error", f"HTTP {resp.status_code}"))
            if resp.status_code == 401: raise AuthenticationError(msg)
            if resp.status_code == 404: raise NotFoundError(msg)
            raise MillionaireError(msg, resp.status_code)
        except requests.Timeout: raise MillionaireError("Timeout")
        except requests.ConnectionError: raise MillionaireError("Connection Error")

class GameSession:
    def __init__(self, client: BaseClient, state: GameState):
        self._client = client
        self._state = state
    @property
    def session_id(self) -> int: return self._state.session_id
    @property
    def in_progress(self) -> bool: return self._state.in_progress
    @property
    def current_question(self) -> Optional[Question]: return self._state.question
    @property
    def current_level(self) -> int: return self._state.current_level
    @property
    def mode(self) -> str: return self._state.mode
    @property
    def state(self) -> GameState: return self._state

    def fetch_audio_question(self) -> bytes:
        return self._client.request("GET", f"/api/game/{self.session_id}/audio/question", raw=True)
    def fetch_audio_option_next(self) -> bytes:
        return self._client.request("GET", f"/api/game/{self.session_id}/audio/option/next", raw=True)
    
    def answer(self, option_id: int) -> AnswerResult:
        res_data = self._client.request("POST", f"/api/game/{self.session_id}/answer", data={"optionId": option_id})
        res = AnswerResult.from_dict(res_data)
        if res.question:
            self._state = GameState.from_dict({
                "sessionId": self.session_id, 
                "competition": self._state.competition.__dict__,
                "status": "in_progress", 
                "earnedAmount": res.earned_amount,
                "currentLevel": res.current_level or self._state.current_level,
                "moneyPyramid": [ml.__dict__ for ml in res.money_pyramid] if res.money_pyramid else [],
                "questionDeadline": res.question_deadline.isoformat() if res.question_deadline else None,
                "mode": self._state.mode, 
                "question": {
                    "id": res.question.id, "text": res.question.text,
                    "options": [{"id": o.id, "text": o.text} for o in res.question.options]
                }
            })
        elif res.game_over:
            if res.timed_out:
                self._state.status = GameStatus.TIMEOUT
            else:
                self._state.status = GameStatus.COMPLETED if res.correct else GameStatus.FAILED
            self._state.earned_amount = res.earned_amount
            self._state.question = None
        return res

class MillionaireClient:
    def __init__(self, base_url: str):
        self._base = BaseClient(base_url)
    def login(self, username, password):
        resp = self._base.request("POST", "/api/auth/login", data={"username": username, "password": password}, auth_required=False)
        return resp
    @property
    def game(self):
        class GameModule:
            def __init__(self, base): self._base = base
            def start(self, competition_id, mode="text"):
                resp = self._base.request("POST", "/api/game/start", data={"competitionId": competition_id, "mode": mode})
                return GameSession(self._base, GameState.from_dict(resp))
        return GameModule(self._base)

# ==============================================================================
# 3. PROJECT MODELS & CONFIG
# ==============================================================================

class ApproachType(str, Enum):
    DIRECT_LLM = "direct_llm"
    RAG = "rag"
    HYBRID = "hybrid"

class QuestionOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class ExperimentConfig:
    experiment_id: str = field(init=False, default_factory=lambda: uuid4().hex)
    username: str = ""
    notes: str = ""
    approach: Optional[ApproachType] = None
    inference_model: str = ""
    inference_model_size: str = ""
    is_rag: bool = False
    embedding_model: Optional[str] = None
    embedding_model_size: Optional[str] = None
    # Metrics
    mean_question_accuracy: float = 0.0
    mean_time: float = 0.0
    mean_search_time: float = 0.0
    mean_reasoning_time: float = 0.0
    mean_transcription_time: float = 0.0

@dataclass
class QuestionResult:
    theme: str = ""
    question_outcome: QuestionOutcome = QuestionOutcome.ERROR
    answer_time: float = 0.0
    search_time: float = 0.0
    reasoning_time: float = 0.0
    transcription_time: float = 0.0
    level: int = 0

# ==============================================================================
# 4. BASE GUESSER
# ==============================================================================

class Guesser(ABC):
    _whisper_models = {}
    _whisper_lock = threading.Lock()

    def __init__(self, config: ExperimentConfig, mode: str = "text", transcription_model: str = "tiny"):
        self.config = config
        self.mode = mode
        self.transcription_model_size = transcription_model
        self.search_time: float = 0.0
        self.reasoning_time: float = 0.0
        self.transcription_time: float = 0.0

    def preload(self):
        if self.mode == "speech": _ = self.whisper_model

    @property
    def whisper_model(self):
        size = self.transcription_model_size
        if size not in Guesser._whisper_models:
            with Guesser._whisper_lock:
                if size not in Guesser._whisper_models:
                    import whisper
                    Guesser._whisper_models[size] = whisper.load_model(size)
        return Guesser._whisper_models[size]

    @abstractmethod
    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int: pass

    def _transcribe_audio(self, audio_data: bytes) -> str:
        import whisper
        audio_np, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        audio_np, _ = librosa.effects.trim(audio_np)
        if audio_np.size == 0: return ""
        audio_padded = whisper.pad_or_trim(audio_np)
        with Guesser._whisper_lock:
            mel = whisper.log_mel_spectrogram(audio_padded).to(self.whisper_model.device)
            options = whisper.DecodingOptions(language="en", fp16=False)
            result = whisper.decode(self.whisper_model, mel, options)
            return self._post_process_text(result.text)

    def _post_process_text(self, text: str) -> str:
        text = text.strip()
        if (text.startswith('(') and text.endswith(')')) or (text.startswith('[') and text.endswith(']')):
            text = text[1:-1]
        text = re.sub(r'[\[\(].*?[\]\)]', '', text)
        patterns = [
            r'(\b\w+)\b([- ]?\1\b){2,}',
            r'^(?:options?|[tp]op\w*).*?\b(?:[abcd]|see|sea)\b',
            r'\b(?:options?|topst?ion|topson|topption|pops|topsynd)\w*\b(?:\s+(?:and\s+)?\w+)?',
            r'\b(?:(?!am\b)[aoumh][hm]+|ha|he|hi|ho|ah)\b'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'^[\s,.;:?!-]+|[\s,.;:?!-]+$', '', text).strip()

    def get_speech_question(self, game_session: Any) -> Question:
        results = {}
        threads = []
        def transcription_task(key, audio_bytes): results[key] = self._transcribe_audio(audio_bytes)

        q_audio = game_session.fetch_audio_question()
        t_q = threading.Thread(target=transcription_task, args=("q", q_audio))
        t_q.start(); threads.append(t_q)

        options_ids = []
        for i in range(4):
            opt_audio = game_session.fetch_audio_option_next()
            t_opt = threading.Thread(target=transcription_task, args=(f"opt_{i}", opt_audio))
            t_opt.start(); threads.append(t_opt)
            options_ids.append(game_session.current_question.options[i].id if game_session.current_question else i)

        start_track = time.time()
        for t in threads: t.join()
        self.transcription_time = time.time() - start_track

        transcribed_options = [Option(id=options_ids[i], text=results.get(f"opt_{i}", "")) for i in range(4)]
        return Question(id=0, text=results.get("q", ""), options=transcribed_options)

# ==============================================================================
# 5. RAG GUESSER IMPLEMENTATION
# ==============================================================================

class CrossEncoderRAGBaseline(Guesser):
    def __init__(self, config: ExperimentConfig, mode="text"):
        super().__init__(config, mode=mode)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(config.embedding_model, device=device)
        self.cross_encoder = CrossEncoder(config.inference_model, device=device)

    def infer_answer(self, question: Question, theme: str, game_session: Any = None) -> int:
        self.search_time = 0.0
        start_time = time.time()
        wiki = wikipediaapi.Wikipedia(user_agent='Polimillionaire', language='en')
        
        search_start = time.time()
        candidate_sections = self._get_sections(wiki, question)
        self.search_time = time.time() - search_start

        if not candidate_sections:
            scores = self.cross_encoder.predict([(question.text, o.text) for o in question.options], activation_fn=torch.nn.Sigmoid())
            best_idx = np.argmax(scores)
        else:
            # Step 1: Retrieval
            question_emb = self.bi_encoder.encode(question.text, convert_to_tensor=True)
            sec_embs = self.bi_encoder.encode(candidate_sections, convert_to_tensor=True)
            cos_scores = util.cos_sim(question_emb, sec_embs)[0]
            top_k = min(10, len(candidate_sections))
            top_indices = torch.topk(cos_scores, k=top_k).indices.cpu().numpy()
            top_sections = [candidate_sections[i] for i in top_indices]

            # Step 2: Reranking
            all_pairs = []
            for opt in question.options:
                for sec in top_sections:
                    all_pairs.append((question.text, f"{opt.text}. {sec}"))
            
            all_scores = self.cross_encoder.predict(all_pairs, activation_fn=torch.nn.Sigmoid(), batch_size=32)
            all_scores = all_scores.reshape(len(question.options), len(top_sections))
            opt_max_scores = np.max(all_scores, axis=1)
            best_idx = np.argmax(opt_max_scores)

        self.reasoning_time = time.time() - start_time - self.search_time
        return question.options[best_idx].id

    def _get_sections(self, wiki, question):
        sections = set()
        for query in [question.text] + [f"{o.text} {question.text}" for o in question.options[:2]]:
            try:
                page_results = wiki.search(query)
                if page_results.pages:
                    title = list(page_results.pages.keys())[0]
                    p = page_results.pages[title]
                    sections.add(p.summary)
                    for s in p.sections[:3]:
                        if s.text.strip(): sections.add(s.text)
            except: pass
        return list(sections)

# ==============================================================================
# 6. BENCHMARK & GAME ENGINE
# ==============================================================================

class Game:
    def __init__(self, client, guesser, competition_id=1):
        self.client, self.guesser, self.competition_id = client, guesser, competition_id
        self.results = []

    def play_game(self):
        mode = self.guesser.mode
        self.guesser.preload()
        session = self.client.game.start(self.competition_id, mode=mode)
        
        while session.in_progress:
            question = session.current_question
            trans_time = 0.0
            if mode == "speech":
                question = self.guesser.get_speech_question(session)
                trans_time = self.guesser.transcription_time
            
            start_t = time.time()
            try:
                ans_id = self.guesser.infer_answer(question, session.state.competition.name, game_session=session)
                outcome = session.answer(ans_id)
                status = QuestionOutcome.CORRECT if outcome.correct else QuestionOutcome.INCORRECT
                if outcome.timed_out: status = QuestionOutcome.TIMEOUT
            except Exception as e:
                print(f"Error: {e}"); status = QuestionOutcome.ERROR
            
            duration = time.time() - start_t
            self.results.append(QuestionResult(
                theme=session.state.competition.name, question_outcome=status,
                answer_time=duration, search_time=self.guesser.search_time,
                reasoning_time=self.guesser.reasoning_time, transcription_time=trans_time,
                level=session.current_level
            ))
            if status == QuestionOutcome.ERROR or not session.in_progress: break

class Benchmark:
    def __init__(self, experiment, guesser, client):
        self.experiment, self.guesser, self.client = experiment, guesser, client

    def run(self, times_per_competition=1, filename="results.xlsx"):
        all_results = []
        for comp_id in [0, 1, 2]: # Example: first 3 comps
            for _ in range(times_per_competition):
                g = Game(self.client, self.guesser, competition_id=comp_id)
                g.play_game()
                all_results.extend(g.results)
        
        # Simple stats calculation
        if all_results:
            self.experiment.mean_question_accuracy = sum(1 for r in all_results if r.question_outcome == QuestionOutcome.CORRECT) / len(all_results)
            self.experiment.mean_time = sum(r.answer_time for r in all_results) / len(all_results)
            print(f"Benchmark Finished. Accuracy: {self.experiment.mean_question_accuracy:.2%}")
            pd.DataFrame([dataclasses.asdict(self.experiment)]).to_excel(filename)

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

def main():
    load_dotenv()
    API_URL = "http://131.175.15.22:51111/"
    USERNAME = os.getenv("POLI_USERNAME", "")
    PASSWORD = os.getenv("POLI_PASSWORD", "")
    
    client = MillionaireClient(API_URL)
    try: client.login(USERNAME, PASSWORD)
    except Exception as e: print(f"Login failed: {e}"); return

    config = ExperimentConfig(
        username="Luca",
        notes="Bi-Encoder & Cross-Encoder RAG",
        approach=ApproachType.RAG,
        is_rag=True,
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_model_size=0.1, # 0.1B
        inference_model="cross-encoder/ms-marco-MiniLM-L12-v2",
        inference_model_size=0.033, # 33.4M
    )
    
    # mode="speech" or "text"
    guesser = CrossEncoderRAGBaseline(config, mode="text")
    benchmark = Benchmark(config, guesser, client)
    benchmark.run(times_per_competition=1, filename="colab_results.xlsx")

if __name__ == "__main__":
    main()
