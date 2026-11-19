"""
Layered lesson retriever over the index built by build_lesson_index.py.

Layer 1 (answer):    static, always on.
Layer 2 (api_class): fired by an API's kind (paginated_list / write), once per API.
Layer 3 (api / app): fired by a doc lookup or call of an API; candidates are the entries tagged with that API
                     or its app, reranked against the agent's current thought; top 2-3 above the cutoff.
Layer 4 (start):     retrieved once from the instruction: hybrid dense + BM25 (RRF), then a cross-encoder
                     over the whole pool; top-k above the cutoff.

Every text the models score is "contextual": scope, app/API and trigger are prepended to the lesson, so the
scorer sees what the lesson is about (the LoCoMo date-prefix finding). Any layer can return nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Import the heavy libraries at module load, never lazily inside a task: AppWorld runs each task with time frozen
# (freezegun replaces datetime), and initialising torch/tokenizers under the fake datetime made a run balloon to
# 384 GB. Guarded so the package still imports in environments without the retrieval stack.
try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover - baseline environment
    np = BM25Okapi = CrossEncoder = SentenceTransformer = None  # type: ignore[assignment,misc]

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Models are cached per process: an AppWorld worker builds a new agent for every task.
_MODEL_CACHE: dict[tuple[str, str, str | None], Any] = {}
TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def contextual_text(entry: dict, where: str) -> str:
    return f"{where} | {entry.get('trigger', '')} | {entry['lesson']}"


@dataclass
class RetrieverConfig:
    embedder: str | None = "BAAI/bge-base-en-v1.5"  # None = no dense stage
    use_bm25: bool = True
    reranker: str | None = "mixedbread-ai/mxbai-rerank-large-v1"  # None = no reranking
    device: str | None = None
    start_k: int = 5
    start_threshold: float | None = None  # reranker score cutoff for layer 4 (None = no cutoff)
    api_k: int = 3
    api_threshold: float | None = None  # reranker score cutoff for layer 3
    class_k: int = 1  # lessons per API class (layer 2)
    # True: entries verify_lessons.py marked kind "on_error" are held back from the start / API layers and served
    # only by on_error(), when an executed step fails.
    error_triggered: bool = False
    rrf_k: int = 60
    extra: dict[str, Any] = field(default_factory=dict)


class LessonRetriever:
    def __init__(self, index: dict | str, config: RetrieverConfig | None = None):
        self.index = json.loads(open(index).read()) if isinstance(index, str) else index
        self.config = config or RetrieverConfig()
        self._embedder = None
        self._reranker = None
        self.start_entries: list[dict] = self.index["start"]
        self.start_texts = [contextual_text(e, "task") for e in self.start_entries]
        on_error = [self.config.error_triggered and e.get("kind") == "on_error" for e in self.start_entries]
        self.start_pool = [i for i, flag in enumerate(on_error) if not flag]
        self.start_error_pool = [i for i, flag in enumerate(on_error) if flag]
        self._start_vectors = None
        self._bm25 = None

    # ---------- models (lazy) ----------
    @property
    def embedder(self):
        if self._embedder is None and self.config.embedder:
            key = ("embedder", self.config.embedder, self.config.device)
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = SentenceTransformer(self.config.embedder, device=self.config.device)
            self._embedder = _MODEL_CACHE[key]
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None and self.config.reranker:
            key = ("reranker", self.config.reranker, self.config.device)
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = CrossEncoder(self.config.reranker, device=self.config.device)
            self._reranker = _MODEL_CACHE[key]
        return self._reranker

    def warm(self) -> None:
        """Load models and precompute the start-pool vectors and BM25 index. Call outside any AppWorld task."""
        if SentenceTransformer is None:
            raise ImportError("lesson retrieval needs sentence-transformers, rank_bm25 and numpy")
        if self.embedder is not None and self._start_vectors is None:
            self._start_vectors = self.embedder.encode(
                self.start_texts, normalize_embeddings=True, show_progress_bar=False
            )
        if self.config.use_bm25 and self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(t) for t in self.start_texts])
        if self.reranker is not None:
            self.reranker.predict([("warm up", "warm up")], show_progress_bar=False)

    def _query_text(self, text: str) -> str:
        return (
            (BGE_QUERY_PREFIX + text)
            if self.config.embedder and "bge" in self.config.embedder.lower()
            else text
        )

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.reranker.predict([(query, t) for t in texts], show_progress_bar=False)]

    # ---------- layer 1 ----------
    def answer_entries(self) -> list[dict]:
        return list(self.index.get("answer", []))

    # ---------- layer 4 ----------
    def first_stage_start(
        self, instruction: str, pool: list[int] | None = None, depth: int = 50
    ) -> list[int]:
        """Candidate indices into start_entries, fused by RRF (or a single ranking if one stage is off)."""
        pool = list(range(len(self.start_entries))) if pool is None else pool
        rankings: list[list[int]] = []
        if self.embedder is not None:
            if self._start_vectors is None:
                self._start_vectors = self.embedder.encode(
                    self.start_texts, normalize_embeddings=True, show_progress_bar=False
                )
            q = self.embedder.encode(
                [self._query_text(instruction)], normalize_embeddings=True, show_progress_bar=False
            )[0]
            scores = self._start_vectors[pool] @ q
            rankings.append([pool[i] for i in np.argsort(-scores)[:depth]])
        if self.config.use_bm25:
            if self._bm25 is None:
                self._bm25 = BM25Okapi([tokenize(t) for t in self.start_texts])
            all_scores = self._bm25.get_scores(tokenize(instruction))
            rankings.append(sorted(pool, key=lambda i: -all_scores[i])[:depth])
        if not rankings:
            return pool[:depth]
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.config.rrf_k + rank + 1)
        return sorted(fused, key=lambda i: -fused[i])[:depth]

    def start(self, instruction: str, pool: list[int] | None = None) -> list[tuple[dict, float]]:
        candidates = self.first_stage_start(instruction, self.start_pool if pool is None else pool)
        if self.reranker is None:
            ranked = [(i, 0.0) for i in candidates]
        else:
            scores = self.rerank(instruction, [self.start_texts[i] for i in candidates])
            ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        cutoff = self.config.start_threshold
        return [(self.start_entries[i], s) for i, s in ranked if cutoff is None or s >= cutoff][
            : self.config.start_k
        ]

    # ---------- layers 2 and 3 ----------
    def api_candidates(self, api: str) -> list[tuple[str, dict]]:
        app = api.split(".")[0]
        out = [(f"{app} API {api}", e) for e in self.index.get("api", {}).get(api, [])]
        out += [(f"{app} app", e) for e in self.index.get("app", {}).get(app, [])]
        return out

    def on_api_event(self, api: str, thought: str, shown: set[str]) -> dict[str, list[tuple[dict, float]]]:
        """Lessons to show when the agent looks up or calls `api`. `shown` holds lesson texts already shown."""
        result: dict[str, list[tuple[dict, float]]] = {"api_class": [], "api": []}
        query = f"{api}: {thought}".strip()
        for kind in self.index.get("api_classes", {}).get(api, []):
            # API-class lessons are universal procedures, so pick by support (the index is sorted by support), not
            # by similarity to the thought: reranking surfaced a support-1 variant over the support-72 lesson.
            entries = [
                e
                for e in self.index.get("api_class", {}).get(kind, [])
                if e["lesson"] not in shown and not self._held_for_errors(e)
            ]
            result["api_class"] += [(e, float(e["support"])) for e in entries[: self.config.class_k]]
        candidates = [
            (where, e)
            for where, e in self.api_candidates(api)
            if e["lesson"] not in shown and not self._held_for_errors(e)
        ]
        if candidates:
            if self.reranker is None:
                ranked = [(e, 0.0) for _, e in candidates]
            else:
                scores = self.rerank(query, [contextual_text(e, where) for where, e in candidates])
                ranked = sorted(((e, s) for (_, e), s in zip(candidates, scores)), key=lambda x: -x[1])
            cutoff = self.config.api_threshold
            result["api"] = [(e, s) for e, s in ranked if cutoff is None or s >= cutoff][: self.config.api_k]
        return result

    # ---------- error-triggered entries (error_triggered=True) ----------
    def _held_for_errors(self, entry: dict) -> bool:
        return self.config.error_triggered and entry.get("kind") == "on_error"

    def on_error(self, apis: list[str], query: str, shown: set[str], k: int = 2) -> list[tuple[dict, float]]:
        """on_error entries for a failed step: those tagged with a failing API, its app or its API class first
        (reranked against the error); if none pass the cutoff, the best start-layer on_error entry."""
        candidates: list[tuple[str, dict]] = []
        for api in apis:
            candidates += [(where, e) for where, e in self.api_candidates(api)]
            for kind in self.index.get("api_classes", {}).get(api, []):
                candidates += [(f"{kind} API", e) for e in self.index.get("api_class", {}).get(kind, [])]
        candidates = [
            (where, e) for where, e in candidates if e.get("kind") == "on_error" and e["lesson"] not in shown
        ]
        candidates = list({e["lesson"]: (where, e) for where, e in candidates}.values())
        picked: list[tuple[dict, float]] = []
        if candidates:
            if self.reranker is None:
                ranked = [(e, 0.0) for _, e in candidates]
            else:
                scores = self.rerank(query, [contextual_text(e, where) for where, e in candidates])
                ranked = sorted(((e, s) for (_, e), s in zip(candidates, scores)), key=lambda x: -x[1])
            cutoff = self.config.api_threshold
            picked = [(e, s) for e, s in ranked if cutoff is None or s >= cutoff][:k]
        if not picked and self.start_error_pool:
            pool = [i for i in self.start_error_pool if self.start_entries[i]["lesson"] not in shown]
            if pool:
                scores = (
                    self.rerank(query, [self.start_texts[i] for i in pool])
                    if self.reranker
                    else [0.0] * len(pool)
                )
                best = max(zip(pool, scores), key=lambda x: x[1])
                if self.config.start_threshold is None or best[1] >= self.config.start_threshold:
                    picked = [(self.start_entries[best[0]], best[1])]
        return picked
