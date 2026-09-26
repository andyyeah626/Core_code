"""E5 query/passage embeddings and directional DeBERTa NLI for the shared pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class E5EncoderLike(Protocol):
    model_name: str

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class DirectionalNLI(Protocol):
    model_name: str

    def entailment(self, premise: str, hypothesis: str) -> float: ...


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if a.shape != b.shape or not a.size:
        raise ValueError("embedding dimensions differ or are empty")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("embeddings must be finite")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("embeddings must have finite nonzero norms")
    return float(np.dot(a, b) / norm)


class E5BaseV2:
    """intfloat/e5-base-v2 with the required prefixes and masked mean pooling."""

    model_name = "intfloat/e5-base-v2"

    def __init__(self, *, cache_dir: str | None = None, local_files_only: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir=cache_dir, local_files_only=local_files_only
        ).eval()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        torch = self.torch
        batch = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            embeddings = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
        pooled = (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(["query: " + str(text) for text in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(["passage: " + str(text) for text in texts])


class DebertaNLI:
    """Directional entailment with an explicitly supplied DeBERTa NLI checkpoint.

    The paper's main experiments use cross-encoder/nli-deberta-v3-xsmall.
    """

    def __init__(self, model_name: str, *, cache_dir: str | None = None,
                 local_files_only: bool = True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_name:
            raise ValueError("a DeBERTa NLI checkpoint ID is required")
        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=cache_dir, local_files_only=local_files_only
        ).eval()
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entailment_index = next((i for i, label in labels.items() if "entail" in label), None)
        if self.entailment_index is None:
            raise ValueError("NLI model must expose an entailment label")

    def entailment(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1)
        return float(probabilities[self.entailment_index].item())
