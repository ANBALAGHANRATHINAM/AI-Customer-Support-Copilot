"""Replaceable pretrained instruction-model adapters for Segment 4."""

from __future__ import annotations

from typing import Protocol


DEFAULT_LLM_MODEL = "google/flan-t5-small"


class InstructionLLM(Protocol):
    """Minimal interface that keeps model providers outside response orchestration."""

    def generate(self, prompt: str) -> str:
        """Generate text from an instruction prompt."""


class HuggingFaceInstructionLLM:
    """Lazy local adapter for the pretrained instruction-tuned FLAN-T5 model."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, *, max_new_tokens: int = 48) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("LLM prompt must be a non-empty string.")
        self._load()
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        with torch.no_grad():
            output = self._model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()


class StaticAcknowledgementLLM:
    """Offline-safe adapter for deterministic tests and demonstrations only."""

    def generate(self, _: str) -> str:
        return "I understand your concern."
