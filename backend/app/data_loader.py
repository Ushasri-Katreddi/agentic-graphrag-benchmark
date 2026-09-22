import json
from pathlib import Path
from typing import Iterator, List

from .schemas import (
    CorpusDocument,
    HiddenEvaluationQuestion,
    PublicEvaluationQuestion,
)


class DatasetLoader:
    """Loads hackathon datasets from JSONL files."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _read_jsonl(self, filename: str) -> Iterator[dict]:
        """Read a JSONL file one record at a time."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {file_path} at line {line_number}"
                    ) from exc

    def load_corpus(self) -> List[CorpusDocument]:
        """Load all corpus documents."""
        return [
            CorpusDocument.model_validate(record)
            for record in self._read_jsonl("corpus.jsonl")
        ]

    def load_public_questions(self) -> List[PublicEvaluationQuestion]:
        """Load public benchmark questions."""
        return [
            PublicEvaluationQuestion.model_validate(record)
            for record in self._read_jsonl("eval_public.jsonl")
        ]

    def load_hidden_questions(self) -> List[HiddenEvaluationQuestion]:
        """Load hidden benchmark questions."""
        return [
            HiddenEvaluationQuestion.model_validate(record)
            for record in self._read_jsonl("eval_hidden.jsonl")
        ]

    def corpus_count(self) -> int:
        """Return number of corpus documents."""
        return sum(1 for _ in self._read_jsonl("corpus.jsonl"))

    def public_question_count(self) -> int:
        """Return number of public questions."""
        return sum(1 for _ in self._read_jsonl("eval_public.jsonl"))

    def hidden_question_count(self) -> int:
        """Return number of hidden questions."""
        return sum(1 for _ in self._read_jsonl("eval_hidden.jsonl"))