import json
from pathlib import Path
from typing import List

from .chunker import DocumentChunker
from .schemas import DocumentChunk


class CorpusIngestion:
    """Create and persist searchable chunks from the corpus."""

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build_chunks(self) -> List[DocumentChunk]:
        """Load the corpus and generate all document chunks."""

        from .data_loader import DatasetLoader

        loader = DatasetLoader(self.raw_dir)

        documents = loader.load_corpus()

        chunks = self.chunker.chunk_documents(documents)

        return chunks

    def save_chunks(
        self,
        chunks: List[DocumentChunk],
        filename: str = "chunks.jsonl",
    ) -> Path:
        """Persist chunks as JSONL."""

        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.processed_dir / filename

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for chunk in chunks:
                file.write(
                    json.dumps(
                        chunk.model_dump(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return output_path

    def run(self) -> Path:
        """Run the complete corpus ingestion process."""

        chunks = self.build_chunks()

        output_path = self.save_chunks(chunks)

        print(f"Documents processed successfully.")
        print(f"Chunks created: {len(chunks)}")
        print(f"Output: {output_path}")

        return output_path