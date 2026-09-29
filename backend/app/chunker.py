from typing import List

from .schemas import CorpusDocument, DocumentChunk


class DocumentChunker:
    """Split corpus documents into deterministic, overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document: CorpusDocument,
    ) -> List[DocumentChunk]:
        """Create chunks for one document."""

        words = document.text.split()

        if not words:
            return []

        chunks: List[DocumentChunk] = []

        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(words):
            end = min(start + self.chunk_size, len(words))

            chunk_text = " ".join(words[start:end])

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.doc_id}_chunk_{chunk_index:03d}",
                    doc_id=document.doc_id,
                    title=document.title,
                    url=document.url,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    approx_tokens=len(chunk_text.split()),
                )
            )

            chunk_index += 1

            if end >= len(words):
                break

            start += step

        return chunks

    def chunk_documents(
        self,
        documents: List[CorpusDocument],
    ) -> List[DocumentChunk]:
        """Create chunks for all documents."""

        chunks: List[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.chunk_document(document))

        return chunks