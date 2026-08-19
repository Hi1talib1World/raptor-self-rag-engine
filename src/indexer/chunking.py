import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
    class ChunkMetadata(BaseModel):
        document_id: str
        source: str
        title: Optional[str] = "Untitled"
        page: Optional[int] = 1
        section: Optional[str] = "Main"
        category: Optional[str] = "General"
        language: Optional[str] = "en"
        created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Chunk(BaseModel):
        chunk_id: str
        content: str
        metadata: ChunkMetadata
        embedding: Optional[List[float]] = None
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class ChunkMetadata:
        document_id: str
        source: str
        title: Optional[str] = "Untitled"
        page: Optional[int] = 1
        section: Optional[str] = "Main"
        category: Optional[str] = "General"
        language: Optional[str] = "en"
        created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

        def model_dump(self) -> Dict[str, Any]:
            return self.__dict__

    @dataclass
    class Chunk:
        chunk_id: str
        content: str
        metadata: ChunkMetadata
        embedding: Optional[List[float]] = None

class SemanticChunker:
    """Semantic Boundary Chunker with Metadata Enrichment."""

    def __init__(self, target_chunk_size: int = 400):
        self.target_chunk_size = target_chunk_size

    def chunk_text(self, document_id: str, content: str, source_name: str = "doc.txt") -> List[Chunk]:
        boundary_pattern = r"(?=\n#{1,4}\s+|\n\n[A-Z0-9\.\s]{3,30}:)"
        parts = re.split(boundary_pattern, content)

        chunks = []
        for idx, p in enumerate(parts):
            p_strip = p.strip()
            if len(p_strip) < 15:
                continue

            meta = ChunkMetadata(
                document_id=document_id,
                source=source_name,
                title=source_name,
                page=1,
                section="General Section",
                category="Documentation",
                language="en"
            )
            cid = f"{document_id}_c_{idx}"
            chunks.append(Chunk(chunk_id=cid, content=p_strip, metadata=meta))

        return chunks
