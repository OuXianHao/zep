"""SQLite-backed memory storage and local semantic retrieval for LoCoMo."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class LocalMemoryStore:
    """Persist typed memories and normalized embeddings without external services."""

    def __init__(self, database: str | Path, model_path: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                sample_id TEXT NOT NULL, memory_id TEXT NOT NULL, text TEXT NOT NULL,
                timestamp TEXT, memory_type TEXT NOT NULL, position INTEGER NOT NULL,
                embedding BLOB NOT NULL, PRIMARY KEY (sample_id, memory_id))"""
        )
        self.model = SentenceTransformer(model_path, local_files_only=True)
        self.batch_size = batch_size

    def add_memory(self, sample_id: str, text: str, timestamp: str | None = None,
                   memory_type: str = "conversation_message",
                   memory_id: str | None = None) -> str:
        memory_id = memory_id or uuid.uuid4().hex
        self.batch_add_memories(sample_id, [{"memory_id": memory_id, "text": text,
                                             "timestamp": timestamp,
                                             "memory_type": memory_type}])
        return memory_id

    def batch_add_memories(self, sample_id: str,
                           memories: Iterable[dict[str, Any]]) -> int:
        records = [{"memory_id": str(item.get("memory_id") or uuid.uuid4().hex),
                    "text": str(item["text"]), "timestamp": item.get("timestamp"),
                    "memory_type": str(item.get("memory_type", "conversation_message"))}
                   for item in memories]
        if not records:
            return 0
        embeddings = np.asarray(self.model.encode(
            [record["text"] for record in records], normalize_embeddings=True,
            batch_size=self.batch_size, show_progress_bar=False), dtype=np.float32)
        start = self.count_memories(sample_id)
        with self.connection:
            self.connection.executemany(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(sample_id, record["memory_id"], record["text"], record["timestamp"],
                  record["memory_type"], start + offset, embedding.tobytes())
                 for offset, (record, embedding) in enumerate(
                     zip(records, embeddings, strict=True))])
        return len(records)

    def get_memories(self, sample_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT sample_id, memory_id, text, timestamp, memory_type FROM memories "
            "WHERE sample_id = ? ORDER BY position", (sample_id,)).fetchall()
        return [dict(row) for row in rows]

    def reset_sample(self, sample_id: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM memories WHERE sample_id = ?", (sample_id,))

    def count_memories(self, sample_id: str) -> int:
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM memories WHERE sample_id = ?", (sample_id,)).fetchone()[0])

    def search(self, sample_id: str, query: str, top_k: int) -> list[str]:
        rows = self.connection.execute(
            "SELECT text, embedding FROM memories WHERE sample_id = ? ORDER BY position",
            (sample_id,)).fetchall()
        if not rows or top_k <= 0:
            return []
        query_embedding = np.asarray(self.model.encode(
            [query], normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=False)[0], dtype=np.float32)
        embeddings = np.stack([np.frombuffer(row["embedding"], dtype=np.float32)
                               for row in rows])
        indices = np.argsort(embeddings @ query_embedding)[::-1][:top_k]
        return [str(rows[int(index)]["text"]) for index in indices]

    def close(self) -> None:
        self.connection.close()
