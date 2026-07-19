#!/usr/bin/env python3
"""Run a fully local Zep-style LoCoMo memory experiment with a remote Qwen LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml
from openai import AsyncOpenAI

LOGGER = logging.getLogger("qwen_locomo")
CATEGORY_NAMES = {1: "Single-Hop", 2: "Multi-Hop", 3: "Temporal", 4: "Open Domain"}
REPORT_ORDER = ("Single-Hop", "Multi-Hop", "Open Domain", "Temporal")


class LocalZepClient:
    """Minimal client for the repository's local Zep Community Edition REST API."""

    def __init__(self, base_url: str) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("memory.base_url must point to a loopback/local Zep server")
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any], allow_conflict: bool = False) -> None:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "X-Zep-Skip-Processing": "true"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status >= 300:
                    raise RuntimeError(f"Local Zep returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            if not (allow_conflict and exc.code == 409):
                raise RuntimeError(f"Local Zep request failed: HTTP {exc.code} {path}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach local Zep at {self.base_url}; start legacy/docker-compose.ce.yaml"
            ) from exc

    def _delete(self, path: str) -> None:
        request = urllib.request.Request(f"{self.base_url}{path}", method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=30):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise RuntimeError(f"Local Zep delete failed: HTTP {exc.code} {path}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach local Zep at {self.base_url}") from exc

    def get_memories(self, session_id: str, count: int) -> list[str]:
        path = f"/sessions/{urllib.parse.quote(session_id)}/memory?lastn={count}"
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=30) as response:
                payload = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Failed to read memories back from local Zep: {path}") from exc
        return [message["content"] for message in payload.get("messages", [])]

    def ingest(self, session_id: str, texts: list[str]) -> int:
        session_path = f"/sessions/{urllib.parse.quote(session_id)}"
        self._delete(session_path)
        self._post("/sessions/", {"session_id": session_id, "metadata": {}}, allow_conflict=True)
        # CE accepts at most 100 messages per request. Batching also bounds request size.
        for start in range(0, len(texts), 100):
            messages = [
                {"role": "locomo", "role_type": "system", "content": text}
                for text in texts[start : start + 100]
            ]
            self._post(f"/sessions/{urllib.parse.quote(session_id)}/memory", {"messages": messages})
        return len(texts)


class LocalZepMemory:
    """Local durable memory/index used by the open-source Zep benchmark adapter.

    Text and embeddings live in SQLite. No hosted Zep SDK, API, or embedding service is used.
    Embeddings are stored as float32 blobs and are normalized at encoding time, so their dot
    product is cosine similarity.
    """

    def __init__(self, database: Path, model_path: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS memories "
            "(sample_id TEXT NOT NULL, position INTEGER NOT NULL, content TEXT NOT NULL, "
            "embedding BLOB NOT NULL, PRIMARY KEY (sample_id, position))"
        )
        self.model = SentenceTransformer(model_path, local_files_only=True)
        self.batch_size = batch_size
        LOGGER.info("[Embedding Backend]\nProvider: local\nModel: %s", model_path)

    def ingest(self, sample_id: str, texts: list[str]) -> int:
        """Construct embeddings in a batch and atomically replace one sample's memories."""
        embeddings = np.asarray(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        with self.connection:
            self.connection.execute("DELETE FROM memories WHERE sample_id = ?", (sample_id,))
            self.connection.executemany(
                "INSERT INTO memories (sample_id, position, content, embedding) VALUES (?, ?, ?, ?)",
                [
                    (sample_id, position, text, embedding.tobytes())
                    for position, (text, embedding) in enumerate(zip(texts, embeddings, strict=True))
                ],
            )
        return len(texts)

    def search(self, sample_id: str, query: str, top_k: int) -> list[str]:
        LOGGER.info("[Memory Retrieval] sample=%s query=%r top_k=%d", sample_id, query, top_k)
        rows = self.connection.execute(
            "SELECT content, embedding FROM memories WHERE sample_id = ? ORDER BY position",
            (sample_id,),
        ).fetchall()
        if not rows:
            return []
        query_embedding = np.asarray(
            self.model.encode(
                [query],
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        embeddings = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        indices = np.argsort(embeddings @ query_embedding)[::-1][:top_k]
        return [rows[int(index)][0] for index in indices]

    def close(self) -> None:
        self.connection.close()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config.get("embedding", {}).get("provider") != "local":
        raise ValueError("This adapter requires embedding.provider=local")
    if config.get("memory", {}).get("provider") != "local_zep":
        raise ValueError("This adapter requires memory.provider=local_zep")
    return config


def conversation_texts(sample: dict[str, Any]) -> list[str]:
    """Parse timestamped messages plus event and session summaries into memories."""
    conversation = sample.get("conversation", {})
    texts: list[str] = []
    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(value, list):
            continue
        timestamp = conversation.get(f"{key}_date_time", "unknown date")
        for message in value:
            content = message.get("text", "")
            if message.get("blip_captions"):
                content += f" [Image: {message['blip_captions']}]"
            texts.append(f"{timestamp} | {message.get('speaker', 'Unknown')}: {content}")
    for field in ("event_summary", "session_summary"):
        value = sample.get(field)
        if isinstance(value, dict):
            texts.extend(f"{field} {key}: {item}" for key, item in value.items())
        elif isinstance(value, list):
            texts.extend(f"{field}: {item}" for item in value)
        elif value:
            texts.append(f"{field}: {value}")
    if not texts:
        raise ValueError("LoCoMo sample contains no conversation or summary text")
    return texts


async def generate(client: AsyncOpenAI, config: dict[str, Any], prompt: str) -> str:
    llm = config["llm"]
    model = os.getenv("LLM_MODEL", llm["model"])
    LOGGER.info(
        "[LLM Backend]\nProvider: %s\nModel: %s\nBase URL: %s",
        os.getenv("LLM_PROVIDER", llm.get("provider", "openai-compatible")),
        model,
        os.environ["OPENAI_BASE_URL"],
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=llm.get("temperature", 0.0),
        max_tokens=llm.get("max_tokens", 1024),
    )
    return response.choices[0].message.content or ""


def is_correct(prediction: str, answer: Any) -> bool:
    """Compute normalized answer-containment accuracy, including Unicode answers."""
    def normalize(value: Any) -> str:
        return re.sub(r"[^\w]+", " ", str(value).lower()).strip()

    normalized_prediction = normalize(prediction)
    golds = answer if isinstance(answer, list) else [answer]
    return any(normalize(gold) in normalized_prediction for gold in golds if normalize(gold))


def print_results(predictions: list[dict[str, Any]]) -> None:
    print("\n========== LoCoMo Results ==========\n")
    for category in REPORT_ORDER:
        rows = [row for row in predictions if row["category"] == category]
        correct = sum(row["correct"] for row in rows)
        accuracy = correct / len(rows) if rows else 0.0
        print(f"{category}:\naccuracy: {accuracy:.4f} ({correct}/{len(rows)})\n")
    correct = sum(row["correct"] for row in predictions)
    overall = correct / len(predictions) if predictions else 0.0
    print(f"Overall:\naccuracy: {overall:.4f} ({correct}/{len(predictions)})")


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI-compatible Qwen API")
    if not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError("OPENAI_BASE_URL is required for the OpenAI-compatible Qwen API")
    with Path(config["dataset"]["path"]).open(encoding="utf-8") as stream:
        samples = json.load(stream)
    if not isinstance(samples, list):
        raise ValueError("LoCoMo data must be a JSON list of samples")
    samples = samples[: args.max_samples] if args.max_samples else samples

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"]
    )
    memory = LocalZepMemory(
        Path(config["memory"]["database"]),
        config["embedding"]["model"],
        config["embedding"].get("batch_size", 32),
    )
    zep = LocalZepClient(config["memory"].get("base_url", "http://127.0.0.1:8000/api/v2"))
    predictions: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    try:
        for sample_index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", sample_index))
            texts = conversation_texts(sample)
            session_id = f"locomo-qwen-{sample_id}"
            inserted_count = zep.ingest(session_id, texts)
            stored_texts = zep.get_memories(session_id, inserted_count)
            if len(stored_texts) != inserted_count:
                raise RuntimeError(
                    f"Local Zep returned {len(stored_texts)} of {inserted_count} inserted memories"
                )
            memory.ingest(sample_id, stored_texts)
            for qa_index, qa in enumerate(sample.get("qa", [])):
                category = CATEGORY_NAMES.get(qa.get("category"))
                if category is None:
                    continue
                retrieval_start = perf_counter()
                recalled = memory.search(
                    sample_id, qa["question"], config["memory"].get("top_k", 10)
                )
                retrieval_latency = perf_counter() - retrieval_start
                prompt = "Conversation memory:\n{}\n\nQuestion:\n{}\n\nAnswer:".format(
                    "\n".join(recalled), qa["question"]
                )
                generation_start = perf_counter()
                prediction = await generate(client, config, prompt)
                generation_latency = perf_counter() - generation_start
                result = {
                    "sample_id": f"{sample_id}-{qa_index}",
                    "question": qa["question"],
                    "prediction": prediction,
                    "answer": qa.get("answer", ""),
                    "category": category,
                    "correct": is_correct(prediction, qa.get("answer", "")),
                }
                predictions.append(result)
                debug_rows.append(
                    {
                        "question": qa["question"],
                        "category": category,
                        "retrieved_memories": recalled,
                        "prediction": prediction,
                        "answer": qa.get("answer", ""),
                        "inserted_memory_count": inserted_count,
                        "retrieved_memory_count": len(recalled),
                        "retrieval_latency_seconds": retrieval_latency,
                        "generation_latency_seconds": generation_latency,
                    }
                )
    finally:
        memory.close()

    output = Path(config.get("output", "outputs/qwen_locomo_predictions.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [{key: value for key, value in row.items() if key != "correct"} for row in predictions],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    debug_output = args.debug_output or config.get("debug_output")
    if debug_output:
        debug_path = Path(debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print_results(predictions)
    LOGGER.info("Saved %d predictions to %s", len(predictions), output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qwen_locomo.yaml"))
    parser.add_argument("--max-samples", type=int, help="Limit samples; use 1 for a smoke run")
    parser.add_argument("--debug-output", type=Path, help="Optional per-QA retrieval/latency JSON")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
