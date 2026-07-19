#!/usr/bin/env python3
"""Run a pure-Python SQLite LoCoMo experiment with a remote Qwen LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from openai import AsyncOpenAI

from memory_store import LocalMemoryStore

LOGGER = logging.getLogger("qwen_locomo")
CATEGORY_NAMES = {1: "Single-Hop", 2: "Multi-Hop", 3: "Temporal", 4: "Open Domain"}
REPORT_ORDER = ("Single-Hop", "Multi-Hop", "Open Domain", "Temporal")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config.get("embedding", {}).get("provider") != "local":
        raise ValueError("This adapter requires embedding.provider=local")
    if config.get("memory", {}).get("backend") != "sqlite":
        raise ValueError("This adapter requires memory.backend=sqlite")
    return config


def conversation_memories(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse timestamped messages plus event and session summaries into memories."""
    conversation = sample.get("conversation", {})
    memories: list[dict[str, Any]] = []
    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(value, list):
            continue
        timestamp = conversation.get(f"{key}_date_time", "unknown date")
        for message in value:
            content = message.get("text", "")
            if message.get("blip_captions"):
                content += f" [Image: {message['blip_captions']}]"
            memories.append({
                "text": f"{timestamp} | {message.get('speaker', 'Unknown')}: {content}",
                "timestamp": timestamp,
                "memory_type": "conversation_message",
            })
    for field in ("event_summary", "session_summary"):
        value = sample.get(field)
        if isinstance(value, dict):
            memories.extend({"text": f"{field} {key}: {item}", "timestamp": None,
                             "memory_type": field} for key, item in value.items())
        elif isinstance(value, list):
            memories.extend({"text": f"{field}: {item}", "timestamp": None,
                             "memory_type": field} for item in value)
        elif value:
            memories.append({"text": f"{field}: {value}", "timestamp": None,
                             "memory_type": field})
    if not memories:
        raise ValueError("LoCoMo sample contains no conversation or summary text")
    return memories


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
    memory = LocalMemoryStore(
        Path(config["memory"]["database"]),
        config["embedding"]["model"],
        config["embedding"].get("batch_size", 32),
    )
    LOGGER.info("[Embedding Backend]\nProvider: local\nModel: %s", config["embedding"]["model"])
    predictions: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    try:
        for sample_index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", sample_index))
            memories = conversation_memories(sample)
            memory.reset_sample(sample_id)
            inserted_count = memory.batch_add_memories(sample_id, memories)
            for qa_index, qa in enumerate(sample.get("qa", [])):
                category = CATEGORY_NAMES.get(qa.get("category"))
                if category is None:
                    continue
                retrieval_start = perf_counter()
                recalled = memory.search(
                    sample_id, qa["question"], config.get("retrieval", {}).get("top_k", 10)
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
                        "retrieval_latency": retrieval_latency,
                        "generation_latency": generation_latency,
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
