"""Small, dependency-isolated end-to-end test for the LoCoMo adapter."""

import argparse
import asyncio
import json
import sys
import types

import numpy as np

from scripts import eval_locomo


class FakeMemory:
    inserted = []

    def __init__(self, *args):
        pass

    def ingest(self, sample_id, texts):
        self.inserted.extend(texts)
        return len(texts)

    def search(self, sample_id, query, top_k):
        assert query == "Where did Alice go?"
        return ["Alice: I went to Paris."]

    def close(self):
        pass


class FakeLocalZep:
    inserted = []

    def __init__(self, *args):
        pass

    def ingest(self, session_id, texts):
        self.inserted.extend(texts)
        return len(texts)

    def get_memories(self, session_id, count):
        return self.inserted[-count:]


class FakeCompletions:
    async def create(self, **kwargs):
        assert kwargs["model"] == "Qwen3-30B-A3B-Instruct-2507"
        assert "Alice: I went to Paris." in kwargs["messages"][0]["content"]
        message = type("Message", (), {"content": "Paris"})()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_local_memory_ingest_and_cosine_retrieval(tmp_path, monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_path, local_files_only):
            assert local_files_only is True

        def encode(self, texts, normalize_embeddings, batch_size, show_progress_bar):
            assert normalize_embeddings is True
            assert batch_size == 2
            vectors = {
                "Paris memory": [1.0, 0.0],
                "Tokyo memory": [0.0, 1.0],
                "Where was the Paris trip?": [1.0, 0.0],
            }
            return np.asarray([vectors[text] for text in texts], dtype=np.float32)

    module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    memory = eval_locomo.LocalZepMemory(tmp_path / "memory.sqlite3", "/local/model", 2)
    assert memory.ingest("sample", ["Paris memory", "Tokyo memory"]) == 2
    assert memory.search("sample", "Where was the Paris trip?", 1) == ["Paris memory"]
    memory.close()


def test_one_sample_pipeline(tmp_path, monkeypatch):
    data = [
        {
            "conversation": {
                "session_1": [{"speaker": "Alice", "text": "I went to Paris."}],
                "session_1_date_time": "1:00 PM on 1 January, 2024",
            },
            "event_summary": {"session_1": "Alice visited Paris"},
            "session_summary": ["A trip was discussed"],
            "qa": [{"question": "Where did Alice go?", "answer": "Paris", "category": 1}],
        }
    ]
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    config = {
        "llm": {"provider": "qwen", "model": "Qwen3-30B-A3B-Instruct-2507"},
        "embedding": {"provider": "local", "model": "/local/model"},
        "memory": {
            "provider": "local_zep",
            "base_url": "http://127.0.0.1:8000/api/v2",
            "database": str(tmp_path / "memory.sqlite3"),
            "top_k": 1,
        },
        "dataset": {"path": str(tmp_path / "data.json")},
        "output": str(tmp_path / "predictions.json"),
        "debug_output": str(tmp_path / "debug.json"),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(eval_locomo.yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://qwen.example/v1")
    monkeypatch.setattr(eval_locomo, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(eval_locomo, "LocalZepMemory", FakeMemory)
    monkeypatch.setattr(eval_locomo, "LocalZepClient", FakeLocalZep)

    asyncio.run(
        eval_locomo.run(
            argparse.Namespace(config=config_path, max_samples=1, debug_output=None)
        )
    )

    result = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))
    assert result == [
        {
            "sample_id": "0-0",
            "question": "Where did Alice go?",
            "prediction": "Paris",
            "answer": "Paris",
            "category": "Single-Hop",
        }
    ]
    assert any("Alice: I went to Paris." in item for item in FakeMemory.inserted)
    assert any("Alice: I went to Paris." in item for item in FakeLocalZep.inserted)
    debug = json.loads((tmp_path / "debug.json").read_text(encoding="utf-8"))
    assert debug[0]["retrieved_memories"] == ["Alice: I went to Paris."]
    assert debug[0]["inserted_memory_count"] == 3
    assert debug[0]["retrieved_memory_count"] == 1
    assert debug[0]["retrieval_latency_seconds"] >= 0
    assert debug[0]["generation_latency_seconds"] >= 0
