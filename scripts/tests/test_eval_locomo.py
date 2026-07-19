"""Unit and smoke tests for the pure-Python LoCoMo adapter."""

import argparse
import asyncio
import json
import sys
import types

import numpy as np

sys.path.insert(0, "scripts")
import eval_locomo  # noqa: E402
from memory_store import LocalMemoryStore  # noqa: E402


class FakeSentenceTransformer:
    def __init__(self, model_path, local_files_only):
        assert local_files_only is True

    def encode(self, texts, normalize_embeddings, batch_size, show_progress_bar):
        assert normalize_embeddings is True
        vectors = {
            "Paris memory": [1.0, 0.0], "Tokyo memory": [0.0, 1.0],
            "Where was the Paris trip?": [1.0, 0.0], "hello": [0.5, 0.5],
        }
        return np.asarray([vectors.get(text, [1.0, 0.0]) for text in texts], dtype=np.float32)


def install_fake_embeddings(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))


def test_local_memory_store_insertion(tmp_path, monkeypatch):
    install_fake_embeddings(monkeypatch)
    store = LocalMemoryStore(tmp_path / "memory.sqlite3", "/local/model", 2)
    memory_id = store.add_memory("sample", "hello", "2024-01-01", "image_caption")
    assert store.count_memories("sample") == 1
    assert store.get_memories("sample") == [{"sample_id": "sample", "memory_id": memory_id,
        "text": "hello", "timestamp": "2024-01-01", "memory_type": "image_caption"}]
    store.reset_sample("sample")
    assert store.count_memories("sample") == 0
    store.close()


def test_retrieval_with_fake_embeddings(tmp_path, monkeypatch):
    install_fake_embeddings(monkeypatch)
    store = LocalMemoryStore(tmp_path / "memory.sqlite3", "/local/model", 2)
    assert store.batch_add_memories("sample", [{"text": "Paris memory"},
                                                {"text": "Tokyo memory"}]) == 2
    assert store.search("sample", "Where was the Paris trip?", 1) == ["Paris memory"]
    store.close()


class FakeMemory:
    inserted = []
    def __init__(self, *args): pass
    def reset_sample(self, sample_id): self.inserted.clear()
    def batch_add_memories(self, sample_id, memories):
        self.inserted.extend(item["text"] for item in memories)
        return len(memories)
    def search(self, sample_id, query, top_k): return ["Alice: I went to Paris."]
    def close(self): pass


class FakeCompletions:
    async def create(self, **kwargs):
        assert kwargs["model"] == "Qwen3-30B-A3B-Instruct-2507"
        assert "Alice: I went to Paris." in kwargs["messages"][0]["content"]
        message = type("Message", (), {"content": "Paris"})()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_one_sample_pipeline(tmp_path, monkeypatch):
    data = [{"conversation": {"session_1": [{"speaker": "Alice", "text": "I went to Paris.",
             "blip_captions": "Eiffel Tower"}], "session_1_date_time": "2024-01-01"},
             "event_summary": {"session_1": "Alice visited Paris"},
             "session_summary": ["A trip was discussed"],
             "qa": [{"question": "Where did Alice go?", "answer": "Paris", "category": 1}]}]
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    config = {"llm": {"provider": "qwen", "model": "Qwen3-30B-A3B-Instruct-2507"},
              "embedding": {"provider": "local", "model": "/local/model"},
              "memory": {"backend": "sqlite", "database": str(tmp_path / "memory.sqlite3")},
              "retrieval": {"top_k": 1}, "dataset": {"path": str(tmp_path / "data.json")},
              "output": str(tmp_path / "predictions.json"),
              "debug_output": str(tmp_path / "debug.json")}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(eval_locomo.yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://qwen.example/v1")
    monkeypatch.setattr(eval_locomo, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(eval_locomo, "LocalMemoryStore", FakeMemory)
    asyncio.run(eval_locomo.run(argparse.Namespace(config=config_path, max_samples=1,
                                                    debug_output=None)))
    result = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))
    assert result[0]["prediction"] == "Paris"
    assert "correct" not in result[0]
    assert any("[Image: Eiffel Tower]" in item for item in FakeMemory.inserted)
    debug = json.loads((tmp_path / "debug.json").read_text(encoding="utf-8"))[0]
    assert debug["retrieved_memories"] == ["Alice: I went to Paris."]
    assert debug["inserted_memory_count"] == 3
    assert debug["retrieved_memory_count"] == 1
    assert debug["retrieval_latency"] >= 0
    assert debug["generation_latency"] >= 0
