"""Backends without real models: Ollama over a fake HTTP session, vLLM precision selection."""

import pytest

from payo_llm.backends import Decoding, OllamaBackend, check_gpu, select_precision, vllm_engine_args
from payo_llm.records import existing_precisions

TAG = "hf.co/aisingapore/Gemma-SEA-LION-v4-4B-VL-GGUF:Q8_0"


class FakeResponse:
    def __init__(self, payload, ok=True):
        self.payload, self.ok = payload, ok

    def json(self):
        return self.payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("http error")


class FakeSession:
    """Records requests and answers like a local Ollama server."""

    def __init__(self, chat_reply):
        self.chat_reply = chat_reply
        self.posts = []

    def get(self, url, timeout):
        if url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": TAG, "digest": "sha256:abc"}]})
        return FakeResponse({"version": "0.0.test"})

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        if url.endswith("/api/show"):
            return FakeResponse({"template": "{{ .Prompt }}", "details": {"quantization_level": "Q8_0"}})
        if url.endswith("/api/chat"):
            return FakeResponse(self.chat_reply)
        return FakeResponse({})


def make(reply, num_ctx=8192):
    session = FakeSession(reply)
    return OllamaBackend(TAG, "http://localhost:11434/", num_ctx, session=session), session


def test_ollama_maps_request_and_response():
    backend, session = make({"message": {"content": "hello"}, "prompt_eval_count": 900,
                             "eval_count": 12, "done_reason": "stop"})
    assert backend.model_ref == f"{TAG}@sha256:abc" and backend.precision == "q8_0"
    [gen] = backend.generate(["prompt text"], Decoding(0.0, 1.0, 64, 42))
    assert (gen.text, gen.n_prompt_tokens, gen.n_output_tokens, gen.truncated, gen.error) == ("hello", 900, 12, False, None)
    url, payload = session.posts[-1]
    assert url.endswith("/api/chat") and payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "prompt text"}]
    assert payload["options"] == {"temperature": 0.0, "top_p": 1.0, "seed": 42, "num_predict": 64, "num_ctx": 8192}


def test_ollama_flags_truncation_and_context_overflow():
    backend, _ = make({"message": {"content": "x"}, "prompt_eval_count": 8000, "eval_count": 320,
                       "done_reason": "length"}, num_ctx=8192)
    [gen] = backend.generate(["p"], Decoding(0.0, 1.0, 320, 42))
    assert gen.truncated and gen.error == "prompt_too_long"


def test_ollama_missing_model_says_how_to_pull():
    session = FakeSession({})
    session.get = lambda url, timeout: FakeResponse({"models": []})
    with pytest.raises(LookupError, match="ollama pull"):
        OllamaBackend(TAG, "http://localhost:11434", 8192, session=session)


@pytest.mark.parametrize("requested, vram, expected", [
    ("auto", 8.0, "fp8"), ("auto", 12.0, "bf16"), ("auto", 11.0, "bf16"), ("bnb4", None, "bnb4"), ("bf16", 4.0, "bf16"),
])
def test_select_precision(requested, vram, expected):
    assert select_precision(requested, vram) == expected


def test_select_precision_rejects_unknown():
    with pytest.raises(ValueError):
        select_precision("int3", 8.0)


def test_check_gpu_accepts_8gb_ampere_and_rejects_old_or_full_cards():
    assert check_gpu((8, 6), 8.0, 7.6, 0.90) == []                       # RTX 3060 Ti / 3070, idle
    assert any("older than RTX 30" in m for m in check_gpu((7, 5), 11.0, 10.8, 0.90))   # RTX 2080 Ti
    assert any("too small" in m for m in check_gpu((8, 6), 6.0, 5.8, 0.90))              # RTX 3050 6 GB
    busy = check_gpu((8, 9), 8.0, 6.5, 0.90)                              # display holds 1.5 GB
    assert len(busy) == 1 and "--gpu-mem 0.78" in busy[0]


def test_existing_precisions(tmp_path):
    path = tmp_path / "out.jsonl"
    assert existing_precisions(path) == set()
    path.write_text('{"precision": "fp8"}\n\n{"precision": "fp8"}\n', encoding="utf-8")
    assert existing_precisions(path) == {"fp8"}


def test_vllm_engine_args_keep_study_settings_and_add_memory_options():
    args = vllm_engine_args("google/gemma-3-4b-it", "abc", "fp8", 8192, 0.85, 42, enforce_eager=True, max_num_seqs=16)
    assert (args["quantization"], args["max_model_len"], args["dtype"], args["seed"]) == ("fp8", 8192, "bfloat16", 42)
    assert (args["enforce_eager"], args["max_num_seqs"], args["gpu_memory_utilization"]) == (True, 16, 0.85)
    assert "quantization" not in vllm_engine_args("m", "r", "bf16", 8192, 0.9, 42)


def test_full_profile_sets_the_8gb_memory_options():
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "config.toml").read_text(encoding="utf-8"))
    full = cfg["profiles"]["full"]
    assert full["enforce_eager"] is True and full["max_num_seqs"] == 16 and full["max_model_len"] == 8192
