"""
Generation backends.

- OllamaBackend: pilot on macOS (GGUF weights through a local Ollama server).
- VLLMBackend: full run on a Linux CUDA GPU (Hugging Face weights at a pinned revision).

Both send each prompt as one user turn and let the backend apply the model's own chat template.
"""

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import requests

BF16_MIN_VRAM_GB = 11.0  # 4B weights are 8.6 GB in bf16, plus KV cache and activations
PRECISIONS = ("bf16", "fp8", "bnb4")


@dataclass(frozen=True)
class Decoding:
    """Decoding settings for one batch of requests."""

    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens, "seed": self.seed}


@dataclass(frozen=True)
class Generation:
    """One model output, with token counts as reported by the backend."""

    text: str
    n_prompt_tokens: int | None
    n_output_tokens: int | None
    finish_reason: str | None
    error: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class Backend(Protocol):
    """What run.py needs from a backend."""

    name: str
    model_ref: str
    precision: str
    chat_template: str | None

    def generate(self, prompts: Sequence[str], decoding: Decoding) -> list[Generation]: ...

    def close(self) -> None: ...

    def versions(self) -> dict[str, str]: ...


def select_precision(requested: str, vram_gb: float | None) -> str:
    """
    Resolve the precision for a vLLM run. One value is used for both models.

    Args:
        requested: "auto" or one of PRECISIONS.
        vram_gb: Total VRAM of GPU 0, needed only for "auto".

    Returns:
        "bf16" when the GPU has room for full-precision weights, otherwise "fp8"
        (weight-only FP8, which vLLM runs on Ampere through its Marlin kernel).
    """
    if requested != "auto":
        if requested not in PRECISIONS:
            raise ValueError(f"precision must be auto or one of {PRECISIONS}, got {requested!r}")
        return requested
    if vram_gb is None:
        raise ValueError("precision=auto needs the GPU's VRAM")
    return "bf16" if vram_gb >= BF16_MIN_VRAM_GB else "fp8"


class OllamaBackend:
    """Local Ollama server, one request at a time (pilot)."""

    name = "ollama"

    def __init__(self, tag: str, host: str, num_ctx: int, session: requests.Session | None = None) -> None:
        self.tag = tag
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.http = session or requests.Session()
        self.digest = self._digest()
        show = self._post("/api/show", {"model": tag})
        self.chat_template: str | None = show.get("template")
        self.precision = show.get("details", {}).get("quantization_level", "unknown").lower()
        self.model_ref = f"{tag}@{self.digest}"

    def _post(self, path: str, payload: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        response = self.http.post(self.host + path, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _digest(self) -> str:
        """Find the pulled model's digest, so the exact weights are logged."""
        try:
            response = self.http.get(self.host + "/api/tags", timeout=10)
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.host}. Open the Ollama app or run `ollama serve`.") from exc
        response.raise_for_status()
        for model in response.json().get("models", []):
            if model.get("name") == self.tag or model.get("model") == self.tag:
                return model.get("digest", "unknown")
        raise LookupError(f"Ollama has no model {self.tag!r}. Run: ollama pull {self.tag}")

    def generate(self, prompts: Sequence[str], decoding: Decoding) -> list[Generation]:
        results = []
        for prompt in prompts:
            payload = {
                "model": self.tag,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": decoding.temperature,
                    "top_p": decoding.top_p,
                    "seed": decoding.seed,
                    "num_predict": decoding.max_new_tokens,
                    "num_ctx": self.num_ctx,
                },
            }
            try:
                body = self._post("/api/chat", payload, timeout=600)
            except requests.RequestException as exc:
                results.append(Generation("", None, None, None, error=f"request_failed: {exc}"))
                continue
            n_prompt = body.get("prompt_eval_count")
            # Ollama silently drops the start of a prompt that exceeds num_ctx; flag it instead of trusting the output
            overflow = n_prompt is not None and n_prompt + decoding.max_new_tokens > self.num_ctx
            results.append(Generation(
                text=body.get("message", {}).get("content", ""),
                n_prompt_tokens=n_prompt,
                n_output_tokens=body.get("eval_count"),
                finish_reason=body.get("done_reason"),
                error="prompt_too_long" if overflow else None,
            ))
        return results

    def close(self) -> None:
        """Unload the model so the next one has the memory."""
        try:
            self._post("/api/generate", {"model": self.tag, "keep_alive": 0})
        except requests.RequestException:
            pass

    def versions(self) -> dict[str, str]:
        response = self.http.get(self.host + "/api/version", timeout=10)
        return {"ollama": response.json().get("version", "unknown") if response.ok else "unknown"}


class VLLMBackend:
    """vLLM engine on a CUDA GPU (full run). vLLM is imported here so the Mac never needs it."""

    name = "vllm"

    def __init__(self, hf_id: str, revision: str, precision: str, max_model_len: int,
                 gpu_memory_utilization: float, seed: int) -> None:
        from vllm import LLM  # noqa: PLC0415  (Linux-only dependency)

        self.precision = precision
        self.max_model_len = max_model_len
        self.model_ref = f"{hf_id}@{revision}"
        kwargs: dict[str, Any] = {
            "model": hf_id,
            "revision": revision,
            "tokenizer_revision": revision,
            "dtype": "bfloat16",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": 0},  # text-only study: no image budget
            "seed": seed,
        }
        if precision == "fp8":
            kwargs["quantization"] = "fp8"
        elif precision == "bnb4":
            kwargs["quantization"] = "bitsandbytes"
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.chat_template = getattr(self.tokenizer, "chat_template", None)

    def _prompt_tokens(self, prompt: str) -> int:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
        return len(ids)

    def generate(self, prompts: Sequence[str], decoding: Decoding) -> list[Generation]:
        from vllm import SamplingParams  # noqa: PLC0415

        params = SamplingParams(temperature=decoding.temperature, top_p=decoding.top_p,
                                max_tokens=decoding.max_new_tokens, seed=decoding.seed)
        results: list[Generation | None] = [None] * len(prompts)
        runnable: list[int] = []
        for i, prompt in enumerate(prompts):
            n = self._prompt_tokens(prompt)
            if n + decoding.max_new_tokens > self.max_model_len:
                results[i] = Generation("", n, None, None, error="prompt_too_long")
            else:
                runnable.append(i)
        if runnable:
            conversations = [[{"role": "user", "content": prompts[i]}] for i in runnable]
            outputs = self.llm.chat(conversations, params, use_tqdm=True)
            for i, out in zip(runnable, outputs, strict=True):
                completion = out.outputs[0]
                results[i] = Generation(
                    text=completion.text,
                    n_prompt_tokens=len(out.prompt_token_ids),
                    n_output_tokens=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                )
        return [r for r in results if r is not None]

    def close(self) -> None:
        import torch  # noqa: PLC0415

        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

    def versions(self) -> dict[str, str]:
        import torch  # noqa: PLC0415
        import vllm  # noqa: PLC0415

        return {"vllm": vllm.__version__, "torch": torch.__version__,
                "cuda": str(torch.version.cuda), "gpu": torch.cuda.get_device_name(0)}


MIN_COMPUTE_CAPABILITY = (8, 0)  # Ampere: bf16 compute and vLLM's FP8 (Marlin) kernels


def check_gpu(capability: tuple[int, int], total_gb: float, free_gb: float, utilization: float) -> list[str]:
    """
    Problems that would stop a vLLM run on this GPU, as readable messages (empty list = fine).

    Args:
        capability: CUDA compute capability, e.g. (8, 6) for an RTX 30-series card.
        total_gb: Total VRAM.
        free_gb: VRAM free right now (a desktop display can hold some).
        utilization: The gpu_memory_utilization vLLM will ask for.

    Returns:
        One message per problem.
    """
    problems = []
    if capability < MIN_COMPUTE_CAPABILITY:
        problems.append(f"compute capability {capability[0]}.{capability[1]} < 8.0: this card is older than RTX 30 "
                        "(Ampere), so vLLM can't run bf16 or FP8 here. Use the Mac profile instead.")
    if total_gb < 7.5:
        problems.append(f"{total_gb:.1f} GB VRAM: too small for a 4B model with an 8k context even at FP8.")
    elif free_gb < total_gb * utilization:
        problems.append(f"only {free_gb:.1f} of {total_gb:.1f} GB free, but vLLM will ask for "
                        f"{total_gb * utilization:.1f} GB (gpu_memory_utilization {utilization}). Close other GPU apps, "
                        f"or pass --gpu-mem {max(0.5, round(free_gb / total_gb - 0.03, 2))}.")
    return problems


def cuda_info() -> tuple[tuple[int, int], float, float, str]:
    """(compute capability, total GB, free GB, name) of GPU 0."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. The full profile needs Linux (or WSL2) + an NVIDIA GPU.")
    free, total = torch.cuda.mem_get_info(0)
    return torch.cuda.get_device_capability(0), total / 1024**3, free / 1024**3, torch.cuda.get_device_name(0)


def cuda_vram_gb() -> float:
    """Total VRAM of GPU 0 in GB (needs torch with CUDA, installed with vLLM)."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. The full profile needs Linux + an NVIDIA GPU.")
    return torch.cuda.get_device_properties(0).total_memory / 1024**3
