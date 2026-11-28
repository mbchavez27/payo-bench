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
FP32_MIN_TOTAL_VRAM_GB = 28.0  # 4B weights are 17.2 GB in fp32, plus an fp32 KV cache for 8k (2x T4 = 30 GB)
PRECISIONS = ("bf16", "fp8", "bnb4", "fp32")


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
        (weight-only FP8, which vLLM runs on Ampere through its Marlin kernel). "fp32" is never picked
        automatically: it is the T4 route (Kaggle 2x T4, tensor parallel 2), which has no bf16 or FP8.
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


def vllm_engine_args(hf_id: str, revision: str, precision: str, max_model_len: int,
                     gpu_memory_utilization: float, seed: int, enforce_eager: bool = False,
                     max_num_seqs: int = 256, tensor_parallel_size: int = 1,
                     max_num_batched_tokens: int | None = None) -> dict[str, Any]:
    """
    Keyword arguments for vllm.LLM.

    fp32 loads the weights unquantized in float32. It exists for Turing GPUs such as the T4, where bf16 and
    FP8 are unavailable and Gemma 3 gives non-finite logits in fp16. tensor_parallel_size splits the model
    across that many GPUs (2 on Kaggle's 2x T4). max_num_batched_tokens caps the tokens per engine step (the
    chunked-prefill chunk); a smaller cap shrinks the activation memory vLLM reserves, which fp32 on a T4 needs
    to leave room for an 8k KV cache. None keeps vLLM's default.

    enforce_eager and max_num_seqs change only vLLM's working memory (no CUDA graphs or torch.compile, fewer
    sequences in flight), not the weights, precision, context length or decoding. They let FP8 Gemma-3-4B
    fit its 8k-token KV cache on an 8 GB card, where the memory-profiling step fails otherwise.
    """
    kwargs: dict[str, Any] = {
        "model": hf_id,
        "revision": revision,
        "tokenizer_revision": revision,
        "dtype": "float32" if precision == "fp32" else "bfloat16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "limit_mm_per_prompt": {"image": 0},  # text-only study: no image budget
        "seed": seed,
        "enforce_eager": enforce_eager,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if precision == "fp8":
        kwargs["quantization"] = "fp8"
    elif precision == "bnb4":
        kwargs["quantization"] = "bitsandbytes"
    return kwargs


class VLLMBackend:
    """vLLM engine on a CUDA GPU (full run). vLLM is imported here so the Mac never needs it."""

    name = "vllm"

    def __init__(self, hf_id: str, revision: str, precision: str, max_model_len: int,
                 gpu_memory_utilization: float, seed: int, enforce_eager: bool = False,
                 max_num_seqs: int = 256, tensor_parallel_size: int = 1,
                 max_num_batched_tokens: int | None = None) -> None:
        from vllm import LLM  # noqa: PLC0415  (Linux-only dependency)

        self.precision = precision
        self.max_model_len = max_model_len
        self.model_ref = f"{hf_id}@{revision}"
        kwargs = vllm_engine_args(hf_id, revision, precision, max_model_len, gpu_memory_utilization, seed,
                                  enforce_eager, max_num_seqs, tensor_parallel_size, max_num_batched_tokens)
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


def check_gpu(capability: tuple[int, int], total_gb: float, free_gb: float, utilization: float,
              precision: str = "fp8", tensor_parallel_size: int = 1, n_gpus: int = 1) -> list[str]:
    """
    Problems that would stop a vLLM run on this GPU, as readable messages (empty list = fine).

    Args:
        capability: CUDA compute capability, e.g. (8, 6) for an RTX 30-series card.
        total_gb: Total VRAM of one GPU (the smallest, when tensor parallel uses several).
        free_gb: VRAM free right now on that GPU (a desktop display can hold some).
        utilization: The gpu_memory_utilization vLLM will ask for, on each GPU.
        precision: The resolved precision ("auto" already turned into a value).
        tensor_parallel_size: How many GPUs the model is split across.
        n_gpus: How many CUDA GPUs this machine has.

    Returns:
        One message per problem.
    """
    problems = []
    if capability < MIN_COMPUTE_CAPABILITY and precision != "fp32":
        problems.append(f"compute capability {capability[0]}.{capability[1]} < 8.0: this card is older than RTX 30 "
                        "(Ampere), so vLLM can't run bf16 or FP8 here. On a machine with two T4s (Kaggle "
                        "\"GPU T4 x2\") use --precision fp32 --tp 2; otherwise use the Mac profile.")
    if tensor_parallel_size > n_gpus:
        problems.append(f"--tp {tensor_parallel_size} needs {tensor_parallel_size} GPUs, but only {n_gpus} found.")
    if precision == "fp32" and total_gb * tensor_parallel_size < FP32_MIN_TOTAL_VRAM_GB:
        problems.append(f"fp32 needs about {FP32_MIN_TOTAL_VRAM_GB:.0f} GB across GPUs, but {tensor_parallel_size} x "
                        f"{total_gb:.1f} GB is {total_gb * tensor_parallel_size:.1f} GB. On Kaggle pick "
                        "\"GPU T4 x2\" and pass --tp 2.")
    if total_gb < 7.5:
        problems.append(f"{total_gb:.1f} GB VRAM: too small for a 4B model with an 8k context even at FP8.")
    elif free_gb < total_gb * utilization:
        problems.append(f"only {free_gb:.1f} of {total_gb:.1f} GB free, but vLLM will ask for "
                        f"{total_gb * utilization:.1f} GB (gpu_memory_utilization {utilization}). Close other GPU apps, "
                        f"or pass --gpu-mem {max(0.5, round(free_gb / total_gb - 0.03, 2))}.")
    return problems


def cuda_info(n_used: int = 1) -> tuple[tuple[int, int], float, float, str, int]:
    """
    (compute capability, total GB, free GB, name, GPU count) for the first n_used GPUs.

    With several GPUs (tensor parallel), total and free are the smallest among them, because vLLM asks
    for the same share of memory on each.
    """
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. The full profile needs Linux (or WSL2) + an NVIDIA GPU.")
    count = torch.cuda.device_count()
    memory = [torch.cuda.mem_get_info(i) for i in range(min(n_used, count))]  # (free, total) bytes
    total = min(t for _, t in memory) / 1024**3
    free = min(f for f, _ in memory) / 1024**3
    return torch.cuda.get_device_capability(0), total, free, torch.cuda.get_device_name(0), count


def cuda_vram_gb() -> float:
    """Total VRAM of GPU 0 in GB (needs torch with CUDA, installed with vLLM)."""
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. The full profile needs Linux + an NVIDIA GPU.")
    return torch.cuda.get_device_properties(0).total_memory / 1024**3
