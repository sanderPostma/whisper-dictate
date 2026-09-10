"""Qwen3-ASR load and transcribe helpers (Hugging Face native)."""

from asr_models import qwen_hf_id

try:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
except ImportError:  # pragma: no cover - exercised only when deps missing
    torch = None
    AutoModelForMultimodalLM = None
    AutoProcessor = None

DEFAULT_MAX_NEW_TOKENS = 512


def load_qwen(model_name, device="cpu"):
    if AutoProcessor is None or AutoModelForMultimodalLM is None or torch is None:
        raise RuntimeError(
            "transformers with AutoModelForMultimodalLM is required for Qwen3-ASR. "
            "Install transformers>=5.13."
        )
    hf_id = qwen_hf_id(model_name)
    processor = AutoProcessor.from_pretrained(hf_id)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(hf_id, dtype=dtype)
    model.to(device)
    model.eval()
    return model, processor


def transcribe_qwen(
    model,
    processor,
    audio,
    language=None,
    prompt=None,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
):
    request = {"audio": audio}
    if language and str(language).strip():
        request["language"] = language
    if prompt and str(prompt).strip():
        request["prompt"] = prompt

    inputs = processor.apply_transcription_request(**request)
    inputs = inputs.to(model.device, model.dtype)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    decoded = processor.decode(generated_ids, return_format="transcription_only")
    if isinstance(decoded, (list, tuple)):
        text = decoded[0] if decoded else ""
    else:
        text = decoded
    return (text or "").strip()
