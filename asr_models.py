"""Model-id helpers shared by the tray client and remote server."""

QWEN_MODELS = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B-hf",
}


def is_qwen_model(model_name):
    return bool(model_name) and model_name in QWEN_MODELS


def is_distil_model(model_name):
    return bool(model_name) and model_name.startswith("distil-")


def is_english_only_model(model_name):
    if not model_name or is_qwen_model(model_name):
        return False
    return model_name.endswith(".en") or is_distil_model(model_name)


def multilingual_model(model_name):
    if not model_name or is_qwen_model(model_name):
        return model_name
    if model_name.endswith(".en"):
        return model_name[: -len(".en")]
    if is_distil_model(model_name):
        return "large"
    return model_name


def qwen_hf_id(model_name):
    try:
        return QWEN_MODELS[model_name]
    except KeyError:
        raise ValueError(f"Unknown Qwen ASR model: {model_name}") from None


def effective_remote_model(local_model, remote_model, language="en"):
    """Prefer a locally selected Qwen model on the remote server."""
    if is_qwen_model(local_model):
        return local_model
    if language != "en":
        return multilingual_model(remote_model)
    return remote_model


def build_remote_header(language, model, sample_rate, audio_size, prompt=None):
    header = {
        "language": language,
        "model": model,
        "sample_rate": sample_rate,
        "audio_size": audio_size,
    }
    if prompt is not None and str(prompt).strip():
        header["prompt"] = prompt
    return header
