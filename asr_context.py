"""ASR context / hotword packs (Qwen prompt, Whisper initial_prompt)."""

from pathlib import Path

CONTEXT_PACKS = {
    "none": "",
    "custom": "",
    "developer": (
        "Vocabulary: git, GitHub, GitLab, pull request, rebase, stash, commit, "
        "branch, merge, cherry-pick, kubectl, Kubernetes, k8s, Docker, Compose, "
        "Dockerfile, YAML, JSON, REST, GraphQL, PostgreSQL, Redis, SQLite, "
        "TypeScript, JavaScript, Python, pytest, FastAPI, Flask, Django, systemd, "
        "nginx, Whisper, Qwen, CUDA, VRAM, Hugging Face, transformers, tokenizer, "
        "LLM, ASR, transcribe, xdotool, Wayland, X11, CLI, stdin, stdout, stderr, "
        "regex, glob, protobuf, gRPC, OpenAPI, JWT, OAuth, npm, pip, venv, conda, "
        "Makefile, CMake, Gradle, Maven, Rust, cargo, Go, Java, Kotlin, Spring, "
        "React, Vue, Next.js, Node.js, WebSocket, HTTP, HTTPS, TCP, UDP, SSH, "
        "sudo, chmod, grep, jq, curl, wget, ffmpeg, numpy, PyTorch, TensorFlow, "
        "cuDNN, ONNX, beads, KTOR, Sphereon."
    ),
}


def _read_context_file(context_file):
    if not context_file:
        return ""
    path = Path(context_file)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return ""


def context_from_config(config, config_dir):
    config = config or {}
    pack = config.get("context_pack", "none")
    extra = config.get("context_prompt", "") or ""
    context_file = None
    if pack == "custom" and config_dir:
        context_file = Path(config_dir) / "context.txt"
    return resolve_asr_context(pack, extra=extra, context_file=context_file)


def resolve_asr_context(pack, extra="", context_file=None):
    """Build the context string sent to the ASR model.

    pack: none | developer | custom (unknown packs are empty).
    extra: always appended when non-blank.
    context_file: read when pack is custom.
    """
    parts = []
    pack_key = pack or "none"
    pack_text = CONTEXT_PACKS.get(pack_key, "")
    if pack_text:
        parts.append(pack_text.strip())
    if pack_key == "custom":
        file_text = _read_context_file(context_file)
        if file_text:
            parts.append(file_text)
    extra_text = (extra or "").strip()
    if extra_text:
        parts.append(extra_text)
    return "\n".join(parts)
