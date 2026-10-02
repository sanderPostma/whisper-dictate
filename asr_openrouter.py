"""Transcription through OpenRouter's /audio/transcriptions API."""

import base64
import io
import json
import os
import urllib.error
import urllib.request
import wave

import numpy as np

from asr_models import openrouter_model_id, openrouter_provider_tag

OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
API_KEY_ENV = "OPENROUTER_API_KEY"


def openrouter_api_key(config=None):
    """The API key: $OPENROUTER_API_KEY, else config["openrouter_api_key"]."""
    key = os.environ.get(API_KEY_ENV) or (config or {}).get("openrouter_api_key") or ""
    return key.strip()


def encode_wav(audio, sample_rate=16000):
    """Float32 mono samples in [-1, 1] -> 16-bit PCM WAV bytes."""
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def vocabulary_phrases(prompt):
    """Terms from 'Vocabulary:' lines. Other lines are prior transcript, not hints."""
    phrases = []
    for raw_line in str(prompt).splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("vocabulary:"):
            continue
        for part in line.split(":", 1)[1].split(","):
            phrase = part.strip().rstrip(".")
            if phrase:
                phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def build_request_body(model_name, wav_bytes, language=None, prompt=None):
    model_id = openrouter_model_id(model_name)
    body = {
        "model": model_id,
        "input_audio": {"data": base64.b64encode(wav_bytes).decode("ascii"), "format": "wav"},
    }
    if language and str(language).strip():
        body["language"] = language
    if prompt and str(prompt).strip():
        # OpenRouter does not normalize prompt. gpt-transcribe takes it under
        # its provider slug. MAI-Transcribe-2 has no prompt; keyword hints are
        # provider.options.azure.phraseList.phrases.
        tag = openrouter_provider_tag(model_id)
        if model_id == "microsoft/mai-transcribe-2":
            phrases = vocabulary_phrases(prompt)
            if phrases:
                body["provider"] = {"options": {tag: {"phraseList": {"phrases": phrases}}}}
        else:
            body["provider"] = {"options": {tag: {"prompt": prompt}}}
    return body


def transcribe_openrouter(audio, model_name, api_key, sample_rate=16000,
                          language=None, prompt=None, timeout=60):
    if not api_key:
        raise RuntimeError(
            f"No OpenRouter API key: set ${API_KEY_ENV} or \"openrouter_api_key\" in config.json"
        )
    body = build_request_body(
        model_name, encode_wav(audio, sample_rate), language=language, prompt=prompt
    )
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from None
    usage = result.get("usage") or {}
    if usage.get("cost") is not None:
        print(f"[openrouter] {model_name}: {usage.get('seconds')}s audio, ${usage['cost']:.6f}")
    return (result.get("text") or "").strip()
