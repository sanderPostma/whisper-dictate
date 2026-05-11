#!/usr/bin/env python3
"""
Whisper Dictate remote transcription server.

Protocol:
  Request  = 4-byte header length (uint32 BE) + JSON header + raw float32 audio bytes
  Response = 4-byte payload length (uint32 BE) + JSON {"text", "elapsed", "error"}
  Health   = send JSON header with {"ping": true, "audio_size": 0}
"""

import argparse
import gc
import json
import os
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import whisper

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError as _transformers_err:
    HAS_TRANSFORMERS = False
    _TRANSFORMERS_IMPORT_ERR = str(_transformers_err)
else:
    _TRANSFORMERS_IMPORT_ERR = None

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9876
DEFAULT_MODEL = "distil-large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_CACHE_SIZE = 2

HAS_WIN_SERVICE = False
if sys.platform == "win32":
    try:
        import servicemanager
        import win32event
        import win32service
        import win32serviceutil

        HAS_WIN_SERVICE = True
    except ImportError:
        HAS_WIN_SERVICE = False


def recv_exact(sock, size):
    """Receive exactly size bytes from a socket."""
    chunks = []
    received = 0
    while received < size:
        chunk = sock.recv(size - received)
        if not chunk:
            raise ConnectionError("Connection closed before full payload was received")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


class WhisperTCPServer:
    def __init__(self, host, port, model_name, device, cache_size=DEFAULT_CACHE_SIZE):
        self.host = host
        self.port = port
        self.device = device
        self.cache_size = max(1, int(cache_size))
        # OrderedDict as LRU: {model_name: (obj, backend)}
        self.model_cache = OrderedDict()
        self.model_name = None
        self.model_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.server_socket = None

        if model_name:
            with self.model_lock:
                self._ensure_model_locked(model_name)
        else:
            print("[whisper-server] No startup model; will load on first client request")

    @staticmethod
    def _is_distil(model_name):
        return model_name.startswith("distil-")

    def _free_vram(self):
        gc.collect()
        if HAS_TORCH and self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_name):
        print(f"[whisper-server] Loading model '{model_name}' on {self.device}...")
        if self._is_distil(model_name):
            if not HAS_TRANSFORMERS:
                detail = f" (import error: {_TRANSFORMERS_IMPORT_ERR})" if _TRANSFORMERS_IMPORT_ERR else ""
                raise RuntimeError(
                    f"transformers not installed; cannot load distil-whisper models{detail}. "
                    "Run: pip install transformers accelerate"
                )
            hf_id = f"distil-whisper/{model_name}"
            torch_dtype = torch.float16 if (HAS_TORCH and self.device == "cuda") else torch.float32
            hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            hf_model.to(self.device)
            processor = AutoProcessor.from_pretrained(hf_id)
            obj = (hf_model, processor)
            backend = "distil"
        else:
            obj = whisper.load_model(model_name, device=self.device)
            backend = "whisper"
        print(f"[whisper-server] Loaded '{model_name}' (backend: {backend})")
        return obj, backend

    def _ensure_model_locked(self, model_name):
        """Select model_name, loading it if missing. Evicts LRU if over cap."""
        if model_name in self.model_cache:
            self.model_cache.move_to_end(model_name)
            self.model_name = model_name
            print(f"[whisper-server] Cache hit: '{model_name}' "
                  f"({len(self.model_cache)}/{self.cache_size} loaded)")
            return

        # Evict LRU entries BEFORE loading so we don't briefly hold old+new in VRAM.
        while len(self.model_cache) >= self.cache_size:
            evicted, _ = self.model_cache.popitem(last=False)
            print(f"[whisper-server] Evicting '{evicted}' from cache")
        self._free_vram()

        try:
            obj, backend = self._load_model(model_name)
        except Exception as exc:
            # OOM or partial load can leak VRAM. Evict everything, free, retry once.
            is_oom = "out of memory" in str(exc).lower() or (
                HAS_TORCH and isinstance(exc, torch.cuda.OutOfMemoryError)
            )
            if not is_oom or not self.model_cache:
                raise
            print(f"[whisper-server] Load failed with OOM; evicting all cached models and retrying")
            while self.model_cache:
                evicted, _ = self.model_cache.popitem(last=False)
                print(f"[whisper-server] OOM recovery: evicting '{evicted}'")
            self._free_vram()
            obj, backend = self._load_model(model_name)

        self.model_cache[model_name] = (obj, backend)
        self.model_name = model_name
        print(f"[whisper-server] Cache: {list(self.model_cache.keys())}")

    @property
    def _cache_entry(self):
        return self.model_cache.get(self.model_name)

    @property
    def model(self):
        entry = self._cache_entry
        if entry is None:
            return None
        obj = entry[0]
        return obj[0] if isinstance(obj, tuple) else obj

    @property
    def processor(self):
        entry = self._cache_entry
        if entry is None:
            return None
        obj = entry[0]
        return obj[1] if isinstance(obj, tuple) else None

    @property
    def backend(self):
        entry = self._cache_entry
        return entry[1] if entry else None

    def _transcribe(self, audio, language, requested_model):
        with self.model_lock:
            if requested_model and requested_model != self.model_name:
                self._ensure_model_locked(requested_model)

            if self.backend == "distil":
                rms = float(np.sqrt(np.mean(audio ** 2)))
                print(f"[whisper-server] distil audio: {len(audio)/16000:.2f}s rms={rms:.4f}")
                torch_dtype = torch.float16 if (HAS_TORCH and self.device == "cuda") else torch.float32
                inputs = self.processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(self.device, dtype=torch_dtype)
                gen_kwargs = {"task": "transcribe"}
                if language:
                    gen_kwargs["language"] = language
                with torch.no_grad():
                    predicted_ids = self.model.generate(inputs, **gen_kwargs)
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                print(f"[whisper-server] distil result: {repr(text)}")
                return {"text": text}

            return self.model.transcribe(
                audio,
                language=language,
                fp16=(self.device == "cuda")
            )

    def handle_client(self, conn, addr):
        with conn:
            response = {"text": "", "elapsed": 0.0, "error": None}
            try:
                header_len = struct.unpack(">I", recv_exact(conn, 4))[0]
                header_raw = recv_exact(conn, header_len)
                header = json.loads(header_raw.decode("utf-8"))

                if header.get("ping"):
                    payload = json.dumps(response).encode("utf-8")
                    conn.sendall(struct.pack(">I", len(payload)))
                    conn.sendall(payload)
                    return

                audio_size = int(header.get("audio_size", 0))
                if audio_size < 0:
                    raise ValueError("audio_size must be non-negative")

                audio_raw = recv_exact(conn, audio_size)
                audio = np.frombuffer(audio_raw, dtype=np.float32).copy()

                sample_rate = int(header.get("sample_rate", 16000))
                if sample_rate != 16000:
                    print(
                        f"[whisper-server] Warning: sample_rate={sample_rate}, "
                        "expected 16000 for Whisper input"
                    )

                requested_model = header.get("model", self.model_name)
                language = header.get("language")

                start = time.time()
                result = self._transcribe(audio, language, requested_model)
                response["text"] = result.get("text", "").strip()
                response["elapsed"] = time.time() - start
            except Exception as e:
                response["error"] = str(e)
                print(f"[whisper-server] Error for {addr}: {e}")

            payload = json.dumps(response).encode("utf-8")
            conn.sendall(struct.pack(">I", len(payload)))
            conn.sendall(payload)

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            self.server_socket = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen()
            sock.settimeout(1.0)
            print(f"[whisper-server] Listening on {self.host}:{self.port}")

            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise

                threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True
                ).start()

    def stop(self):
        self.stop_event.set()
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass


if HAS_WIN_SERVICE:
    class WhisperDictateServerService(win32serviceutil.ServiceFramework):
        _svc_name_ = "WhisperDictateServer"
        _svc_display_name_ = "Whisper Dictate Server"
        _svc_description_ = "Remote Whisper transcription server for whisper-dictate"

        def __init__(self, args):
            super().__init__(args)
            self.stop_handle = win32event.CreateEvent(None, 0, 0, None)
            self.server = None

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            if self.server:
                self.server.stop()
            win32event.SetEvent(self.stop_handle)

        def _redirect_output(self):
            """Send stdout/stderr and the logging module to a log file next to the script."""
            try:
                import logging
                script_dir = os.path.dirname(os.path.abspath(__file__))
                log_path = os.path.join(script_dir, "service.log")
                log_file = open(log_path, "a", buffering=1, encoding="utf-8")
                sys.stdout = log_file
                sys.stderr = log_file
                # pythonservice.exe starts with stderr=None; the logging module's
                # default handlers captured that None and fail with
                # "AttributeError: 'NoneType' object has no attribute 'write'"
                # whenever a library (huggingface_hub, transformers) emits a warning.
                # Point them all at our redirected log file.
                logging.lastResort = logging.StreamHandler(log_file)
                logging.basicConfig(stream=log_file, level=logging.WARNING, force=True)
                servicemanager.LogInfoMsg(f"WhisperDictateServer logging to {log_path}")
            except Exception as e:
                servicemanager.LogErrorMsg(f"Failed to redirect output: {e}")

        def SvcDoRun(self):
            import traceback
            self._redirect_output()
            print(f"\n===== Service start at {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
            print(f"Python: {sys.executable}")
            print(f"Script: {os.path.abspath(__file__)}")
            print(f"cwd:    {os.getcwd()}")

            # Keep SCM happy while the model loads: periodic START_PENDING pings.
            stop_pending = threading.Event()
            def pending_pings():
                checkpoint = 1
                while not stop_pending.is_set():
                    self.ReportServiceStatus(
                        win32service.SERVICE_START_PENDING,
                        waitHint=30000,
                    )
                    checkpoint += 1
                    if stop_pending.wait(10):
                        return
            threading.Thread(target=pending_pings, daemon=True).start()

            try:
                host = os.environ.get("WHISPER_SERVER_HOST", DEFAULT_HOST)
                port = int(os.environ.get("WHISPER_SERVER_PORT", str(DEFAULT_PORT)))
                model = os.environ.get("WHISPER_SERVER_MODEL", DEFAULT_MODEL)
                device = os.environ.get("WHISPER_SERVER_DEVICE", DEFAULT_DEVICE)
                cache_size = int(os.environ.get(
                    "WHISPER_SERVER_CACHE_SIZE", str(DEFAULT_CACHE_SIZE)
                ))
                print(f"Config: host={host} port={port} model={model} "
                      f"device={device} cache_size={cache_size}")
                self.server = WhisperTCPServer(
                    host, port, model, device, cache_size=cache_size
                )
            except Exception as e:
                stop_pending.set()
                err = f"Startup failed: {e}\n{traceback.format_exc()}"
                print(err)
                servicemanager.LogErrorMsg(err)
                return
            finally:
                stop_pending.set()

            self.ReportServiceStatus(win32service.SERVICE_RUNNING)
            servicemanager.LogInfoMsg("WhisperDictateServer running")
            print("Service now RUNNING")

            # Serve in a thread; wait on stop_handle.
            def serve():
                try:
                    self.server.serve_forever()
                except Exception as e:
                    print(f"serve_forever error: {e}\n{traceback.format_exc()}")
            threading.Thread(target=serve, daemon=True).start()

            win32event.WaitForSingleObject(self.stop_handle, win32event.INFINITE)
            print(f"===== Service stop at {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
            servicemanager.LogInfoMsg("WhisperDictateServer stopped")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Whisper Dictate remote TCP server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Execution device"
    )
    parser.add_argument(
        "--model-cache-size",
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help=f"How many models to keep loaded in VRAM (default: {DEFAULT_CACHE_SIZE})"
    )
    parser.add_argument(
        "--install-service",
        action="store_true",
        help="Install WhisperDictateServer Windows service"
    )
    parser.add_argument(
        "--remove-service",
        action="store_true",
        help="Remove WhisperDictateServer Windows service"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.install_service and args.remove_service:
        raise SystemExit("Use either --install-service or --remove-service, not both")

    if args.install_service or args.remove_service:
        if not HAS_WIN_SERVICE:
            raise SystemExit("Windows service mode requires pywin32 on Windows")

        cmd = "install" if args.install_service else "remove"
        win32serviceutil.HandleCommandLine(
            WhisperDictateServerService,
            argv=[sys.argv[0], cmd]
        )
        return

    server = WhisperTCPServer(
        args.host, args.port, args.model, args.device,
        cache_size=args.model_cache_size,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[whisper-server] Stopping...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
