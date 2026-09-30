"""Recording rate vs. model rate.

Audio is recorded at config["sample_rate"]; Whisper, Qwen and the remote
server take 16 kHz, so it is converted at the model boundary only.
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

MODEL_RATE = 16000


def resample(audio, src_rate, dst_rate=MODEL_RATE):
    """Band-limited resampling of mono float32 audio."""
    audio = np.asarray(audio, dtype=np.float32)
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
