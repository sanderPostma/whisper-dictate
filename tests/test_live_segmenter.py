import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_segmenter import ChunkReady, LiveSegmenter, LongPause

SR = 16000
BLOCK = 800  # 50 ms


def tone(seconds, amp=0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def feed_all(seg, audio, block=BLOCK):
    events = []
    for i in range(0, len(audio), block):
        events.extend(seg.feed(audio[i:i + block]))
    return events


def chunks_of(events):
    return [e for e in events if isinstance(e, ChunkReady)]


class LiveSegmenterTests(unittest.TestCase):
    def test_pause_emits_chunk_with_trimmed_trailing_silence(self):
        seg = LiveSegmenter(SR)
        chunks = chunks_of(feed_all(seg, np.concatenate([tone(1.0), silence(0.7)])))
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].t0, 0.0)
        self.assertAlmostEqual(chunks[0].t1, 1.15)
        self.assertEqual(len(chunks[0].audio), int(1.15 * SR))
        self.assertFalse(chunks[0].forced)

    def test_short_pause_does_not_split(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(0.5), tone(0.5)]))
        self.assertEqual(chunks_of(events), [])

    def test_leading_preroll_is_kept(self):
        seg = LiveSegmenter(SR)
        chunk = chunks_of(feed_all(seg, np.concatenate([silence(1.0), tone(1.0), silence(0.7)])))[0]
        self.assertAlmostEqual(chunk.t0, 0.85)

    def test_long_pause_fires_once_after_chunk(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(1.0), silence(3.0)]))
        self.assertEqual([type(e) for e in events], [ChunkReady, LongPause])

    def test_forced_split_is_contiguous(self):
        seg = LiveSegmenter(SR, max_chunk_s=8.0)
        chunks = chunks_of(feed_all(seg, tone(9.0)) + seg.flush())
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].forced)
        self.assertAlmostEqual(chunks[0].t1, 8.0)
        self.assertAlmostEqual(chunks[1].t0, chunks[0].t1)
        self.assertEqual(len(chunks[0].audio) + len(chunks[1].audio), 9 * SR)

    def test_odd_block_sizes_keep_timing(self):
        seg = LiveSegmenter(SR)
        chunk = chunks_of(feed_all(seg, np.concatenate([silence(0.5), tone(1.0), silence(0.8)]), block=333))[0]
        self.assertAlmostEqual(chunk.t0 + len(chunk.audio) / SR, chunk.t1)
        self.assertLess(abs(chunk.t0 - 0.35), 0.03)

    def test_zero_threshold_uses_default(self):
        seg = LiveSegmenter(SR, threshold=0.0)
        self.assertEqual(seg.threshold, 0.01)
        self.assertEqual(feed_all(seg, np.concatenate([tone(1.0, amp=0.001), silence(1.0)])), [])

    def test_blip_shorter_than_min_speech_is_dropped(self):
        seg = LiveSegmenter(SR)
        events = feed_all(seg, np.concatenate([tone(0.1), silence(2.0)]))
        self.assertEqual(events, [])

    def test_flush_without_speech_is_empty(self):
        seg = LiveSegmenter(SR)
        feed_all(seg, silence(1.0))
        self.assertEqual(seg.flush(), [])

    def test_flush_emits_pending_speech_once(self):
        seg = LiveSegmenter(SR)
        feed_all(seg, tone(1.0))
        chunks = seg.flush()
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0].t1, 1.0)
        self.assertEqual(seg.flush(), [])


class BufferReuseTests(unittest.TestCase):
    def test_chunks_survive_the_caller_reusing_its_buffer(self):
        # sounddevice reuses the callback's buffer after the callback returns.
        seg = LiveSegmenter(16000, pause_ms=100, threshold=0.01, pad_ms=0, min_speech_ms=0)
        buf = np.zeros(800, dtype=np.float32)
        events = []
        for _ in range(10):
            buf[:] = 0.5
            events += seg.feed(buf)
        for _ in range(10):
            buf[:] = 0.0
            events += seg.feed(buf)
        chunks = [e for e in events if isinstance(e, ChunkReady)]
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(float(chunks[0].audio.max()), 0.5)


if __name__ == "__main__":
    unittest.main()
