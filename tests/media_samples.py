"""Small synthetic media payloads for tests."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def image_bytes(width: int = 64, height: int = 48, color=(200, 30, 30), fmt: str = "PNG", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def png_b64(width: int = 64, height: int = 48, color=(200, 30, 30)) -> str:
    return b64(image_bytes(width, height, color))


def jpeg_scan_offsets(data: bytes) -> list[tuple[int, int]]:
    """(start, end) of every scan (SOS segment plus its entropy-coded data) of a JPEG."""
    scans = []
    position = 2
    while data[position + 1] != 0xD9:
        length = int.from_bytes(data[position + 2 : position + 4], "big")
        if data[position + 1] != 0xDA:
            position += 2 + length
            continue
        end = position + 2 + length
        while not (data[end] == 0xFF and data[end + 1] != 0x00 and not 0xD0 <= data[end + 1] <= 0xD7):
            end += 1
        scans.append((position, end))
        position = end
    return scans


def progressive_jpeg(width: int = 64, height: int = 48, extra_scans: int = 0, **save_kwargs) -> bytes:
    """Progressive JPEG; extra_scans repeats its last scan (decode time grows with the scan count)."""
    data = image_bytes(width, height, (90, 140, 30), fmt="JPEG", progressive=True, **save_kwargs)
    start, end = jpeg_scan_offsets(data)[-1]
    eoi = len(data) - 2
    return data[:eoi] + data[start:end] * extra_scans + data[eoi:]


def audio_bytes(
    seconds: float = 1.0,
    rate: int = 16000,
    channels: int = 1,
    fmt: str = "WAV",
    subtype: str = "PCM_16",
    values: tuple[float, ...] | None = None,
) -> bytes:
    import soundfile as sf

    frames = int(round(seconds * rate))
    t = np.arange(frames) / rate
    if values is None:
        mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        data = np.stack([mono] * channels, axis=1)
    else:
        data = np.tile(np.asarray(values, dtype=np.float32), (frames, 1))
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


def wav_b64(seconds: float = 1.0, rate: int = 16000, **kwargs) -> str:
    return b64(audio_bytes(seconds, rate, **kwargs))


def frames_b64(count: int = 4, width: int = 64, height: int = 48) -> list[str]:
    return [png_b64(width, height, (index * 40 % 256, 80, 120)) for index in range(count)]
