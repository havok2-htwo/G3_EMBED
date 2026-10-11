"""Image, audio and video (frame list) inputs: decoding, limits, token accounting and VRAM chunking.

Model-free: nothing here loads a model or touches the GPU, so it is unit-testable on any machine.
Token formulas match the EmbeddingGemma 2 processor (sentence-transformers 6.1 / transformers 5.19).
"""
from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Mapping, Sequence

import numpy as np

from .genesis_embed_server_globals import MEDIA_LIMIT_SETTINGS, MEDIA_TOGGLE_SETTINGS

MEDIA_KINDS = ("image", "audio", "video")
# Placeholders the processor expands into media tokens; inside text they break preprocessing.
RESERVED_TOKENS = ("<|image|>", "<|audio|>", "<|video|>")

IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")
FRAME_FORMATS = ("PNG", "JPEG", "WEBP")
AUDIO_FORMATS = frozenset({"WAV", "WAVEX", "FLAC", "OGG", "MP3"})
# Fixed set: arbitrary rates (e.g. 44101 Hz, gcd 1) make polyphase resampling filters explode.
AUDIO_SAMPLE_RATES = frozenset({8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000})
AUDIO_TARGET_RATE = 16000
MAX_AUDIO_CHANNELS = 2
MIN_AUDIO_SECONDS = 0.1
MAX_VIDEO_FRAME_PIXELS = 1920 * 1080
# 1 px high frames are read as channels-first by the video processor (token count no longer matches).
MIN_VIDEO_FRAME_SIDE = 8
# Progressive JPEG decode time grows with the scan count and libjpeg has no limit; real files have about 10.
MAX_JPEG_SCANS = 64
# Slack on top of the 4/3 base64 rule (padding, a short trailing newline is still rejected by strict decode).
BASE64_SLACK_CHARS = 16

# EmbeddingGemma 2 processor constants.
SPECIAL_TOKENS = 2  # <bos> ... <eos> per input
WRAP_TOKENS = 2  # <|image> ... <image|> per image / video frame, <|audio> ... <audio|> per clip
_PATCH, _POOL = 16, 3
_IMAGE_MAX_SOFT = 280
_FRAME_MAX_SOFT = 140
_PROCESSOR_MAX_FRAMES = 32
_AUDIO_MAX_SAMPLES = 480_000  # feature extractor truncation (30 s at 16 kHz)
_AUDIO_HOP, _AUDIO_FRAME = 160, 320
# The processor resizes to at most 280 (frames: 140) x 9 patches of 16x16 px anyway. Decoded pixels above 4x
# that are downscaled right away, so a request never keeps full-resolution images in RAM.
MAX_KEPT_IMAGE_PIXELS = 4 * _IMAGE_MAX_SOFT * _POOL**2 * _PATCH**2
MAX_KEPT_FRAME_PIXELS = 4 * _FRAME_MAX_SOFT * _POOL**2 * _PATCH**2


class MediaRequestError(ValueError):
    """A request problem that maps to an HTTP error with a detail object {"error", "message", ...}."""

    def __init__(self, error: str, message: str, *, status_code: int = 400, **extra: Any):
        self.error = error
        self.message = message
        self.status_code = status_code
        self.extra = extra
        super().__init__(message)

    def detail(self) -> dict[str, Any]:
        return {"error": self.error, "message": self.message, **self.extra}


@dataclass(frozen=True)
class MediaLimits:
    image_inputs_enabled: bool
    audio_inputs_enabled: bool
    video_inputs_enabled: bool
    max_images_per_input: int
    max_images_per_request: int
    max_audio_clips_per_input: int
    max_audio_seconds_per_clip: int
    max_audio_seconds_per_request: int
    max_videos_per_request: int
    max_video_frames: int
    max_media_item_mb: int
    max_image_megapixels: int
    max_request_mb: int
    media_pass_budget_mb: int
    max_concurrent_media_requests: int

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "MediaLimits":
        values: dict[str, Any] = {name: bool(settings.get(name, default)) for name, default in MEDIA_TOGGLE_SETTINGS.items()}
        for name, (default, minimum, maximum) in MEDIA_LIMIT_SETTINGS.items():
            try:
                numeric = int(settings.get(name, default))
            except (TypeError, ValueError):
                numeric = default
            values[name] = max(minimum, min(maximum, numeric))
        return cls(**values)

    def enabled(self, kind: str) -> bool:
        return bool(getattr(self, f"{kind}_inputs_enabled"))

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class MediaInputData:
    """One undecoded object input: base64 strings per field, in request order."""

    text: str | None = None
    image: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    video: list[str] | None = None

    def kinds(self) -> list[str]:
        return [kind for kind in MEDIA_KINDS if self.count(kind)]

    def count(self, kind: str) -> int:
        if kind == "video":
            return 1 if self.video else 0
        return len(getattr(self, kind))

    def has_media(self) -> bool:
        return bool(self.image or self.audio or self.video)


@dataclass
class DecodedMedia:
    """One decoded object input: RGB images, mono float32 16 kHz audio, one video as equally sized RGB frames."""

    text: str | None = None
    images: list[Any] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)
    video: list[Any] | None = None

    @property
    def audio_seconds(self) -> float:
        return sum(len(clip) for clip in self.audio) / AUDIO_TARGET_RATE

    @property
    def longest_clip_seconds(self) -> float:
        return min(max((len(clip) for clip in self.audio), default=0), _AUDIO_MAX_SAMPLES) / AUDIO_TARGET_RATE

    @property
    def video_frames(self) -> int:
        return len(self.video or [])

    def media_tokens(self) -> int:
        total = sum(image_tokens(image.height, image.width) for image in self.images)
        total += sum(audio_tokens(len(clip)) for clip in self.audio)
        if self.video:
            total += video_tokens(len(self.video), self.video[0].height, self.video[0].width)
        return total

    def summary(self) -> dict[str, Any]:
        return {"images": len(self.images), "audio_seconds": round(self.audio_seconds, 3), "video_frames": self.video_frames}

    def to_st_input(self) -> dict[str, Any]:
        # Key order is the order of the parts in the embedded sequence: text, images, audio, video.
        payload: dict[str, Any] = {}
        if self.text:
            payload["text"] = self.text
        if self.images:
            payload["image"] = list(self.images)
        if self.audio:
            payload["audio"] = [{"array": clip, "sampling_rate": AUDIO_TARGET_RATE} for clip in self.audio]
        if self.video:
            # A list of frames without metadata: the processor keeps every frame (no re-sampling).
            payload["video"] = list(self.video)
        return payload


# --- token accounting ---------------------------------------------------------------------------------

def _vision_soft_tokens(height: int, width: int, max_soft: int) -> int:
    max_patches = max_soft * _POOL**2
    target_px = max_patches * _PATCH**2
    factor = math.sqrt(target_px / (height * width))
    side = _POOL * _PATCH
    target_h = int(math.floor(factor * height / side)) * side
    target_w = int(math.floor(factor * width / side)) * side
    if target_h == 0 and target_w == 0:
        raise ValueError("image too small")
    max_side = (max_patches // _POOL**2) * side
    if target_h == 0:
        target_h, target_w = side, min(int(math.floor(width / height)) * side, max_side)
    elif target_w == 0:
        target_w, target_h = side, min(int(math.floor(height / width)) * side, max_side)
    return (target_h // _PATCH) * (target_w // _PATCH) // _POOL**2


def image_tokens(height: int, width: int) -> int:
    return _vision_soft_tokens(height, width, _IMAGE_MAX_SOFT) + WRAP_TOKENS


def video_tokens(frames: int, height: int, width: int) -> int:
    return min(frames, _PROCESSOR_MAX_FRAMES) * (_vision_soft_tokens(height, width, _FRAME_MAX_SOFT) + WRAP_TOKENS)


def audio_tokens(samples_16k: int) -> int:
    samples = min(samples_16k, _AUDIO_MAX_SAMPLES)
    mel_frames = (samples + _AUDIO_FRAME // 2 - (_AUDIO_FRAME + 1)) // _AUDIO_HOP + 1
    if mel_frames <= 0:
        return WRAP_TOKENS
    tokens = mel_frames
    for _ in range(2):
        tokens = (tokens - 1) // 2 + 1
    return tokens + WRAP_TOKENS


def input_tokens(text_tokens: int, item: DecodedMedia) -> int:
    """Total sequence length of one object input; text_tokens excludes special tokens."""
    return SPECIAL_TOKENS + text_tokens + item.media_tokens()


# --- VRAM cost and chunking ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputCost:
    images: int = 0
    video_frames: int = 0
    audio_clips: int = 0
    longest_clip_seconds: float = 0.0
    tokens: int = 0


def item_cost(item: DecodedMedia, tokens: int) -> InputCost:
    return InputCost(len(item.images), item.video_frames, len(item.audio), item.longest_clip_seconds, tokens)


def estimate_pass_mb(costs: Sequence[InputCost], cost_mb: Mapping[str, float]) -> float:
    """Estimated extra VRAM of one forward pass over these inputs (towers and LM peak at different times).

    The feature extractor pads every clip of a pass to the longest one: audio costs clips x longest seconds.
    """
    if not costs:
        return 0.0
    padded_audio_seconds = sum(cost.audio_clips for cost in costs) * max(cost.longest_clip_seconds for cost in costs)
    tower = (
        float(cost_mb.get("image", 0)) * sum(cost.images for cost in costs)
        + float(cost_mb.get("video_frame", 0)) * sum(cost.video_frames for cost in costs)
        + float(cost_mb.get("audio_second", 0)) * padded_audio_seconds
    )
    language_model = float(cost_mb.get("lm_token", 0)) * len(costs) * max(cost.tokens for cost in costs)
    return max(tower, language_model) + float(cost_mb.get("overhead", 0))


def plan_chunks(costs: Sequence[InputCost], budget_mb: float, cost_mb: Mapping[str, float]) -> list[list[int]]:
    """Greedy, order-preserving split into forward passes whose estimate stays within budget_mb."""
    chunks: list[list[int]] = []
    current: list[int] = []
    for index, cost in enumerate(costs):
        if estimate_pass_mb([cost], cost_mb) > budget_mb:
            raise ValueError(f"input {index} alone exceeds the media pass budget of {budget_mb} MB")
        if current and estimate_pass_mb([costs[i] for i in current] + [cost], cost_mb) > budget_mb:
            chunks.append(current)
            current = []
        current.append(index)
    if current:
        chunks.append(current)
    return chunks


# --- request checks (before any decoding) -------------------------------------------------------------

def find_reserved_token(text: str | None) -> str | None:
    if not text:
        return None
    return next((token for token in RESERVED_TOKENS if token in text), None)


def _limit_error(limit: str, value: Any, maximum: Any, message: str, **extra: Any) -> MediaRequestError:
    return MediaRequestError("media_limit_exceeded", message, limit=limit, value=value, max=maximum, **extra)


def max_base64_chars(max_bytes: int) -> int:
    return math.ceil(max_bytes / 3) * 4 + BASE64_SLACK_CHARS


def _split_data_uri(value: str) -> tuple[str, bool]:
    """Return the base64 payload and whether the value was a valid base64 data URI (or plain base64)."""
    if not value.startswith("data:"):
        return value, True
    header, separator, payload = value.partition(",")
    if not separator or not header.lower().endswith(";base64"):
        return "", False
    return payload, True


def check_request(
    items: Sequence[tuple[int, MediaInputData]],
    *,
    model_id: str,
    model_modalities: Sequence[str],
    backend: str,
    limits: MediaLimits,
) -> dict[str, int]:
    """Support, enabled, count, item size and budget checks that need no decoding. items: (request index, input)."""
    used = [kind for kind in MEDIA_KINDS if any(item.count(kind) for _, item in items)]
    for kind in used:
        if kind not in model_modalities or backend != "pytorch":
            reason = "is text-only" if kind not in model_modalities else f"runs on {backend}, media need pytorch"
            raise MediaRequestError(
                "media_unsupported",
                f"Model '{model_id}' does not accept {kind} inputs here ({reason}).",
                model_id=model_id,
                modality=kind,
            )
    for kind in used:
        if not limits.enabled(kind):
            raise MediaRequestError("media_disabled", f"{kind} inputs are disabled on this server.", modality=kind)

    totals = {"images": 0, "videos": 0}
    max_chars = max_base64_chars(limits.max_media_item_mb * 1024 * 1024)
    for index, item in items:
        if len(item.image) > limits.max_images_per_input:
            raise _limit_error(
                "max_images_per_input", len(item.image), limits.max_images_per_input,
                f"Input {index} has {len(item.image)} images; at most {limits.max_images_per_input} are allowed.",
                index=index,
            )
        if len(item.audio) > limits.max_audio_clips_per_input:
            raise _limit_error(
                "max_audio_clips_per_input", len(item.audio), limits.max_audio_clips_per_input,
                f"Input {index} has {len(item.audio)} audio clips; at most {limits.max_audio_clips_per_input} are allowed.",
                index=index,
            )
        if item.video and len(item.video) > limits.max_video_frames:
            raise _limit_error(
                "max_video_frames", len(item.video), limits.max_video_frames,
                f"Input {index} has a video with {len(item.video)} frames; at most {limits.max_video_frames} are allowed.",
                index=index,
            )
        totals["images"] += len(item.image)
        totals["videos"] += item.count("video")
        for kind in MEDIA_KINDS:
            for position, value in enumerate(getattr(item, kind) or []):
                payload, _ = _split_data_uri(value)
                if len(payload) > max_chars:
                    raise _limit_error(
                        "max_media_item_mb", round(len(payload) * 3 / 4 / 1024 / 1024, 2), limits.max_media_item_mb,
                        f"Input {index} {kind}[{position}] is larger than {limits.max_media_item_mb} MB.",
                        index=index, field=kind, position=position,
                    )
    if totals["images"] > limits.max_images_per_request:
        raise _limit_error(
            "max_images_per_request", totals["images"], limits.max_images_per_request,
            f"The request has {totals['images']} images; at most {limits.max_images_per_request} are allowed.",
        )
    if totals["videos"] > limits.max_videos_per_request:
        raise _limit_error(
            "max_videos_per_request", totals["videos"], limits.max_videos_per_request,
            f"The request has {totals['videos']} videos; at most {limits.max_videos_per_request} are allowed.",
        )
    return totals


def check_pass_budget(index: int, cost: InputCost, limits: MediaLimits, cost_mb: Mapping[str, float]) -> None:
    estimate = estimate_pass_mb([cost], cost_mb)
    if estimate > limits.media_pass_budget_mb:
        raise _limit_error(
            "media_pass_budget_mb", round(estimate, 1), limits.media_pass_budget_mb,
            f"Input {index} needs an estimated {estimate:.0f} MB of GPU memory per pass; the budget is "
            f"{limits.media_pass_budget_mb} MB. Send fewer images, frames or audio seconds per input.",
            index=index,
        )


# --- decoding -------------------------------------------------------------------------------------------

def _invalid(index: int, kind: str, position: int, reason: str) -> MediaRequestError:
    return MediaRequestError(
        "invalid_media",
        f"Input {index} {kind}[{position}]: {reason}.",
        index=index,
        field=kind,
        position=position,
        reason=reason,
    )


def decode_base64(value: str, max_bytes: int) -> bytes:
    """Strict base64 (optionally as a data URI) with the size checked before decoding. Raises ValueError."""
    payload, ok = _split_data_uri(value)
    if not ok:
        raise ValueError("data URI must be base64 encoded")
    if not payload:
        raise ValueError("empty payload")
    if len(payload) > max_base64_chars(max_bytes):
        raise ValueError(f"larger than {max_bytes} bytes")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 ({exc})") from exc
    if len(data) > max_bytes:
        raise ValueError(f"larger than {max_bytes} bytes")
    return data


def jpeg_scan_count(data: bytes) -> int:
    """Scan (SOS) markers of a JPEG, without decoding.

    Header segments before the first scan are skipped by length (an EXIF thumbnail or ICC profile may contain
    FF DA bytes). From there on FF DA is always a marker: entropy-coded data byte-stuffs 0xFF as FF 00.
    """
    position = 2
    while position + 4 <= len(data) and data[position] == 0xFF:
        marker = data[position + 1]
        if marker == 0xFF:  # fill byte
            position += 1
        elif marker == 0xDA:
            return data.count(b"\xff\xda", position)
        else:
            position += 2 + int.from_bytes(data[position + 2 : position + 4], "big")
    return data.count(b"\xff\xda")


def _open_image(data: bytes, formats: Sequence[str]) -> Any:
    from PIL import Image

    if "JPEG" in formats and data[:2] == b"\xff\xd8":
        scans = jpeg_scan_count(data)
        if scans > MAX_JPEG_SCANS:
            raise ValueError(f"too many JPEG scans ({scans}, at most {MAX_JPEG_SCANS})")
    try:
        return Image.open(io.BytesIO(data), formats=list(formats))
    except Exception as exc:
        raise ValueError(f"not a {'/'.join(formats)} image") from exc


def reduced_size(size: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    """Aspect-preserving size within max_pixels; a side is never cut below MIN_VIDEO_FRAME_SIDE."""
    width, height = size
    if width * height <= max_pixels:
        return size
    scale = math.sqrt(max_pixels / (width * height))
    new_height = max(min(height, MIN_VIDEO_FRAME_SIDE), int(height * scale))
    new_width = max(min(width, MIN_VIDEO_FRAME_SIDE), min(int(width * scale), max_pixels // new_height))
    return new_width, new_height


def _finish_image(image: Any, keep_pixels: int) -> Any:
    from PIL import Image, ImageOps

    try:
        image.seek(0)  # GIF / animated WEBP: first frame
        target = reduced_size(image.size, keep_pixels)
        if target != image.size:
            image.draft(None, target)  # JPEG decodes at 1/2, 1/4 or 1/8 scale; other formats ignore it
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        target = reduced_size(image.size, keep_pixels)
        if target != image.size:
            image = image.resize(target, Image.Resampling.BICUBIC)
        return image
    except Exception as exc:
        raise ValueError(f"cannot decode image ({type(exc).__name__})") from exc


def decode_image(data: bytes, max_pixels: int) -> Any:
    """Raises ValueError for bad data, OverflowError when the header size exceeds max_pixels.

    Images over MAX_KEPT_IMAGE_PIXELS come back downscaled.
    """
    image = _open_image(data, IMAGE_FORMATS)
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image has no pixels")
    if width * height > max_pixels:
        raise OverflowError(width * height)
    return _finish_image(image, MAX_KEPT_IMAGE_PIXELS)


def decode_frame(data: bytes) -> Any:
    """Frames over MAX_KEPT_FRAME_PIXELS come back downscaled (equal input sizes give equal output sizes)."""
    image = _open_image(data, FRAME_FORMATS)
    width, height = image.size
    if min(width, height) < MIN_VIDEO_FRAME_SIDE:
        raise ValueError(f"frame {width}x{height} is under {MIN_VIDEO_FRAME_SIDE} pixels on a side")
    if width * height > MAX_VIDEO_FRAME_PIXELS:
        raise ValueError(f"frame {width}x{height} exceeds 1920x1080 pixels")
    return _finish_image(image, MAX_KEPT_FRAME_PIXELS)


def _mpeg_sync(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0


def is_allowed_audio_container(data: bytes) -> bool:
    """Magic bytes of WAV (RIFF/WAVE), FLAC, OGG or MP3 (ID3 tag or MPEG frame sync).

    Checked before libsndfile sees the data, so its parsers for other formats (AIFF, CAF, PAF, ...) never run.
    """
    if data[:3] == b"ID3" and len(data) >= 10:
        # ID3v2 header: syncsafe tag size, optional 10-byte footer; FLAC or MPEG audio must follow.
        size = 10 + ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 | (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
        if data[5] & 0x10:
            size += 10
        rest = data[size : size + 4]
        return rest == b"fLaC" or _mpeg_sync(rest)
    return (data[:4] == b"RIFF" and data[8:12] == b"WAVE") or data[:4] in (b"fLaC", b"OggS") or _mpeg_sync(data)


def read_audio_header(data: bytes) -> tuple[str, int, int, int]:
    """(format, sample rate, channels, frames) from the header, without reading samples."""
    import soundfile as sf

    if not is_allowed_audio_container(data):
        raise ValueError("not a WAV/FLAC/OGG/MP3 audio file")
    try:
        info = sf.info(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("not a WAV/FLAC/OGG/MP3 audio file") from exc
    return str(info.format).upper(), int(info.samplerate), int(info.channels), int(info.frames)


def decode_audio_samples(data: bytes, sample_rate: int, max_frames: int) -> np.ndarray:
    """Read at most max_frames + 1 frames (so a lying header cannot force a large read), downmix, resample to 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        samples, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True, frames=max_frames + 1)
    except Exception as exc:
        raise ValueError(f"cannot decode audio ({type(exc).__name__})") from exc
    mono = samples.mean(axis=1, dtype=np.float32) if samples.shape[1] > 1 else samples[:, 0]
    if not np.all(np.isfinite(mono)):
        raise ValueError("audio contains non-finite samples")
    if sample_rate != AUDIO_TARGET_RATE:
        divisor = gcd(AUDIO_TARGET_RATE, sample_rate)
        mono = resample_poly(mono, AUDIO_TARGET_RATE // divisor, sample_rate // divisor)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _decode_audio_clip(
    index: int, position: int, data: bytes, limits: MediaLimits, request_seconds: float
) -> np.ndarray:
    try:
        fmt, sample_rate, channels, frames = read_audio_header(data)
    except ValueError as exc:
        raise _invalid(index, "audio", position, str(exc)) from exc
    if fmt not in AUDIO_FORMATS:
        raise _invalid(index, "audio", position, f"audio format {fmt} is not allowed (WAV, FLAC, OGG, MP3)")
    if sample_rate not in AUDIO_SAMPLE_RATES:
        rates = ", ".join(str(rate) for rate in sorted(AUDIO_SAMPLE_RATES))
        raise _invalid(index, "audio", position, f"sample rate {sample_rate} Hz is not allowed ({rates})")
    if not 1 <= channels <= MAX_AUDIO_CHANNELS:
        raise _invalid(index, "audio", position, f"{channels} channels; mono or stereo only")

    def check_duration(seconds: float) -> None:
        if seconds > limits.max_audio_seconds_per_clip:
            raise _limit_error(
                "max_audio_seconds_per_clip", round(seconds, 3), limits.max_audio_seconds_per_clip,
                f"Input {index} audio[{position}] is {seconds:.1f} s long; at most "
                f"{limits.max_audio_seconds_per_clip} s per clip are allowed.",
                index=index, field="audio", position=position,
            )
        if seconds < MIN_AUDIO_SECONDS:
            raise _invalid(index, "audio", position, f"shorter than {MIN_AUDIO_SECONDS} s")
        if request_seconds + seconds > limits.max_audio_seconds_per_request:
            raise _limit_error(
                "max_audio_seconds_per_request", round(request_seconds + seconds, 3),
                limits.max_audio_seconds_per_request,
                f"The request has more than {limits.max_audio_seconds_per_request} s of audio.",
            )

    check_duration(frames / sample_rate)
    try:
        clip = decode_audio_samples(data, sample_rate, int(limits.max_audio_seconds_per_clip * sample_rate))
    except ValueError as exc:
        raise _invalid(index, "audio", position, str(exc)) from exc
    # The header frame count can be wrong (or an estimate for MP3); check what was actually decoded.
    check_duration(len(clip) / AUDIO_TARGET_RATE)
    return clip


def decode_input(index: int, item: MediaInputData, limits: MediaLimits, request_audio_seconds: float = 0.0) -> DecodedMedia:
    max_bytes = limits.max_media_item_mb * 1024 * 1024
    max_pixels = limits.max_image_megapixels * 1_000_000

    def raw(kind: str, position: int, value: str) -> bytes:
        try:
            return decode_base64(value, max_bytes)
        except ValueError as exc:
            raise _invalid(index, kind, position, str(exc)) from exc

    decoded = DecodedMedia(text=item.text)
    for position, value in enumerate(item.image):
        data = raw("image", position, value)  # outside the try: raw() already raises invalid_media
        try:
            decoded.images.append(decode_image(data, max_pixels))
        except OverflowError as exc:
            pixels = int(exc.args[0])
            raise _limit_error(
                "max_image_megapixels", round(pixels / 1_000_000, 2), limits.max_image_megapixels,
                f"Input {index} image[{position}] has {pixels / 1_000_000:.1f} megapixels; at most "
                f"{limits.max_image_megapixels} are allowed.",
                index=index, field="image", position=position,
            ) from exc
        except ValueError as exc:
            raise _invalid(index, "image", position, str(exc)) from exc
    for position, value in enumerate(item.audio):
        clip = _decode_audio_clip(index, position, raw("audio", position, value), limits, request_audio_seconds)
        request_audio_seconds += len(clip) / AUDIO_TARGET_RATE
        decoded.audio.append(clip)
    if item.video:
        frames = []
        for position, value in enumerate(item.video):
            data = raw("video", position, value)
            try:
                frame = decode_frame(data)
            except ValueError as exc:
                raise _invalid(index, "video", position, str(exc)) from exc
            if frames and frame.size != frames[0].size:
                raise _invalid(
                    index, "video", position,
                    f"frame size {frame.width}x{frame.height} differs from {frames[0].width}x{frames[0].height}",
                )
            frames.append(frame)
        decoded.video = frames
    return decoded


def decode_inputs(items: Sequence[Any], limits: MediaLimits) -> list[Any]:
    """Decode every MediaInputData in items (strings pass through). Raises MediaRequestError."""
    decoded: list[Any] = []
    audio_seconds = 0.0
    for index, item in enumerate(items):
        if isinstance(item, MediaInputData):
            result = decode_input(index, item, limits, audio_seconds)
            audio_seconds += result.audio_seconds
            decoded.append(result)
        else:
            decoded.append(item)
    return decoded


def media_summary(items: Sequence[Any]) -> dict[str, Any] | None:
    """Counts for responses and history; never payloads. None when the request has no media."""
    decoded = [item for item in items if isinstance(item, DecodedMedia)]
    raw = [item for item in items if isinstance(item, MediaInputData)]
    if decoded:
        return {
            "images": sum(len(item.images) for item in decoded),
            "audio_seconds": round(sum(item.audio_seconds for item in decoded), 3),
            "video_frames": sum(item.video_frames for item in decoded),
        }
    if raw:
        return {
            "images": sum(len(item.image) for item in raw),
            "audio_seconds": None,
            "video_frames": sum(len(item.video or []) for item in raw),
        }
    return None


def public_limits(limits: MediaLimits) -> dict[str, Any]:
    """Limits clients can discover via /health."""
    payload = {name: value for name, value in limits.as_dict().items() if not name.endswith("_enabled")}
    payload.update(
        {
            "max_video_frame_pixels": MAX_VIDEO_FRAME_PIXELS,
            "min_video_frame_side": MIN_VIDEO_FRAME_SIDE,
            "min_audio_seconds": MIN_AUDIO_SECONDS,
            "audio_sample_rates": sorted(AUDIO_SAMPLE_RATES),
            "audio_formats": ["WAV", "FLAC", "OGG", "MP3"],
            "image_formats": list(IMAGE_FORMATS),
            "video_frame_formats": list(FRAME_FORMATS),
        }
    )
    return payload
