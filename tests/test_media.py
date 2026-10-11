"""Model-free unit tests for backend/genesis_embed_server_media.py."""
from __future__ import annotations

import base64
import io
import random

import numpy as np
import pytest
from PIL import Image

from backend import genesis_embed_server_media as media
from backend.genesis_embed_server_globals import MODEL_SPECS
from media_samples import audio_bytes, b64, image_bytes, jpeg_scan_offsets, progressive_jpeg

COST_MB = MODEL_SPECS["google/embeddinggemma-2"]["media_cost_mb"]


def limits(**overrides) -> media.MediaLimits:
    return media.MediaLimits.from_settings(overrides)


# --- base64 / data URIs -----------------------------------------------------------------------------------

def test_base64_plain_and_data_uri():
    data = b"\x89PNG-ish bytes"
    assert media.decode_base64(b64(data), 1024) == data
    assert media.decode_base64("data:image/png;base64," + b64(data), 1024) == data
    assert media.decode_base64("data:application/octet-stream;BASE64," + b64(data), 1024) == data


@pytest.mark.parametrize(
    "value",
    [
        "data:image/png," + b64(b"abc"),  # not base64 encoded
        "data:image/png;base64",  # no payload separator
        "",
        "abc$def",  # invalid alphabet
        "YWJj\nZGVm",  # whitespace is not allowed in strict mode
        "YWJjZA",  # bad padding
        "YW-j",  # urlsafe alphabet
    ],
)
def test_base64_rejects_malformed(value):
    with pytest.raises(ValueError):
        media.decode_base64(value, 1024)


def test_base64_size_checked_before_decoding(monkeypatch):
    payload = b64(b"x" * 4096)

    def fail(*args, **kwargs):
        raise AssertionError("decoded an oversized payload")

    monkeypatch.setattr(media.base64, "b64decode", fail)
    with pytest.raises(ValueError, match="larger than"):
        media.decode_base64(payload, 1024)


def test_base64_size_rule_allows_exact_limit():
    assert len(media.decode_base64(b64(b"y" * 3000), 3000)) == 3000
    with pytest.raises(ValueError):
        media.decode_base64(b64(b"y" * 3001 + b"z" * 30), 3000)


# --- images -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "BMP", "GIF"])
def test_image_formats_decode_to_rgb(fmt):
    image = media.decode_image(image_bytes(40, 30, fmt=fmt), 10_000_000)
    assert image.mode == "RGB" and image.size == (40, 30)


def test_image_rgba_and_palette_become_rgb():
    buffer = io.BytesIO()
    Image.new("RGBA", (10, 10), (1, 2, 3, 4)).save(buffer, format="PNG")
    assert media.decode_image(buffer.getvalue(), 1_000_000).mode == "RGB"
    buffer = io.BytesIO()
    Image.new("P", (10, 10)).save(buffer, format="PNG")
    assert media.decode_image(buffer.getvalue(), 1_000_000).mode == "RGB"


def test_image_tiff_rejected():
    with pytest.raises(ValueError, match="not a"):
        media.decode_image(image_bytes(20, 20, fmt="TIFF"), 1_000_000)


def test_image_eps_like_bytes_rejected():
    eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"
    with pytest.raises(ValueError):
        media.decode_image(eps, 1_000_000)
    with pytest.raises(ValueError):
        media.decode_image(b"not an image at all", 1_000_000)


def test_image_pixel_cap_from_header_before_load(monkeypatch):
    data = image_bytes(1200, 1000)

    def fail(self, *args, **kwargs):
        raise AssertionError("pixels loaded before the size check")

    monkeypatch.setattr(Image.Image, "load", fail)
    from PIL import ImageFile

    monkeypatch.setattr(ImageFile.ImageFile, "load", fail)
    with pytest.raises(OverflowError):
        media.decode_image(data, 1_000_000)


def test_image_exif_orientation_applied():
    image = Image.new("RGB", (40, 20), (10, 200, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    assert media.decode_image(buffer.getvalue(), 1_000_000).size == (20, 40)


def test_image_gif_uses_first_frame():
    red = Image.new("RGB", (16, 16), (255, 0, 0))
    blue = Image.new("RGB", (16, 16), (0, 0, 255))
    buffer = io.BytesIO()
    red.save(buffer, format="GIF", save_all=True, append_images=[blue], duration=100, loop=0)
    pixel = media.decode_image(buffer.getvalue(), 1_000_000).getpixel((8, 8))
    assert pixel[0] > 200 and pixel[2] < 50


def test_jpeg_scan_count():
    data = progressive_jpeg(64, 48)
    assert media.jpeg_scan_count(data) == len(jpeg_scan_offsets(data)) > 1
    assert media.jpeg_scan_count(progressive_jpeg(64, 48, extra_scans=10)) == len(jpeg_scan_offsets(data)) + 10
    assert media.jpeg_scan_count(image_bytes(64, 48, fmt="JPEG")) == 1


def test_jpeg_with_many_scans_rejected_before_decoding(monkeypatch):
    data = progressive_jpeg(64, 48, extra_scans=media.MAX_JPEG_SCANS)

    def fail(*args, **kwargs):
        raise AssertionError("opened a JPEG with too many scans")

    monkeypatch.setattr(Image, "open", fail)
    for decode in (lambda: media.decode_image(data, 1_000_000), lambda: media.decode_frame(data)):
        with pytest.raises(ValueError, match="too many JPEG scans"):
            decode()


def test_jpeg_scan_bytes_in_header_segments_not_counted():
    # FF DA inside an APPn payload (EXIF thumbnail, ICC profile) is not a scan.
    data = progressive_jpeg(64, 48)
    payload = b"\xff\xda" * 200
    data = data[:2] + b"\xff\xef" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:]
    assert media.jpeg_scan_count(data) == len(jpeg_scan_offsets(progressive_jpeg(64, 48)))
    assert media.decode_image(data, 1_000_000).size == (64, 48)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_large_images_are_downscaled_after_decode(fmt):
    image = media.decode_image(image_bytes(4000, 3000, fmt=fmt), 25_000_000)
    assert image.mode == "RGB"
    assert image.width * image.height <= media.MAX_KEPT_IMAGE_PIXELS
    assert image.width * image.height > media.MAX_KEPT_IMAGE_PIXELS * 0.9
    assert abs(image.width / image.height - 4 / 3) < 0.01
    assert media.decode_image(image_bytes(1600, 1200), 25_000_000).size == (1600, 1200)  # under the cap


def test_decoded_request_does_not_pin_full_resolution():
    # 16 x 5000x5000 PNG: about 1.7 MB of base64, 1.2 GB of RGB at full resolution.
    item = media.MediaInputData(image=[b64(image_bytes(5000, 5000))] * 4)
    decoded = media.decode_inputs([item] * 4, limits())
    sizes = [image.size for result in decoded for image in result.images]
    assert len(sizes) == 16 and all(width * height <= media.MAX_KEPT_IMAGE_PIXELS for width, height in sizes)


def test_reduced_size():
    assert media.reduced_size((640, 480), 1_000_000) == (640, 480)
    width, height = media.reduced_size((4000, 3000), 1_000_000)
    assert width * height <= 1_000_000 and (width, height) == (1154, 866)
    width, height = media.reduced_size((50_000_000, 1), 1_000_000)  # extreme aspect: the short side stays
    assert height == 1 and width * height <= 1_000_000
    width, height = media.reduced_size((250_000, 10), 1_000_000)
    assert height == 8 and width * height <= 1_000_000


# --- audio ------------------------------------------------------------------------------------------------

def decode_clip(data: bytes, **overrides):
    item = media.MediaInputData(audio=[b64(data)])
    return media.decode_input(0, item, limits(**overrides)).audio[0]


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_audio_wav_16k(subtype):
    clip = decode_clip(audio_bytes(1.0, 16000, subtype=subtype))
    assert clip.dtype == np.float32 and clip.ndim == 1 and len(clip) == 16000


@pytest.mark.parametrize("fmt,subtype", [("FLAC", "PCM_16"), ("OGG", "VORBIS"), ("MP3", "MPEG_LAYER_III")])
def test_audio_compressed_formats(fmt, subtype):
    clip = decode_clip(audio_bytes(1.0, 16000, fmt=fmt, subtype=subtype))
    assert abs(len(clip) - 16000) <= 2000


def test_audio_stereo_downmix():
    clip = decode_clip(audio_bytes(0.5, 16000, channels=2, values=(0.5, -0.5)))
    assert len(clip) == 8000 and float(np.abs(clip).max()) < 1e-3


@pytest.mark.parametrize("rate", [8000, 11025, 22050, 24000, 32000, 44100, 48000])
def test_audio_resampled_to_16k(rate):
    clip = decode_clip(audio_bytes(1.0, rate))
    assert len(clip) == 16000
    assert clip.dtype == np.float32


def test_audio_odd_sample_rate_rejected():
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(audio_bytes(0.5, 44101))
    assert error.value.error == "invalid_media" and "44101" in error.value.message


def test_audio_three_channels_rejected():
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(audio_bytes(0.5, 16000, channels=3))
    assert error.value.error == "invalid_media"


def test_audio_header_long_duration_rejected_before_reading(monkeypatch):
    # 8 kHz 8-bit packs 40 s into ~320 KB; the header alone must reject it.
    data = audio_bytes(40.0, 8000, subtype="PCM_U8")

    def fail(*args, **kwargs):
        raise AssertionError("samples read for an over-long clip")

    monkeypatch.setattr(media, "decode_audio_samples", fail)
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(data)
    assert error.value.error == "media_limit_exceeded"
    assert error.value.extra["limit"] == "max_audio_seconds_per_clip"
    assert error.value.extra["max"] == 30


def test_audio_lying_header_is_bounded(monkeypatch):
    # A header that under-reports a real 40 s clip as 1 s (like an MP3 frame estimate) passes the header check;
    # the read is still bounded to 30 s + 1 frame and the decoded length is checked again.
    import soundfile

    monkeypatch.setattr(media, "read_audio_header", lambda data: ("WAV", 16000, 1, 16000))
    reads = []
    real_read = soundfile.read

    def bounded_read(*args, **kwargs):
        reads.append(kwargs.get("frames"))
        return real_read(*args, **kwargs)

    monkeypatch.setattr(soundfile, "read", bounded_read)
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(audio_bytes(40.0, 16000))
    assert reads == [30 * 16000 + 1]
    assert error.value.error == "media_limit_exceeded"
    assert error.value.extra["limit"] == "max_audio_seconds_per_clip" and error.value.extra["value"] == 30.0


def test_audio_too_short_rejected():
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(audio_bytes(0.05, 16000))
    assert error.value.error == "invalid_media" and "shorter" in error.value.message


@pytest.mark.parametrize("fmt", ["AIFF", "CAF", "W64", "NIST", "VOC", "IRCAM", "MAT5", "SVX", "RF64", "AU", "PAF", "HTK"])
def test_audio_non_whitelisted_container_never_reaches_libsndfile(monkeypatch, fmt):
    import soundfile

    data = audio_bytes(0.5, 16000, fmt=fmt)

    def fail(*args, **kwargs):
        raise AssertionError("libsndfile parsed a non-whitelisted container")

    monkeypatch.setattr(soundfile, "info", fail)
    monkeypatch.setattr(soundfile, "read", fail)
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(data)
    assert error.value.error == "invalid_media" and "WAV/FLAC/OGG/MP3" in error.value.message


@pytest.mark.parametrize(
    "fmt,subtype", [("WAV", "PCM_16"), ("WAVEX", "PCM_16"), ("FLAC", "PCM_16"), ("OGG", "VORBIS"), ("MP3", "MPEG_LAYER_III")]
)
def test_audio_whitelisted_magic_bytes(fmt, subtype):
    assert media.is_allowed_audio_container(audio_bytes(0.2, 16000, fmt=fmt, subtype=subtype))


def test_audio_id3_tag_must_precede_mp3_or_flac():
    flac = audio_bytes(0.2, 16000, fmt="FLAC")
    aiff = audio_bytes(0.2, 16000, fmt="AIFF")
    tag = b"ID3\x04\x00\x00" + bytes([0, 0, 0, 6]) + b"\x00" * 6
    assert media.is_allowed_audio_container(tag + flac)
    assert not media.is_allowed_audio_container(tag + aiff)
    assert not media.is_allowed_audio_container(b"ID3")


def test_audio_garbage_rejected():
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(b"RIFF....WAVEfmt garbage")
    assert error.value.error == "invalid_media"


def test_audio_request_total_enforced_across_inputs():
    items = [media.MediaInputData(audio=[b64(audio_bytes(2.0))]) for _ in range(2)]
    with pytest.raises(media.MediaRequestError) as error:
        media.decode_inputs(items, limits(max_audio_seconds_per_request=3))
    assert error.value.extra["limit"] == "max_audio_seconds_per_request"


def test_audio_custom_clip_limit():
    with pytest.raises(media.MediaRequestError) as error:
        decode_clip(audio_bytes(3.0), max_audio_seconds_per_clip=2)
    assert error.value.extra["limit"] == "max_audio_seconds_per_clip"
    assert error.value.extra["position"] == 0 and error.value.extra["field"] == "audio"


# --- video frames -----------------------------------------------------------------------------------------

def test_frames_decode_same_size():
    item = media.MediaInputData(video=[b64(image_bytes(64, 48, fmt=fmt)) for fmt in ("PNG", "JPEG", "WEBP")])
    decoded = media.decode_input(0, item, limits())
    assert decoded.video_frames == 3 and all(frame.size == (64, 48) for frame in decoded.video)


def test_frames_size_mismatch_rejected():
    item = media.MediaInputData(video=[b64(image_bytes(64, 48)), b64(image_bytes(48, 64))])
    with pytest.raises(media.MediaRequestError) as error:
        media.decode_input(2, item, limits())
    assert error.value.error == "invalid_media"
    assert error.value.extra == {"index": 2, "field": "video", "position": 1, "reason": error.value.extra["reason"]}


def test_frame_pixel_cap():
    with pytest.raises(ValueError, match="1920x1080"):
        media.decode_frame(image_bytes(1921, 1081))
    assert media.decode_frame(image_bytes(1920, 1080)).size == media.reduced_size((1920, 1080), media.MAX_KEPT_FRAME_PIXELS)
    assert media.decode_frame(image_bytes(1080, 1920)).size == media.reduced_size((1080, 1920), media.MAX_KEPT_FRAME_PIXELS)


@pytest.mark.parametrize("width,height", [(97, 1), (1, 97), (2000, 1), (200, 7), (7, 7)])
def test_degenerate_frames_rejected(width, height):
    # 1 px high frames make the processor's token count differ from the formula (it then fails with a 500).
    with pytest.raises(ValueError, match="under 8 pixels"):
        media.decode_frame(image_bytes(width, height))
    assert media.decode_frame(image_bytes(max(width, 8), 8)).size == (max(width, 8), 8)


def test_large_frames_downscaled_equally():
    frames = [media.decode_frame(image_bytes(1920, 1080, (index, 0, 0), fmt=fmt)) for index, fmt in enumerate(["PNG", "JPEG", "WEBP"])]
    assert len({frame.size for frame in frames}) == 1
    width, height = frames[0].size
    assert width * height <= media.MAX_KEPT_FRAME_PIXELS and abs(width / height - 16 / 9) < 0.01


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize("value,reason", [("data:image/png;base64,", "empty payload"), ("data:image/png,AAAA", "data URI must be base64 encoded")])
def test_bad_base64_error_not_wrapped_twice(kind, value, reason):
    item = media.MediaInputData(**{kind: [value]})
    with pytest.raises(media.MediaRequestError) as error:
        media.decode_input(3, item, limits())
    assert error.value.message == f"Input 3 {kind}[0]: {reason}."
    assert error.value.extra == {"index": 3, "field": kind, "position": 0, "reason": reason}


def test_frames_reject_gif_and_bmp():
    for fmt in ("GIF", "BMP"):
        with pytest.raises(ValueError):
            media.decode_frame(image_bytes(32, 32, fmt=fmt))


def test_image_megapixel_limit_reported():
    item = media.MediaInputData(image=[b64(image_bytes(1500, 1000))])
    with pytest.raises(media.MediaRequestError) as error:
        media.decode_input(1, item, limits(max_image_megapixels=1))
    assert error.value.error == "media_limit_exceeded"
    assert error.value.extra["limit"] == "max_image_megapixels" and error.value.extra["value"] == 1.5


# --- token formula ----------------------------------------------------------------------------------------

def test_token_formula_known_values():
    # Values measured with the real processor (sequence minus <bos>/<eos>).
    assert media.image_tokens(600, 800) == 268
    assert media.image_tokens(480, 640) == 268
    assert media.audio_tokens(2 * 16000) == 52
    assert media.audio_tokens(30 * 16000) == 752
    assert media.audio_tokens(60 * 16000) == 752  # the feature extractor stops at 30 s
    assert media.video_tokens(4, 480, 640) == 528
    assert media.video_tokens(40, 480, 640) == 32 * 132


def test_token_formula_bounds():
    for height, width in [(1, 1), (10, 5000), (5000, 10), (4000, 3000), (1080, 1920), (37, 2000)]:
        assert 2 < media.image_tokens(height, width) <= 282
        assert 2 < media.video_tokens(1, height, width) <= 142
    assert media.video_tokens(16, 1080, 1920) <= 16 * 142


def test_input_tokens_adds_specials_and_media():
    item = media.DecodedMedia(
        text="x",
        images=[Image.new("RGB", (800, 600))],
        audio=[np.zeros(32000, np.float32)],
        video=[Image.new("RGB", (640, 480))] * 4,
    )
    assert media.input_tokens(3, item) == 2 + 3 + 268 + 52 + 528


def test_st_input_key_order_and_shapes():
    clip = np.zeros(1600, np.float32)
    frame = Image.new("RGB", (8, 8))
    item = media.DecodedMedia(text="t", images=[frame], audio=[clip], video=[frame, frame])
    payload = item.to_st_input()
    assert list(payload) == ["text", "image", "audio", "video"]
    assert payload["audio"][0]["sampling_rate"] == 16000 and payload["audio"][0]["array"] is clip
    assert payload["video"] == [frame, frame]
    assert list(media.DecodedMedia(images=[frame]).to_st_input()) == ["image"]


# --- cost model and chunking ------------------------------------------------------------------------------

def test_estimate_pass_mb():
    assert media.estimate_pass_mb([], COST_MB) == 0
    assert media.estimate_pass_mb([media.InputCost(images=1, tokens=270)], COST_MB) == pytest.approx(74 + 50)
    assert media.estimate_pass_mb([media.InputCost(video_frames=16, tokens=2274)], COST_MB) == pytest.approx(576 + 50)
    assert media.estimate_pass_mb([media.InputCost(audio_clips=4, longest_clip_seconds=30)], COST_MB) == pytest.approx(420 + 50)
    # LM term: rows x longest
    costs = [media.InputCost(tokens=4096), media.InputCost(tokens=10)]
    assert media.estimate_pass_mb(costs, COST_MB) == pytest.approx(0.125 * 2 * 4096 + 50)


def test_estimate_pads_audio_to_longest_clip():
    # The feature extractor pads every clip of a pass to the longest one: 1 x 30 s + 11 x (2 x 0.1 s) runs as
    # 23 clips of 30 s, not as 32.2 s of audio.
    long_clip = media.InputCost(audio_clips=1, longest_clip_seconds=30, tokens=754)
    short_clips = media.InputCost(audio_clips=2, longest_clip_seconds=0.1, tokens=12)
    assert media.estimate_pass_mb([long_clip] + [short_clips] * 11, COST_MB) == pytest.approx(3.5 * 23 * 30 + 50)
    assert media.estimate_pass_mb([short_clips] * 11, COST_MB) == pytest.approx(0.125 * 11 * 12 + 50)
    # Inside one input too: 1 x 30 s + 3 x 0.1 s is 4 padded clips.
    clips = [np.zeros(30 * 16000, np.float32)] + [np.zeros(1600, np.float32)] * 3
    cost = media.item_cost(media.DecodedMedia(audio=clips), 800)
    assert (cost.audio_clips, cost.longest_clip_seconds) == (4, 30.0)
    assert media.estimate_pass_mb([cost], COST_MB) == pytest.approx(3.5 * 4 * 30 + 50)


def test_chunker_splits_long_clip_from_many_short_ones():
    costs = [media.InputCost(audio_clips=1, longest_clip_seconds=30, tokens=754)]
    costs += [media.InputCost(audio_clips=2, longest_clip_seconds=0.1, tokens=12)] * 11
    chunks = media.plan_chunks(costs, 1200, COST_MB)
    assert len(chunks) > 1 and [index for chunk in chunks for index in chunk] == list(range(12))
    # The pass with the 30 s clip: (1 + 2k) padded 30 s clips must fit, so at most 5 short inputs join it.
    assert chunks[0] == [0, 1, 2, 3, 4]
    for chunk in chunks:
        assert padded_reference_mb([costs[index] for index in chunk]) <= 1200


def padded_reference_mb(costs) -> float:
    """Independent restatement of the cost model: every clip of a pass costs the longest clip's seconds."""
    clips = sum(cost.audio_clips for cost in costs)
    longest = max(cost.longest_clip_seconds for cost in costs)
    tower = 74 * sum(cost.images for cost in costs) + 36 * sum(cost.video_frames for cost in costs) + 3.5 * clips * longest
    return max(tower, 0.125 * len(costs) * max(cost.tokens for cost in costs)) + 50


def random_allowed_cost(rng: random.Random) -> media.InputCost:
    images = rng.randint(0, 8)
    frames = rng.choice([0, 0, rng.randint(1, 16)])
    clips = [rng.choice([0.1, rng.uniform(0.1, 30), 30.0]) for _ in range(rng.randint(0, 4))]
    if not (images or frames or clips):
        images = 1
    tokens = 2 + rng.randint(0, 1500) + images * 282 + frames * 142 + int(sum(clips) * 25) + len(clips) * 2
    return media.InputCost(images, frames, len(clips), max(clips, default=0.0), min(tokens, 4096))


def test_chunker_respects_budget_and_order():
    rng = random.Random(1234)
    for _ in range(500):
        budget = rng.choice([1200, 1200, 800, 2000, 4096])
        costs = [cost for cost in (random_allowed_cost(rng) for _ in range(rng.randint(1, 12)))
                 if media.estimate_pass_mb([cost], COST_MB) <= budget]
        if not costs:
            continue
        chunks = media.plan_chunks(costs, budget, COST_MB)
        assert [index for chunk in chunks for index in chunk] == list(range(len(costs)))
        for chunk in chunks:
            assert chunk and padded_reference_mb([costs[index] for index in chunk]) <= budget


def test_chunker_groups_small_inputs():
    costs = [media.InputCost(images=1, tokens=300)] * 10
    chunks = media.plan_chunks(costs, 1200, COST_MB)
    # 10 x 74 + 50 = 790 <= 1200 (LM: 0.125 x 10 x 300 + 50 = 425).
    assert [len(chunk) for chunk in chunks] == [10]
    chunks = media.plan_chunks([media.InputCost(images=4, tokens=1200)] * 5, 1200, COST_MB)
    assert all(media.estimate_pass_mb([media.InputCost(images=4, tokens=1200)] * len(chunk), COST_MB) <= 1200 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == 5 and len(chunks) == 2


def test_chunker_rejects_single_over_budget_input():
    with pytest.raises(ValueError):
        media.plan_chunks([media.InputCost(images=20)], 1200, COST_MB)


def test_default_limits_fit_budget():
    # The worst input allowed by the default limits must fit the default budget.
    defaults = limits()
    worst = media.InputCost(
        images=defaults.max_images_per_input,
        video_frames=defaults.max_video_frames,
        audio_clips=defaults.max_audio_clips_per_input,
        longest_clip_seconds=defaults.max_audio_seconds_per_clip,
        tokens=4096,
    )
    assert media.estimate_pass_mb([worst], COST_MB) <= defaults.media_pass_budget_mb


# --- request checks ---------------------------------------------------------------------------------------

def check(items, backend="pytorch", modalities=("image", "audio", "video"), **overrides):
    return media.check_request(
        list(enumerate(items)), model_id="m", model_modalities=list(modalities), backend=backend, limits=limits(**overrides)
    )


def test_check_request_support_and_enabled():
    with pytest.raises(media.MediaRequestError) as error:
        check([media.MediaInputData(image=["AAAA"])], modalities=())
    assert error.value.error == "media_unsupported" and error.value.extra == {"model_id": "m", "modality": "image"}
    with pytest.raises(media.MediaRequestError) as error:
        check([media.MediaInputData(audio=["AAAA"])], backend="openvino")
    assert error.value.error == "media_unsupported"
    with pytest.raises(media.MediaRequestError) as error:
        check([media.MediaInputData(video=["AAAA"])], video_inputs_enabled=False)
    assert error.value.error == "media_disabled" and error.value.extra == {"modality": "video"}
    assert check([media.MediaInputData(image=["AAAA"])], video_inputs_enabled=False) == {"images": 1, "videos": 0}


@pytest.mark.parametrize(
    "items,overrides,limit",
    [
        ([media.MediaInputData(image=["AAAA"] * 5)], {}, "max_images_per_input"),
        ([media.MediaInputData(image=["AAAA"] * 4)] * 5, {}, "max_images_per_request"),
        ([media.MediaInputData(audio=["AAAA"] * 3)], {}, "max_audio_clips_per_input"),
        ([media.MediaInputData(video=["AAAA"] * 9)], {}, "max_video_frames"),
        ([media.MediaInputData(video=["AAAA"])] * 2, {}, "max_videos_per_request"),
        ([media.MediaInputData(image=[b64(b"x" * (1024 * 1024 + 100))])], {"max_media_item_mb": 1}, "max_media_item_mb"),
    ],
)
def test_check_request_limits(items, overrides, limit):
    with pytest.raises(media.MediaRequestError) as error:
        check(items, **overrides)
    assert error.value.error == "media_limit_exceeded" and error.value.extra["limit"] == limit
    assert error.value.status_code == 400


def test_limits_clamped_from_settings():
    clamped = limits(max_video_frames=500, media_pass_budget_mb=1, max_images_per_input="junk")
    assert clamped.max_video_frames == 16 and clamped.media_pass_budget_mb == 200 and clamped.max_images_per_input == 4


def test_reserved_tokens():
    assert media.find_reserved_token("a <|audio|> b") == "<|audio|>"
    assert media.find_reserved_token("<|image> is fine") is None
    assert media.find_reserved_token(None) is None


def test_media_summary_never_contains_payload():
    raw = media.MediaInputData(text="t", image=["QUJD" * 100], video=["QUJD"] * 3)
    summary = media.media_summary(["text", raw])
    assert summary == {"images": 1, "audio_seconds": None, "video_frames": 3}
    assert media.media_summary(["only text"]) is None
    assert base64.b64decode("QUJD") == b"ABC"
