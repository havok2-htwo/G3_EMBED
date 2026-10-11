"""The model-free token formula must equal the real EmbeddingGemma 2 processor (CPU, no forward pass).

Skipped when the model files are not in models/. Loads only the processor (tokenizer, image/video
processors, audio feature extractor), never the weights.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from backend import genesis_embed_server_media as media
from conftest import model_files_cached
from media_samples import audio_bytes, b64, image_bytes

SNAPSHOT = model_files_cached()
pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="google/embeddinggemma-2 not in models/")


@pytest.fixture(scope="module")
def processor():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(SNAPSHOT))


def real_features(processor, batch: list[media.DecodedMedia]):
    """Processor output as sentence-transformers builds it: one user message per input, parts in key order."""
    messages = []
    for item in batch:
        content = []
        for kind, value in item.to_st_input().items():
            if kind == "text":
                content.append({"type": "text", "text": value})
            elif kind == "video":
                content.append({"type": "video", "video": value})
            else:
                for part in value:
                    content.append({"type": kind, kind: part["array"] if kind == "audio" else part})
        messages.append([{"role": "user", "content": content}])
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"text_kwargs": {"padding": True}, "audio_kwargs": {"sampling_rate": 16000}},
    )


def real_tokens(processor, batch: list[media.DecodedMedia]) -> list[int]:
    return real_features(processor, batch)["attention_mask"].sum(1).tolist()


def predicted(processor, item: media.DecodedMedia) -> int:
    text_tokens = len(processor.tokenizer(item.text, add_special_tokens=False)["input_ids"]) if item.text else 0
    return media.input_tokens(text_tokens, item)


IMAGE_SIZES = [
    (640, 480), (480, 640), (800, 600), (1920, 1080), (1080, 1920), (4000, 3000), (100, 100), (300, 1200), (2000, 37),
    (17, 900), (97, 1), (1, 97), (5000, 1), (1, 1),
]


@pytest.mark.parametrize("width,height", IMAGE_SIZES)
def test_image_tokens_match_processor(processor, width, height):
    # Through the real decoder, so the counted size is the decoded size.
    image = media.decode_image(image_bytes(width, height, fmt="PNG"), 50_000_000)
    item = media.DecodedMedia(images=[image])
    assert real_tokens(processor, [item]) == [predicted(processor, item)]


# Includes the smallest frames decode_frame accepts (MIN_VIDEO_FRAME_SIDE); 1 px high frames do not match and are
# rejected there.
VIDEO_SIZES = [
    (1, 640, 480), (4, 640, 480), (8, 1280, 720), (16, 640, 480), (16, 1920, 1080), (16, 1080, 1920), (3, 320, 240),
    (2, 97, 8), (2, 8, 97), (2, 8, 8), (2, 1920, 8), (2, 8, 1080),
]


@pytest.mark.parametrize("frames,width,height", VIDEO_SIZES)
def test_video_tokens_match_processor(processor, frames, width, height):
    video = [media.decode_frame(image_bytes(width, height, (index * 10, 0, 0))) for index in range(frames)]
    item = media.DecodedMedia(video=video)
    assert real_tokens(processor, [item]) == [predicted(processor, item)]


@pytest.mark.parametrize("seconds,rate", [(0.1, 16000), (1.0, 16000), (2.5, 44100), (11.2, 48000), (30.0, 8000), (7.3, 22050)])
def test_audio_tokens_match_processor(processor, seconds, rate):
    clip = media.decode_input(0, media.MediaInputData(audio=[b64(audio_bytes(seconds, rate))]), media.MediaLimits.from_settings({})).audio[0]
    item = media.DecodedMedia(audio=[clip])
    assert real_tokens(processor, [item]) == [predicted(processor, item)]


def test_combined_inputs_match_processor_in_one_batch(processor):
    image = Image.new("RGB", (800, 600))
    frames = [Image.new("RGB", (640, 480))] * 4
    batch = [
        media.DecodedMedia(text="task: search result | query: red shoe", images=[image, Image.new("RGB", (300, 1200))]),
        media.DecodedMedia(text="clip", images=[image], audio=[np.zeros(16000 * 2, np.float32)], video=frames),
        media.DecodedMedia(audio=[np.zeros(16000 * 3 + 123, np.float32), np.zeros(8000, np.float32)]),
        media.DecodedMedia(text="title: none | text: Ein Satz mit Umlauten: äöü", video=frames[:2]),
    ]
    assert real_tokens(processor, batch) == [predicted(processor, item) for item in batch]


def test_audio_clips_padded_to_longest_in_a_pass(processor):
    # The cost model (estimate_pass_mb) charges every clip of a pass at the longest clip's length.
    long_clip = np.zeros(16000 * 30, np.float32)
    short_clip = np.zeros(1600, np.float32)
    batch = [media.DecodedMedia(audio=[long_clip])] + [media.DecodedMedia(audio=[short_clip, short_clip])] * 3
    features = real_features(processor, batch)
    padded = features["input_features"].shape
    single = real_features(processor, [batch[0]])["input_features"].shape
    assert padded[0] == 7 and padded[1] == single[1]
