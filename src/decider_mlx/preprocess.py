"""Text and image preprocessing without torch or transformers: the only module that tokenizes or resizes.

It reproduces, bit for bit, what transformers 5.17 gave the engine (AutoTokenizer / Qwen2Tokenizer and
Qwen3VLProcessor with Qwen2VLImageProcessor on its torchvision backend, torch 2.14 CPU):
  - text: the `tokenizers` library on the model's tokenizer.json; the chat template as fixed strings
    (`context_text`, identical to chat_template.jinja of both models for [system, user(images..., text)] with
    enable_thinking=False and add_generation_prompt=False); each <|image_pad|> expanded after tokenization to
    grid.prod() // merge**2 tokens.
  - images: smart_resize to a multiple of 32 within [65536, max_pixels] pixels; torchvision resize(uint8, BICUBIC,
    antialias=True), i.e. separable fixed-point resampling with int16 weights (horizontal pass first, uint8
    intermediate); rescale + normalize fused as (x - 127.5) / 127.5 in float32; patches (sum h*w, 3*2*16*16)
    float32 and image_grid_thw (n, 3) int64.
tests/reference_preprocess.py holds a numpy reference of the image path.
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- image

def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=1638400):
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize (same float ops)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _cubic(x):
    # torch HelperInterpCubic::aa_filter, a = -0.5 (cubic_convolution1 / cubic_convolution2), in double
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


_WEIGHTS = {}


def int16_weights(in_size, out_size):
    """torch _compute_index_ranges_int16_weights for bicubic antialias, align_corners=False.
    Returns (index (out, k) int64 clipped to the input, weights (out, k) int32 with zeros past xsize, precision)."""
    key = (in_size, out_size)
    if key in _WEIGHTS:
        return _WEIGHTS[key]
    scale = in_size / out_size
    support = (4 * 0.5) * scale if scale >= 1.0 else 4 * 0.5
    k = int(math.ceil(support)) * 2 + 1
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    w = np.zeros((out_size, k), np.float64)
    xmins = np.zeros(out_size, np.int64)
    wt_max = 0.0
    for i in range(out_size):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(int(center + support + 0.5), in_size) - xmin
        xsize = min(max(xsize, 0), k)
        row = [_cubic((j + xmin - center + 0.5) * invscale) for j in range(xsize)]
        total = 0.0
        for v in row:
            total += v
        if total != 0.0:
            row = [v / total for v in row]
            wt_max = max(wt_max, *row)
        w[i, :xsize] = row
        xmins[i] = xmin
    precision = 0
    while precision < 22:
        if int(0.5 + wt_max * (1 << (precision + 1))) >= (1 << 15):
            break
        precision += 1
    v = w * (1 << precision)
    wi = np.where(v < 0, v - 0.5, v + 0.5).astype(np.int64).astype(np.int16).astype(np.int32)  # C cast truncates toward 0
    index = np.minimum(xmins[:, None] + np.arange(k)[None, :], in_size - 1)
    _WEIGHTS[key] = (index, wi, precision)
    return _WEIGHTS[key]


# (x - mean/rescale) / (std/rescale) in float32 for x in 0..255: torch fuses rescale into mean and std (127.5 each).
_MEAN = np.float32(np.float32(0.5) * np.float32(1.0 / 0.00392156862745098))
_STD = np.float32(np.float32(0.5) * np.float32(1.0 / 0.00392156862745098))
NORM = ((np.arange(256, dtype=np.float32) - _MEAN) / _STD).astype(np.float32)


class ImageProcessor:
    """Qwen2VLImageProcessor (patch 16, merge 2, temporal 2, mean = std = 0.5, BICUBIC) for RGB inputs."""

    def __init__(self, config=None, *, min_pixels=65536, max_pixels=1638400):
        c = config or {}
        self.patch = c.get("patch_size", 16)
        self.merge = c.get("merge_size", 2)
        self.temporal = c.get("temporal_patch_size", 2)
        if (c.get("image_mean", [0.5] * 3), c.get("image_std", [0.5] * 3), c.get("resample", 3),
                c.get("rescale_factor", 1 / 255)) != ([0.5] * 3, [0.5] * 3, 3, 0.00392156862745098):
            raise ValueError("Only mean = std = 0.5, BICUBIC and rescale 1/255 are implemented")
        self.min_pixels, self.max_pixels = min_pixels, max_pixels

    def target(self, height, width):
        return smart_resize(height, width, self.patch * self.merge, self.min_pixels, self.max_pixels)


# ---------------------------------------------------------------- text


class Tokenizer:
    """The parts of transformers Qwen2Tokenizer the engines use, on tokenizers.Tokenizer (same tokenizer.json)."""

    def __init__(self, directory):
        from tokenizers import Tokenizer as T
        directory = Path(directory)
        self.backend = T.from_file(str(directory / "tokenizer.json"))
        config = json.loads((directory / "tokenizer_config.json").read_text())
        names = [config.get(k) for k in ("bos_token", "eos_token", "unk_token", "pad_token")]
        names += list((config.get("model_specific_special_tokens") or {}).values())
        names += list(config.get("extra_special_tokens") or [])
        self.all_special_tokens = list(dict.fromkeys(n for n in names if n))

    def get_vocab(self):
        return self.backend.get_vocab(with_added_tokens=True)

    def token_id(self, token):
        return self.backend.token_to_id(token)

    def add_special_tokens(self, tokens):
        from tokenizers import AddedToken
        self.backend.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in tokens])
        self.all_special_tokens += [t for t in tokens if t not in self.all_special_tokens]

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_offsets(self, text):
        e = self.backend.encode(text, add_special_tokens=False)
        return e.ids, e.offsets


IM_START, IM_END = "<|im_start|>", "<|im_end|>"
VISION = "<|vision_start|><|image_pad|><|vision_end|>"


def context_text(system, state_text, n_images):
    """chat_template.jinja (Flash and 27B) for [system, user(images..., text)] with enable_thinking=False and
    add_generation_prompt=False, cut before the text end (so the trailing |trim never applies)."""
    if system != system.strip():
        raise ValueError("system text must not need trimming")
    return f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n" + VISION * n_images + state_text


# ---------------------------------------------------------------- MLX image path

_SRC_PASS = """
    uint g = thread_position_in_grid.x;
    int n_in = p[0], n_out = p[1], k = p[2], prec = p[3], inner = p[4];   // inner = W*3 (vertical) or 3 (horizontal)
    int lanes = p[5];                                                        // elements per outer step of the output
    if ((int)g >= p[6]) return;
    int outer = g / lanes, rest = g % lanes;
    int o, base;
    if (p[7] == 1) {               // horizontal: rest = x*3 + c, outer = row
        o = rest / 3; int c = rest % 3;
        base = outer * n_in * 3 + c;
    } else {                       // vertical: outer = output row, rest = x*3 + c
        o = outer; base = rest;
    }
    int acc = 1 << (prec - 1);
    int start = xmin[o];
    for (int j = 0; j < k; j++) {
        int i = min(start + j, n_in - 1);
        acc += int(src[base + i * inner]) * w[o * k + j];
    }
    acc >>= prec;
    out[g] = uint8_t(clamp(acc, 0, 255));
"""

_SRC_PATCH = """
    uint g = thread_position_in_grid.x;
    if ((int)g >= p[2]) return;
    int width = p[0], gw2 = p[1] / 2;
    int n = g / 1536, k = g % 1536;
    int c = k / 512, pp = k % 256, py = pp / 16, px = pp % 16;
    int blk = n / 4, sub = n % 4;
    int y = ((blk / gw2) * 2 + sub / 2) * 16 + py, x = ((blk % gw2) * 2 + sub % 2) * 16 + px;
    out[g] = lut[src[(y * width + x) * 3 + c]];
"""

_KERNELS = {}


def _kernel(name):
    if name not in _KERNELS:
        import mlx.core as mx
        if name == "pass":
            _KERNELS[name] = mx.fast.metal_kernel(name="decider_resample_pass", input_names=["src", "xmin", "w", "p"],
                                                  output_names=["out"], source=_SRC_PASS)
        else:
            _KERNELS[name] = mx.fast.metal_kernel(name="decider_patchify", input_names=["src", "lut", "p"],
                                                  output_names=["out"], source=_SRC_PATCH)
    return _KERNELS[name]


_MX_WEIGHTS = {}


def _mx_weights(n_in, n_out):
    import mlx.core as mx
    key = (n_in, n_out)
    if key not in _MX_WEIGHTS:
        index, w, precision = int16_weights(n_in, n_out)
        _MX_WEIGHTS[key] = (mx.array(index[:, 0].astype(np.int32)), mx.array(w.astype(np.int32).ravel()), w.shape[1], precision)
        if len(_MX_WEIGHTS) > 256:
            _MX_WEIGHTS.pop(next(iter(_MX_WEIGHTS)))
    return _MX_WEIGHTS[key]


def _mx_pass(src, h, w, axis, n_out):
    import mlx.core as mx
    n_in = w if axis == 1 else h
    xmin, wt, k, precision = _mx_weights(n_in, n_out)
    oh, ow = (h, n_out) if axis == 1 else (n_out, w)
    total = oh * ow * 3
    lanes = ow * 3
    inner = 3 if axis == 1 else w * 3
    p = mx.array([n_in, n_out, k, precision, inner, lanes, total, 1 if axis == 1 else 0], dtype=mx.int32)
    out = _kernel("pass")(inputs=[src, xmin, wt, p], grid=(total, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(total,)], output_dtypes=[mx.uint8])[0]
    return out, oh, ow


_MX_LUT = []


class MLXImageProcessor(ImageProcessor):
    """The resize and the patch layout as two Metal kernels (integer arithmetic and a float32 table: the same
    values as the fixed-point reference). Runs on the engine worker thread. Returns numpy arrays."""

    def patches_mx(self, rgb):
        import mlx.core as mx
        if (self.patch, self.merge, self.temporal) != (16, 2, 2):
            raise ValueError("The Metal patch kernel is written for patch 16, merge 2, temporal 2")
        if not _MX_LUT:
            _MX_LUT.append(mx.array(NORM))
        h0, w0 = rgb.shape[:2]
        h, w = self.target(h0, w0)
        x = mx.array(np.ascontiguousarray(rgb).reshape(-1))
        ch, cw = h0, w0
        if cw != w:
            x, ch, cw = _mx_pass(x, ch, cw, 1, w)
        if ch != h:
            x, ch, cw = _mx_pass(x, ch, cw, 0, h)
        gh, gw = h // 16, w // 16
        total = gh * gw * 1536
        p = mx.array([w, gw, total], dtype=mx.int32)
        out = _kernel("patch")(inputs=[x, _MX_LUT[0], p], grid=(total, 1, 1), threadgroup=(256, 1, 1),
                               output_shapes=[(gh * gw, 1536)], output_dtypes=[mx.float32])[0]
        return out, (1, gh, gw)

    def __call__(self, images):
        import mlx.core as mx
        outs, grid = [], []
        for image in images:
            x, g = self.patches_mx(np.asarray(image, dtype=np.uint8))
            outs.append(x)
            grid.append(g)
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, 0)
        mx.eval(out)
        return np.array(out, copy=False), np.asarray(grid, np.int64)


# ---------------------------------------------------------------- the engine's interface

def load_image(value):
    """A PIL image, a file path, or a data:image/...;base64, URL -> an RGB PIL image."""
    from PIL import Image
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    source = io.BytesIO(base64.b64decode(value.split(",", 1)[1], validate=True)) if isinstance(value, str) and value.startswith("data:image/") else value
    with Image.open(source) as image:
        return image.convert("RGB")


class Preprocessor:
    """What protocol.Encoder and locate.Locate use: one tokenizer (own instance) and the image processor."""

    def __init__(self, directory, system, max_pixels=1638400):
        directory = Path(directory)
        self.system = system
        self.tokenizer = Tokenizer(directory)
        self.image_pad_id = self.tokenizer.token_id("<|image_pad|>")
        self._image_config = json.loads((directory / "processor_config.json").read_text())["image_processor"]
        self.max_pixels = max_pixels
        self._images = None

    def chat_context(self, state_text, images=0):
        return context_text(self.system, state_text, images)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def encode_offsets(self, text):
        return self.tokenizer.encode_offsets(text)

    def token_id(self, token):
        return self.tokenizer.token_id(token)

    def special_tokens(self):
        return list(self.tokenizer.all_special_tokens)

    def add_special_token(self, token):
        if token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens([token])

    def load_processor(self):
        if self._images is None:
            self._images = MLXImageProcessor(self._image_config, min_pixels=65536, max_pixels=self.max_pixels)
        return self._images

    def encode_images(self, text, images):
        """Context ids with each <|image_pad|> expanded, pixel_values float32 and image_grid_thw int64."""
        processor = self.load_processor()
        pixels, grid = processor(images)
        counts = iter((grid.prod(axis=1) // (processor.merge ** 2)).tolist())
        out = []
        for t in self.tokenizer.encode(text):   # special tokens split the text, so expanding after tokenizing is exact
            if t == self.image_pad_id:
                out.extend([t] * next(counts))
            else:
                out.append(t)
        return out, pixels, grid
