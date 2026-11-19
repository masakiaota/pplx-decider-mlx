"""numpy reference of the image path in decider_mlx/preprocess.py (the same integer arithmetic on the CPU)."""
import numpy as np

from decider_mlx.preprocess import NORM, ImageProcessor, int16_weights


def _pass(src, axis, out_size):
    """One fixed-point pass along axis (0 = rows / vertical, 1 = columns / horizontal) of an (H, W, C) uint8 array."""
    index, w, precision = int16_weights(src.shape[axis], out_size)
    acc = np.full((out_size, src.shape[1], src.shape[2]) if axis == 0 else (src.shape[0], out_size, src.shape[2]),
                  1 << (precision - 1), np.int32)
    s = src.astype(np.int32)
    for j in range(w.shape[1]):
        wj = w[:, j]
        if not wj.any():
            continue
        if axis == 0:
            acc += s[index[:, j]] * wj[:, None, None]
        else:
            acc += s[:, index[:, j]] * wj[None, :, None]
    acc >>= precision
    np.clip(acc, 0, 255, out=acc)
    return acc.astype(np.uint8)


def resize_bicubic(rgb, height, width):
    """(H, W, 3) uint8 -> (height, width, 3) uint8, identical to torchvision v2 resize(BICUBIC, antialias=True) on CPU."""
    if rgb.shape[:2] == (height, width):
        return rgb
    out = rgb
    if out.shape[1] != width:
        out = _pass(out, 1, width)
    if out.shape[0] != height:
        out = _pass(out, 0, height)
    return out



class ReferenceImageProcessor(ImageProcessor):
    def patches(self, rgb):
        """(H, W, 3) uint8 -> pixel rows (gh*gw, 1536) float32 and (1, gh, gw)."""
        h, w = self.target(*rgb.shape[:2])
        img = resize_bicubic(np.ascontiguousarray(rgb), h, w)
        p, m, t = self.patch, self.merge, self.temporal
        gh, gw = h // p, w // p
        x = img.reshape(gh // m, m, p, gw // m, m, p, 3).transpose(0, 3, 1, 4, 6, 2, 5)  # gh/m gw/m m m c p p
        x = NORM[x.reshape(gh * gw, 3, 1, p * p)]
        x = np.broadcast_to(x, (gh * gw, 3, t, p * p)).reshape(gh * gw, 3 * t * p * p)
        return x, (1, gh, gw)

    def __call__(self, images):
        """images: PIL RGB images or (H, W, 3) uint8 arrays -> (pixel_values float32, image_grid_thw int64)."""
        rows, grid = [], []
        for image in images:
            x, g = self.patches(np.asarray(image, dtype=np.uint8))
            rows.append(x)
            grid.append(g)
        return (rows[0] if len(rows) == 1 else np.concatenate(rows, 0)), np.asarray(grid, np.int64)
