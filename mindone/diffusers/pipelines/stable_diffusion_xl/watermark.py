import numpy as np

import mindspore
from mindspore import ops

from ...utils import is_invisible_watermark_available

if is_invisible_watermark_available():
    from imwatermark import WatermarkEncoder


# Copied from https://github.com/Stability-AI/generative-models/blob/613af104c6b85184091d42d374fef420eddb356d/scripts/demo/streamlit_helpers.py#L66
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


class StableDiffusionXLWatermarker:
    def __init__(self):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()

        self.encoder.set_watermark("bits", self.watermark)

    def apply_watermark(self, images: mindspore.Tensor):
        # can't encode images that are smaller than 256
        dtype = images.dtype
        if images.shape[-1] < 256:
            return images

        images = (255 * (images / 2 + 0.5)).permute(0, 2, 3, 1).numpy()

        images = [self.encoder.encode(image, "dwtDct") for image in images]

        images = mindspore.Tensor(np.array(images), dtype=dtype).permute(0, 3, 1, 2)

        images = ops.clamp(2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        return images
