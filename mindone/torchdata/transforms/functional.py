import numbers
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None

from . import _functional_pil as F_pil, _functional_tensor as F_t


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``nearest-exact``, ``bilinear``, ``bicubic``, ``box``, ``hamming``,
    and ``lanczos``.
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


# TODO: Once torchscript supports Enums with staticmethod
# this can be put into InterpolationMode as staticmethod
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.NEAREST_EXACT: 0,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}

_is_pil_image = F_pil._is_pil_image


def get_dimensions(img) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].

    Args:
        img (PIL Image): The image to be checked.

    Returns:
        List[int]: The image dimensions.
    """
    return F_pil.get_dimensions(img)


def get_image_size(img) -> List[int]:
    """Returns the size of an image as [width, height].

    Args:
        img (PIL Image): The image to be checked.

    Returns:
        List[int]: The image size.
    """
    return F_pil.get_image_size(img)


def get_image_num_channels(img) -> int:
    """Returns the number of channels of an image.

    Args:
        img (PIL Image): The image to be checked.

    Returns:
        int: The number of channels.
    """
    return F_pil.get_image_num_channels(img)


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def to_tensor(pic) -> np.ndarray:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        np.ndarray: Converted image.
    """
    if not (F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = pic.transpose((2, 0, 1))
        # backward compatibility
        if img.dtype == np.uint8:
            return img.astype(dtype=np.float32) / 255
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return nppic.astype(dtype=np.float32)

    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)

    if pic.mode == "1":
        img = 255 * img
    img = img.reshape((pic.size[1], pic.size[0], F_pil.get_image_num_channels(pic)))
    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    if img.dtype == np.uint8:
        return img.astype(dtype=np.float32) / 255
    else:
        return img


def normalize(tensor: np.ndarray, mean: List[float], std: List[float], inplace: bool = False) -> np.ndarray:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")

    return F_t.normalize(tensor, mean=mean, std=std, inplace=inplace)


def _compute_resized_output_size(
    image_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
) -> List[int]:
    if len(size) == 1:  # specified size only for the smallest edge
        h, w = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]


def resize(
    img,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[Union[str, bool]] = "warn",
):
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image: if the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``, then
            the image is resized again so that the longer edge is equal to
            ``max_size``. As a result, ``size`` might be overruled, i.e. the
            smaller edge may be shorter than ``size``. This is only supported
            if ``size`` is an int (or a sequence of length 1 in torchscript
            mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if isinstance(size, (list, tuple)):
        if len(size) not in [1, 2]:
            raise ValueError(
                f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
            )
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )

    _, image_height, image_width = get_dimensions(img)
    if isinstance(size, int):
        size = [size]
    output_size = _compute_resized_output_size((image_height, image_width), size, max_size)

    if (image_height, image_width) == output_size:
        return img

    antialias = _check_antialias(img, antialias, interpolation)

    if antialias is False:
        warnings.warn("Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.")
    pil_interpolation = pil_modes_mapping[interpolation]
    return F_pil.resize(img, size=output_size, interpolation=pil_interpolation)


def pad(img, padding: List[int], fill: Union[int, float] = 0, padding_mode: str = "constant"):
    r"""Pad the given image on all sides with the given "pad" value.

    Args:
        img (PIL Image): Image to be padded.
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image: Padded image.
    """
    return F_pil.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def crop(img, top: int, left: int, height: int, width: int):
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    return F_pil.crop(img, top, left, height, width)


def center_crop(img, output_size: List[int]):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        _, image_height, image_width = get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def resized_crop(
    img,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
):
    """Crop the given image and resize it to desired size.

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
    Returns:
        PIL Image: Cropped image.
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation, antialias=antialias)
    return img


def hflip(img):
    """Horizontally flip the given image.

    Args:
        img (PIL Image): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image:  Horizontally flipped image.
    """
    return F_pil.hflip(img)


def vflip(img):
    """Vertically flip the given image.

    Args:
        img (PIL Image): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    return F_pil.vflip(img)


# TODO in v0.17: remove this helper and change default of antialias to True everywhere
def _check_antialias(
    img: np.ndarray, antialias: Optional[Union[str, bool]], interpolation: InterpolationMode
) -> Optional[bool]:
    if isinstance(antialias, str):  # it should be "warn", but we don't bother checking against that
        if isinstance(img, np.ndarray) and (
            interpolation == InterpolationMode.BILINEAR or interpolation == InterpolationMode.BICUBIC
        ):
            warnings.warn(
                "The default value of the antialias parameter of all the resizing transforms "
                "(Resize(), RandomResizedCrop(), etc.) "
                "will change from None to True in v0.17, "
                "in order to be consistent across the PIL and Tensor backends. "
                "To suppress this warning, directly pass "
                "antialias=True (recommended, future default), antialias=None (current default, "
                "which means False for Tensors and True for PIL), "
                "or antialias=False (only works on Tensors - PIL will still use antialiasing). "
                "This also applies if you are using the inference transforms from the models weights: "
                "update the call to weights.transforms(antialias=True)."
            )
        antialias = None

    return antialias
