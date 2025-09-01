"""Data preprocessing functions."""
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import numpy
import torch
from torchvision import transforms
from torchvision.ops import masks_to_boxes

class DropEntries:
    """Drop entries from data dictionary."""

    def __init__(self, keys: List[str]):
        """Initialize DropEntries.

        Args:
            keys: Entries that should be dropped from the input dict.
        """
        self.keys = tuple(keys)

    def __call__(self, data: Dict[str, Any]):
        return {k: v for k, v in data.items() if k not in self.keys}


class CheckFormat:
    """Check format of data."""

    def __init__(self, shape: List[int], one_hot: bool = False, class_dim: int = 0):
        """Initialize CheckFormat.

        Args:
            shape: Shape of input tensor.
            one_hot: Check if input tensor is one hot.
            class_dim: Axis along which tensor should be one hot.
        """
        self.shape = tuple(shape)
        self.one_hot = one_hot
        self.class_dim = class_dim

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape != self.shape:
            raise ValueError(f"Expected shape to be {self.shape}, but is {data.shape}")

        if self.one_hot:
            if not torch.all(data.sum(self.class_dim) == 1):
                raise ValueError("Data is not one-hot")

        return data


class CompressMask:
    """Compress masks using a binary encoding format.

    This works for up to 64 objects.
    """

    def __call__(self, mask: numpy.ndarray) -> numpy.ndarray:
        non_empty = numpy.any(mask != 0, axis=(0, 2, 3))
        # Preserve first object beeing empty. This is often considered the
        # foreground mask and sometimes ignored.
        last_nonempty_index = len(non_empty) - non_empty[::-1].argmax()
        input_arr = mask[:, :last_nonempty_index]
        n_objects = input_arr.shape[1]
        dtype = numpy.uint8
        if n_objects > 8:
            dtype = numpy.uint16
        if n_objects > 16:
            dtype = numpy.uint32
        if n_objects > 32:
            dtype = numpy.uint64
        if n_objects > 64:
            raise RuntimeError("We do not support more than 64 objects at the moment.")

        object_flag = (1 << numpy.arange(n_objects, dtype=dtype))[None, :, None, None]
        output_arr = numpy.sum(input_arr.astype(dtype) * object_flag, axis=1).astype(dtype)
        return output_arr


class CompressedMaskToTensor:
    """Decompress a mask compressed with [CompressMask][ocl.preprocessing.CompressMask]."""

    def __call__(self, compressed_mask: numpy.ndarray) -> torch.Tensor:
        maximum_value = numpy.max(compressed_mask)
        n_objects = 0
        while maximum_value > 0:
            maximum_value //= 2
            n_objects += 1

        if n_objects == 0:
            # Cover edge case of no objects.
            n_objects = 1

        squeeze = False
        if len(compressed_mask.shape) == 2:
            compressed_mask = compressed_mask[None, ...]
            squeeze = True
        # Not really sure why we need to invert the order here, but it seems
        # to be necessary for the index to remain consistent between compression
        # and decompression.
        is_bit_active = (1 << numpy.arange(n_objects, dtype=compressed_mask.dtype))[
            None, :, None, None
        ]
        expanded_mask = (compressed_mask[:, None, :, :] & is_bit_active) > 0
        if squeeze:
            expanded_mask = numpy.squeeze(expanded_mask, axis=0)
        return torch.from_numpy(expanded_mask).to(torch.float32)


class MaskToTensor:
    """Convert a segmentation mask numpy array to a tensor."""

    def __init__(self, singleton_dim_last: bool = True):
        self.singleton_dim_last = singleton_dim_last

    def __call__(self, mask: numpy.ndarray) -> torch.Tensor:
        """Apply transformation.

        Args:
            mask: Mask tensor of shape (..., K, H, W, 1), i.e. one-hot encoded
                with K classes and any number of leading dimensions.

        Returns:
            Tensor of shape (..., K, H, W), containing binary entries.
        """

        mask_binary = mask > 0.0
        if self.singleton_dim_last:
            assert mask_binary.shape[-1] == 1
            return torch.from_numpy(mask_binary).squeeze(-1).to(torch.float32)
        else:
            return torch.from_numpy(mask_binary).to(torch.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DenseMaskToTensor:
    """Convert a dense segmentation mask numpy array to a tensor.

    Mask is assumed to be of shape (..., K, H, W, 1), i.e. densely encoded with K classes and any
    number of leading dimensions. Returned tensor is of shape (..., K, H, W).
    """

    def __call__(self, mask: numpy.ndarray) -> torch.Tensor: 
        assert mask.shape[-1] == 1
        tensor_map=torch.from_numpy(mask).squeeze(-1).to(torch.uint8)
        return tensor_map

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MultiMaskToTensor:
    """Discretize mask, where multiple objects are partially masked into an exclusive binary mask."""

    def __init__(self, axis: int = -4):
        self.axis = axis

    def __call__(self, mask: numpy.ndarray) -> torch.Tensor:
        int_mask = numpy.argmax(mask, axis=self.axis).squeeze(-1)
        out_mask = torch.nn.functional.one_hot(torch.from_numpy(int_mask), mask.shape[self.axis])
        # Ensure the object axis is again at the same location.
        # We operate on the shape prior to squeezing for axis to be consistent.
        last_index = len(out_mask.shape) - 1
        indices = list(range(len(out_mask.shape) + 1))
        indices.insert(self.axis, last_index)
        indices = indices[:-2]  # Remove last indices as they are squeezed or inserted.
        out_mask = out_mask.permute(*indices).to(torch.float32)
        return out_mask


class IntegerToOneHotMask:
    """Convert an integer mask to a one-hot mask.

    Integer masks are masks where the instance ID is written into the mask.
    This transform expands them to a one-hot encoding.
    """

    def __init__(
        self,
        ignore_typical_background: bool = True,
        output_axis: int = -4,
        max_instances: Optional[int] = None,
    ):
        """Initialize IntegerToOneHotMask.

        Args:
            ignore_typical_background: Ignore pixels where the mask is zero or 255.
                This often corresponds to the background or to the segmentation boundary.
            output_axis: Axis along which the output should be one hot.
            max_instances: The maximum number of instances.

        """
        self.ignore_typical_background = ignore_typical_background
        self.output_axis = output_axis
        self.max_instances = max_instances

    def __call__(self, array: numpy.ndarray) -> numpy.ndarray:
        max_value = array.max()
        if self.ignore_typical_background:
            if max_value == 255:
                # Replace 255 with zero, both are ignored.
                array[array == 255] = 0
                max_value = array.max()
            max_instances = self.max_instances if self.max_instances else max_value
            to_one_hot = numpy.concatenate(
                [
                    numpy.zeros((1, max_instances), dtype=numpy.uint8),
                    numpy.eye(max_instances, dtype=numpy.uint8),
                ],
                axis=0,
            )
        else:
            max_instances = self.max_instances if self.max_instances else max_value
            to_one_hot = numpy.eye(max_instances + 1, dtype=numpy.uint8)
        return numpy.moveaxis(to_one_hot[array], -1, self.output_axis)


class VOCInstanceMasksToDenseMasks:
    """Convert a segmentation mask with integer encoding into a one-hot segmentation mask.

    We use this transform as Pascal VOC segmentatation and object annotations seems to not
    be aligned.
    """

    def __init__(
        self,
        instance_mask_key: str = "segmentation-instance",
        class_mask_key: str = "segmentation-class",
        classes_key: str = "instance_category",
        ignore_mask_key: str = "ignore_mask",
        instance_axis: int = -4,
    ):
        self.instance_mask_key = instance_mask_key
        self.class_mask_key = class_mask_key
        self.classes_key = classes_key
        self.ignore_mask_key = ignore_mask_key
        self.instance_axis = instance_axis

    def __call__(self, data: Dict[str, Any]):
        data[self.ignore_mask_key] = (data[self.class_mask_key] == 255)[None]  # 1 x H x W x 1
        expanded_segmentation_mask = data[self.instance_mask_key] * numpy.expand_dims(
            data[self.class_mask_key], axis=self.instance_axis
        )
        assert expanded_segmentation_mask.max() != 255
        data[self.instance_mask_key] = expanded_segmentation_mask
        classes = []
        for instance_slice in numpy.rollaxis(expanded_segmentation_mask, self.instance_axis):
            unique_values = numpy.unique(instance_slice)
            assert len(unique_values) == 2  # Should contain 0 and class id.
            classes.append(unique_values[1])
        data[self.classes_key] = numpy.array(classes)

        return data


class AddImageSize:
    """Add height and width of image as data entry.

    Args:
        key: Key of image.
        target_key: Key under which to store size.
    """

    def __init__(self, key: str = "image", target_key: str = "image_size"):
        self.key = key
        self.target_key = target_key

    def __call__(self, data: Dict[str, Any]):
        height, width, _ = data[self.key].shape
        data[self.target_key] = numpy.array([height, width], dtype=numpy.int64)
        return data


class AddEmptyMasks:
    """Add empty masks to data if the data does not include them already."""

    def __init__(self, mask_keys: Union[str, Sequence[str]], take_size_from: str = "image"):
        """Initialize AddEmptyMasks.

        Args:
            mask_keys: One or several keys of empty masks to be added.
            take_size_from: Key of element whose height and width is used to create mask. Element is
                assumed to have shape of (H, W, C).
        """
        if isinstance(mask_keys, str):
            self.mask_keys = (mask_keys,)
        else:
            self.mask_keys = tuple(mask_keys)
        self.source_key = take_size_from

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        height, width, _ = data[self.source_key].shape
        for key in self.mask_keys:
            if key not in data:
                data[key] = numpy.zeros((1, height, width, 1), dtype=numpy.uint8)

        return data


class AddEmptyBboxes:
    """Add empty bounding boxes to data if the data does not include them already.

    Args:
        keys: One or several keys of empty boxes to be added.
        empty_value: Value of the empty box at all coordinates.
    """

    def __init__(self, keys: Union[str, Sequence[str]] = "instance_bbox", empty_value: float = -1.0):
        if isinstance(keys, str):
            self.keys = (keys,)
        else:
            self.keys = tuple(keys)
        self.empty_value = empty_value

    def __call__(self, data: Dict[str, Any]):
        for key in self.keys:
            if key not in data:
                data[key] = numpy.ones((1, 4), dtype=numpy.float32) * self.empty_value

        return data


class CanonicalizeBboxes:
    """Convert bounding boxes to canonical (x1, y1, x2, y2) format.

    Args:
        key: Key of bounding box, assumed to have shape K x 4.
        format: Format of bounding boxes. Either "xywh" or "yxyx".
    """

    def __init__(self, key: str = "instance_bbox", format: str = "xywh"):
        self.key = key

        self.format_xywh = False
        self.format_yxyx = False
        if format == "xywh":
            self.format_xywh = True
        elif format == "yxyx":
            self.format_yxyx = True
        else:
            raise ValueError(f"Unknown input format `{format}`")

    def __call__(self, data: Dict[str, Any]):
        if self.key not in data:
            return data

        bboxes = data[self.key]
        if self.format_xywh:
            x1, y1, w, h = numpy.split(bboxes, 4, axis=1)
            x2 = x1 + w
            y2 = y1 + h
        elif self.format_yxyx:
            y1, x1, y2, x2 = numpy.split(bboxes, 4, axis=1)

        data[self.key] = numpy.concatenate((x1, y1, x2, y2), axis=1)

        return data


class RescaleBboxes:
    """Rescale bounding boxes by size taken from data.

    Bounding boxes are assumed to have format (x1, y1, x2, y2). The rescaled box is
        (x1 * width, y1 * height, x2 * width, y2 * height).

    Args:
        key: Key of bounding box, assumed to have shape K x 4.
        take_size_from: Key of element to take the size for rescaling from, assumed to have shape
            H x W x C.
    """

    def __init__(self, key: str = "instance_bbox", take_size_from: str = "image"):
        self.key = key
        self.take_size_from = take_size_from

    def __call__(self, data: Dict[str, Any]):
        if self.key not in data:
            return data

        height, width, _ = data[self.take_size_from].shape
        scaling = numpy.array([[width, height, width, height]], dtype=numpy.float32)
        data[self.key] = data[self.key] * scaling

        return data


def expand_dense_mask(mask: numpy.ndarray) -> numpy.ndarray:
    """Convert dense segmentation mask to one where each class occupies one dimension.

    Args:
        mask: Densely encoded segmentation mask of shape 1 x H x W x 1.

    Returns: Densely encoded segmentation mask of shape K x H x W x 1, where K is the
        number of classes in the mask. Zero is taken to indicate an unoccupied pixel.
    """
    classes = numpy.unique(mask)[:, None, None, None]
    mask = (classes == mask) * classes

    # Strip empty class, but only if there is something else in the mask
    if classes[0].squeeze() == 0 and len(classes) != 1:
        mask = mask[1:]

    return mask


class AddBinaryMapFromInstanceMask:
    """Convert instance masks to binary maps.

    Converts all marked pixels to 1 and unmarked pixels to 0. Provides an option to ignore the
    background channel (assumed to be the first channel).
    """

    def __init__(
        self,
        instance_mask_key: str = "instance_mask",
        target_key: str = "binary_map",
        ignore_background: bool = False,
    ):
        self.instance_mask_key = instance_mask_key
        self.target_key = target_key
        self.ignore_background = ignore_background

    @staticmethod
    def convert(instance_mask: numpy.ndarray, ignore_background: bool) -> numpy.ndarray:

        if ignore_background:
            instance_mask = instance_mask[1:]  

        # Reduce instance mask to single dimension
        binary_map = (instance_mask > 0).max(axis=0, keepdims=True).astype(numpy.float32)
        return binary_map

    def __call__(self, data: Dict[str, Any]):
        if self.instance_mask_key not in data:
            return data

        data[self.target_key] = self.convert(data[self.instance_mask_key], self.ignore_background)

        return data

class PseudoLabelsToBinaryMap:
    """Convert instance masks to a binary map.

    Converts all non-zero pixels to 1 and zero pixels to 0.
    """

    def __init__(
        self,
        instance_mask_key: str = "label",
        target_key: str = "binary_map",
    ):
        self.instance_mask_key = instance_mask_key
        self.target_key = target_key
        
    @staticmethod
    def convert(instance_mask: numpy.ndarray) -> numpy.ndarray:
        binary_map = (instance_mask != 0).astype(numpy.uint8)
        
        # Add an extra dimension to match shape H x W x 1
        binary_map = numpy.expand_dims(binary_map, axis=-1)
        binary_map = numpy.expand_dims(binary_map, axis=0) 
        return binary_map

    def __call__(self, data: Dict[str, Any]):
        if self.instance_mask_key not in data:
            return data

        data[self.target_key] = self.convert(data[self.instance_mask_key])

        return data
    
class PseudoLabelsToBinaryInstanceMaps:
    """Convert instance masks to binary maps.

    Converts all marked pixels to 1 and unmarked pixels to 0. Provides an option to ignore the
    background channel (assumed to be the first channel).
    """

    def __init__(
        self,
        instance_mask_key: str = "label",
        target_key: str = "binary_instance_maps",
    ):
        self.instance_mask_key = instance_mask_key
        self.target_key = target_key

    @staticmethod
    def convert(instance_mask: numpy.ndarray) -> numpy.ndarray:
        instance_ids, counts = np.unique(instance_mask, return_counts=True)
        instance_ids = instance_ids[instance_ids != 0]
        # Get height and width of the input mask
        H, W = instance_mask.shape

        # Initialize binary instance maps
        binary_instance_maps = numpy.zeros((len(instance_ids), H, W, 1), dtype=numpy.uint8)

        # Fill the binary instance maps
        for i, instance_id in enumerate(instance_ids):
            binary_instance_maps[i, :, :, 0] = (instance_mask == instance_id).astype(numpy.uint8)
        return binary_instance_maps

    def __call__(self, data: Dict[str, Any]):
        if self.instance_mask_key not in data:
            return data

        data[self.target_key] = self.convert(data[self.instance_mask_key])

        return data
    
    
    
class AddSegmentationMaskFromInstanceMask:
    """Convert instance to segmentation masks by joining instances with the same category.

    Overlaps of instances of different classes are resolved by taking the class with the higher class
    id.
    """

    def __init__(
        self,
        instance_mask_key: str = "instance_mask",
        target_key: str = "segmentation_mask",
    ):
        self.instance_mask_key = instance_mask_key
        self.target_key = target_key

    @staticmethod
    def convert(instance_mask: numpy.ndarray) -> numpy.ndarray:
        """Convert instance to segmentation mask.

        Args:
            instance_mask: Densely encoded instance masks of shape I x H x W x 1, where I is the
                number of instances.
        """
        # Reduce instance mask to single dimension
        instance_mask = instance_mask.max(axis=0, keepdims=True)

        return expand_dense_mask(instance_mask)

    def __call__(self, data: Dict[str, Any]):
        if self.instance_mask_key not in data:
            return data

        data[self.target_key] = self.convert(data[self.instance_mask_key])

        return data


class RenameFields:
    def __init__(self, mapping: Dict):
        self.mapping = mapping

    def __call__(self, d: Dict):
        # Create shallow copy to avoid issues target key is already used.
        out = d.copy()
        for source, target in self.mapping.items():
            del out[source]
            out[target] = d[source]
        return out

    
class InstanceMasksToDenseMasks:
    """Convert binary instance masks to dense masks, i.e. where the mask value encodes the class id.

    Class ids are taken from a list containing a class id per instance.
    """

    def __init__(
        self,
        instance_mask_key: str = "instance_mask",
        category_key: str = "instance_category",
    ):
        self.instance_mask_key = instance_mask_key
        self.category_key = category_key

    @staticmethod
    def convert(instance_mask: numpy.ndarray, categories: numpy.ndarray) -> numpy.ndarray:
        if numpy.min(categories) <= 0:
            raise ValueError("Detected category smaller equal than 0 in instance masks.")
        if numpy.max(categories) > 255:
            raise ValueError(
                "Detected category greater than 255 in instance masks. This does not fit in uint8."
            )

        categories = categories[:, None, None, None]
        return (instance_mask * categories).astype(numpy.uint8)

    def __call__(self, data: Dict[str, Any]):
        if self.instance_mask_key not in data:
            return data

        data[self.instance_mask_key] = self.convert(
            data[self.instance_mask_key], data[self.category_key]
        )
        return data


class MergeCocoThingsAndStuff:
    """Merge COCO things and stuff segmentation masks.

    Args:
        things_key: Key to things instance mask. Mask is assumed to be densely encoded, i.e.
            the mask value encodes the class id, of shape I x H x W x 1, where I is the number of
            things instances.
        stuff_key: Key to stuff segmentation mask. Mask is assumed to be densely encoded, i.e.
            the mask value encodes the class id, of shape K x H x W x 1, where K is the number stuff
            classes.
        output_key: Key under which the merged mask is stored. Returns mask of shape L x H x W x 1,
            where K <= L <= K + I.
        include_crowd: Whether to include pixels marked as crowd with their class, or with class
            zero.
    """

    def __init__(
        self,
        output_key: str,
        things_key: str = "instance_mask",
        stuff_key: str = "stuff_mask",
        include_crowd: bool = False,
    ):
        self.things_key = things_key
        self.stuff_key = stuff_key
        self.output_key = output_key
        self.include_crowd = include_crowd

    def __call__(self, data: Dict[str, Any]):
        if self.things_key in data:
            things_instance_mask = data[self.things_key]
            things_mask = things_instance_mask.max(axis=0, keepdims=True)
        else:
            things_mask = None

        stuff_mask = data[self.stuff_key]
        merged_mask = stuff_mask.max(axis=0, keepdims=True)

        # In stuff annotations, thing pixels are encoded as class 183.
        use_thing_mask = merged_mask == 183

        if things_mask is not None:
            if self.include_crowd:
                # In the stuff annotations, things marked with the "crowd" label are NOT encoded as
                # class 183, but as class 0. We can take the value of the things mask for those
                # pixels.
                use_thing_mask |= merged_mask == 0
            merged_mask[use_thing_mask] = things_mask[use_thing_mask]
        else:
            # No pixel should have value 183 if the things_mask does not exist, but convert it to
            # zero anyways just to be sure.
            merged_mask[use_thing_mask] = 0

        data[self.output_key] = expand_dense_mask(merged_mask)

        return data


class FlowToTensor:
    """Convert an optical flow numpy array to a tensor.

    Flow is assumed to be of shape (..., H, W, 2), returned tensor is of shape (..., 2, H, W) to
    match with VideoTensor format.
    """

    def __call__(self, flow: numpy.ndarray) -> torch.Tensor:
        flow_torch = torch.from_numpy(flow.astype(float)).to(torch.float32)
        return torch.moveaxis(flow_torch, -1, -3)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ConvertCocoStuff164kMasks:
    """Convert COCO-Stuff-164k PNG segmentation masks to our format.

    Args:
        output_key: Key under which the output mask is stored. Returns uint8 mask of shape
            K x H x W x 1, where K is the number of classes in the image. Mask is densely encoded,
            i.e. the mask values encode the class id.
        stuffthings_key: Key to COCO-Stuff-164k PNG mask. Mask has shape H x W x 3.
        ignore_key: Key under which the ignore mask is stored. Returns bool mask of shape
            1 x H x W x 1. Ignores pixels where PNG mask has value 255 (crowd).
        drop_stuff: If true, remove all stuff classes (id >= 92), keeping only thing classes.
    """

    def __init__(
        self,
        output_key: str,
        stuffthings_key: str = "stuffthings_mask",
        ignore_key: str = "ignore_mask",
        drop_stuff: bool = False,
    ):
        self.stuffthings_key = stuffthings_key
        self.ignore_key = ignore_key
        self.output_key = output_key
        self.drop_stuff = drop_stuff

    def __call__(self, data: Dict[str, Any]):
        mask = data[self.stuffthings_key]  # H x W x 3, mask is encoded as an image
        assert mask.shape[-1] == 3
        mask = mask[:, :, :1]  # Take first channel, all channels are the same

        ignore_mask = mask == 255

        # In PNG annotations, classes occupy indices 0-181, shift by 1
        mask = mask + 1
        mask[ignore_mask] = 0

        if self.drop_stuff:
            mask[mask >= 92] = 0

        data[self.ignore_key] = ignore_mask[None]  # 1 x H x W x 1
        data[self.output_key] = expand_dense_mask(mask[None])  # K x H x W x 1

        return data


class VideoToTensor:
    """Convert a video numpy array of shape (T, H, W, C) to a torch tensor of shape (T, C, H, W)."""

    def __call__(self, video):
        """Convert a numpy array of a video into a torch tensor.

        Assumes input is a numpy array of shape T x H x W x C (or T x H x W for monochrome videos)
        and convert it into torch tensor of shape T x C x H x W in order to allow application of
        Conv3D operations.
        """
        if isinstance(video, numpy.ndarray):
            # Monochrome video such as mask
            if video.ndim == 3:
                video = video[..., None]

            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).contiguous()
            # backward compatibility
            if isinstance(video, torch.ByteTensor):
                return video.to(dtype=torch.get_default_dtype()).div(255)
            else:
                return video
        else:
            # Should be torch tensor.
            if video.ndim == 3:
                video = video[..., None]

            video = video.permute(0, 3, 1, 2).contiguous()
            # backward compatibility
            if isinstance(video, torch.ByteTensor):
                return video.to(dtype=torch.get_default_dtype()).div(255)
            else:
                return video


class ToSingleFrameVideo:
    """Convert image in tensor format to video format by adding frame dimension with single element.

    Converts C x H x W tensors into tensors of shape 1 x C x H x W.
    """

    def __call__(self, image):
        return image.unsqueeze(0)


class NormalizeVideo:
    """Normalize a video tensor of shape (T, C, H, W)."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[None, :, None, None]
        self.std = torch.tensor(std)[None, :, None, None]

    def __call__(self, video):
        return (video - self.mean) / self.std


class Denormalize(torch.nn.Module):
    """Denormalize a tensor of shape (..., C, H, W) with any number of leading dimensions."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean)[:, None, None])
        self.register_buffer("std", torch.tensor(std)[:, None, None])

    def __call__(self, tensor):
        return tensor * self.std + self.mean


class ResizeNearestExact:
    """Resize a tensor using mode nearest-exact.

    This mode is not available in torchvision.transforms.Resize as of v0.12. This class was adapted
    from torchvision.transforms.functional_tensor.resize.
    """

    def __init__(self, size: Union[int, List[int]], max_size: Optional[int] = None):
        self.size = size
        self.max_size = max_size

    @staticmethod
    def _cast_squeeze_in(
        img: torch.Tensor, req_dtypes: List[torch.dtype]
    ) -> Tuple[torch.Tensor, bool, bool, torch.dtype]:
        need_squeeze = False
        # make image NCHW
        if img.ndim < 4:
            img = img.unsqueeze(dim=0)
            need_squeeze = True

        out_dtype = img.dtype
        need_cast = False
        if out_dtype not in req_dtypes:
            need_cast = True
            req_dtype = req_dtypes[0]
            img = img.to(req_dtype)
        return img, need_cast, need_squeeze, out_dtype

    @staticmethod
    def _cast_squeeze_out(
        img: torch.Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype
    ) -> torch.Tensor:
        if need_squeeze:
            img = img.squeeze(dim=0)

        if need_cast:
            if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                # it is better to round before cast
                img = torch.round(img)
            img = img.to(out_dtype)

        return img

    @staticmethod
    def resize(img: torch.Tensor, size: Union[int, List[int]], max_size: Optional[int] = None):
        h, w = img.shape[-2:]
        if isinstance(size, int) or len(size) == 1:  # specified size only for the smallest edge
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

            if (w, h) == (new_w, new_h):
                return img
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]

        img, need_cast, need_squeeze, out_dtype = ResizeNearestExact._cast_squeeze_in(
            img, (torch.float32, torch.float64)
        )

        img = torch.nn.functional.interpolate(img, size=[new_h, new_w], mode="nearest-exact")

        img = ResizeNearestExact._cast_squeeze_out(
            img, need_cast=need_cast, need_squeeze=need_squeeze, out_dtype=out_dtype
        )
        return img

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return ResizeNearestExact.resize(img, self.size, self.max_size)

class OrigCenterCrop:
    """Returns center crop at original image resolution."""

    def __call__(self, image):
        height, width = image.shape[-2:]
        return transforms.functional.center_crop(image, min(height, width))


class JointRandomResizedCropwithParameters(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=transforms.functional.InterpolationMode.BILINEAR,
    ):
        super().__init__(size, scale, ratio, interpolation)
        self.mask_to_tensor = DenseMaskToTensor()
        self.mask_resize = ResizeNearestExact((size, size))

    def forward(self, img: torch.Tensor, masks: Optional[Dict] = None) -> torch.Tensor:
        """Returns parameters of the resize in addition to the crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        params = self.get_params(img, self.scale, self.ratio)
        img = transforms.functional.resized_crop(img, *params, self.size, self.interpolation)

        for mask_key, mask in masks.items():
            if not isinstance(mask, torch.Tensor):
                mask = self.mask_to_tensor(mask)
            mask = transforms.functional.crop(mask, *params)
            mask = self.mask_resize(mask)
            masks[mask_key] = mask
        return img, masks, params


class MultiCrop(object):
    def __init__(
        self,
        size: int = 224,
        input_key: str = "image",
        teacher_key: str = "teacher",
        student_key: str = "student",
        global_scale: Tuple[float, float] = (0.8, 1.0),
        local_scale: Tuple[float, float] = (0.7, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        mask_keys: Optional[Tuple[str]] = None,
    ):
        self.ratio = ratio
        self.teacher_key = teacher_key
        self.student_key = student_key
        self.global_crop = JointRandomResizedCropwithParameters(size, global_scale, ratio)
        self.local_crop = JointRandomResizedCropwithParameters(size, local_scale, ratio)
        self.input_key = input_key
        self.mask_keys = tuple(mask_keys) if mask_keys is not None else tuple()

    def __call__(self, data):
        if self.input_key not in data:
            raise ValueError(f"Wrong input key {self.input_key}")
        img = transforms.functional.to_tensor(data[self.input_key])
        masks = {mask_key: data[mask_key] for mask_key in self.mask_keys}
        teacher_view, global_masks, params = self.global_crop(img, masks)
        data[self.teacher_key] = teacher_view
        for k, mask in global_masks.items():
            data[f"{self.teacher_key}_{k}"] = mask

        student_view, local_masks, params = self.local_crop(teacher_view, global_masks)
        data[self.student_key] = student_view
        for k, mask in local_masks.items():
            data[f"{self.student_key}_{k}"] = mask
        data["params"] = torch.Tensor(numpy.array(params))
        return data


class TokenizeText:
    def __init__(self, context_length: int = 77, truncate: bool = False):
        self.context_length = context_length
        self.truncate = truncate

    def __call__(self, texts: Union[List[str], str]):
        # TODO: Understand if there is any performance impact in importing here.
        try:
            import clip
        except ImportError:
            raise Exception("Using clip models requires installation with extra `clip`.")
        tokenized = clip.tokenize(texts, self.context_length, self.truncate)
        if isinstance(texts, str):
            # If the input is a single string, then the tokenization adds an additional dimension
            # which we don't need.
            tokenized = tokenized[0]
        return tokenized


class RandomSample:
    """Draw a random sample from the first axis of a list or array."""

    def __call__(self, tokens):
        return random.choice(tokens)


class IsElementOfList:
    def __init__(self, list: List[str]):
        self.list = set(list)

    def __call__(self, key):
        return key in self.list


class SampleFramesUsingIndices:
    """Sample frames form a tensor dependent on indices provided in the instance."""

    def __init__(self, frame_fields: List[str], index_field: str):
        self.frame_fields = frame_fields
        self.index_field = index_field

    def __call__(self, inputs: dict):
        indices = inputs[self.index_field]
        for frame_field in self.frame_fields:
            inputs[frame_field] = inputs[frame_field][indices]
        return inputs


class MaskInstances:
    """Filter instances by masking non matching with NaN."""

    def __init__(
        self,
        fields: List[str],
        keys_to_keep: List[str],
        mask_video: bool = False,
    ):
        self.fields = fields
        self.keys_to_keep = set(keys_to_keep)
        self.mask_video = mask_video
        if self.mask_video:
            self.video_key_to_frame_mapping = defaultdict(set)
            for key in self.keys_to_keep:
                video_key, frame = key.split("_")
                self.video_key_to_frame_mapping[video_key].add(int(frame))

    def mask_instance(self, instance):
        key = instance["__key__"]

        if key not in self.keys_to_keep:
            for field in self.fields:
                data = instance[field]
                if isinstance(data, numpy.ndarray):
                    instance[field] = numpy.full_like(data, numpy.NaN)
                elif isinstance(data, torch.Tensor):
                    instance[field] = torch.full_like(data, numpy.NaN)
                else:
                    raise RuntimeError(f"Field {field} is of unexpected type {type(data)}.")
        return instance

    def mask_instance_video(self, instance):
        key = instance["__key__"]
        output = instance.copy()
        for field in self.fields:
            data = instance[field]
            if isinstance(data, numpy.ndarray):
                output[field] = numpy.full_like(data, numpy.NaN)
            elif isinstance(data, torch.Tensor):
                output[field] = torch.full_like(data, numpy.NaN)
            else:
                raise RuntimeError(f"Field {field} is of unexpected type {type(data)}.")

        # We need to do some special handling here due to the strided decoding.
        # This is not really nice, but fixing it nicely would require significantly
        # more work for which we do not have the time at the moment.
        if "decoded_indices" in instance.keys():
            # Input comes from strided decoding, we thus need to adapt
            # key and frames.
            key, _ = key.split("_")  # Get video key.
            key = str(int(key))
            if key in self.video_key_to_frame_mapping.keys():
                frames_to_keep = self.video_key_to_frame_mapping[key]
                decoded_indices = instance["decoded_indices"]
                frames_to_keep = [index for index in decoded_indices if index in frames_to_keep]
                for field in self.fields:
                    data = instance[field]
                    output[field][frames_to_keep] = data[frames_to_keep]
        else:
            if key in self.video_key_to_frame_mapping.keys():
                frames_to_keep = self.video_key_to_frame_mapping[key]
                for field in self.fields:
                    data = instance[field]
                    output[field][frames_to_keep] = data[frames_to_keep]
        return output

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.mask_video:
            return self.mask_instance_video(input_dict)
        return self.mask_instance(input_dict)
