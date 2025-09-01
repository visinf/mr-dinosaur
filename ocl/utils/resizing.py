"""Utilities related to resizing of tensors."""
import math
from typing import Optional, Tuple, Union, Sequence
import torch
from torch import nn

class Resize(nn.Module):
    """Module resizing tensors."""

    MODES = {"nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"}

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        resize_mode: str = "bilinear",
        patch_mode: bool = False,
        channels_last: bool = False,
    ): 
        super().__init__()

        self.size = size

        if resize_mode not in Resize.MODES:
            raise ValueError(f"`mode` must be one of {Resize.MODES}")
        self.resize_mode = resize_mode
        self.patch_mode = patch_mode
        self.channels_last = channels_last
        self.expected_dims = 3 if patch_mode else 4

    def forward(
        self, input: torch.Tensor, size_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Resize tensor.

        Args:
            input: Tensor to resize. If `patch_mode=False`, assumed to be of shape (..., C, H, W).
                If `patch_mode=True`, assumed to be of shape (..., C, P), where P is the number of
                patches. Patches are assumed to be viewable as a perfect square image. If
                `channels_last=True`, channel dimension is assumed to be the last dimension instead.
            size_tensor: Tensor which size to resize to. If tensor has <=2 dimensions and the last
                dimension of this tensor has length 2, the two entries are taken as height and width.
                Otherwise, the size of the last two dimensions of this tensor are used as height
                and width.

        Returns: Tensor of shape (..., C, H, W), where height and width are either specified by
            `size` or `size_tensor`.
        """
        dims_to_flatten = input.ndim - self.expected_dims
        if dims_to_flatten > 0:
            flattened_dims = input.shape[: dims_to_flatten + 1]
            input = input.flatten(0, dims_to_flatten)
        elif dims_to_flatten < 0:
            raise ValueError(
                f"Tensor needs at least {self.expected_dims} dimensions, but only has {input.ndim}"
            )

        if self.patch_mode:
            if self.channels_last:
                input = input.transpose(-2, -1)
            n_channels, n_patches = input.shape[-2:]
            patch_size_float = math.sqrt(n_patches)
            patch_size = int(math.sqrt(n_patches))
            if patch_size_float != patch_size:
                raise ValueError(
                    f"The number of patches needs to be a perfect square, but is {n_patches}."
                )
            input = input.view(-1, n_channels, patch_size, patch_size)
        else:
            if self.channels_last:
                input = input.permute(0, 3, 1, 2)

        if self.size is None:
            if size_tensor is None:
                raise ValueError("`size` is `None` but no `size_tensor` was passed.")
            if size_tensor.ndim <= 2 and size_tensor.shape[-1] == 2:
                height, width = size_tensor.unbind(-1)
                height = torch.atleast_1d(height)[0].squeeze().detach().cpu()
                width = torch.atleast_1d(width)[0].squeeze().detach().cpu()
                size = (int(height), int(width))
            else:
                size = size_tensor.shape[-2:]
        else:
            size = self.size

        input = torch.nn.functional.interpolate(
            input,
            size=size,
            mode=self.resize_mode,
        )

        if dims_to_flatten > 0:
            input = input.unflatten(0, flattened_dims)
        return input

def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[Union[Sequence[int], torch.Size]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """
    Convert and resize patch tensors to image shapes, supporting non-square images and higher-dimensional tensors.
    
    This method requires the number of patches to be convertible to the specified grid size (height × width).

    Args:
        patches (torch.Tensor): Patches to be converted, with shape (..., C, P), where C is the number of channels and P is the number of patches.
        image_shape (Optional[Union[Sequence[int], torch.Size]]): 
            The shape of the target image. It can have any number of dimensions, 
            but the last two dimensions must represent (Height, Width). 
            Example shapes:
                - (Batch, Channels, Height, Width)
                - (Channels, Height, Width)
                - (Height, Width)
            If provided, `scale_factor` must be `None`.
        scale_factor (Optional[Union[float, Tuple[float, float]]]): 
            The scaling factor. Can be a single float or a tuple (height_scale, width_scale). 
            Mutually exclusive with `image_shape`.
        resize_mode (str): 
            The resizing method. Valid options include "nearest", "nearest-exact", "bilinear", "bicubic".

    Returns:
        torch.Tensor: 
            A tensor with the same leading dimensions as `patches` (..., C, S_h, S_w), 
            where S_h and S_w are the resized image height and width.
    """
    # Check the selection of image_shape and scale_factor
    has_image_shape = size is not None
    has_scale = scale_factor is not None
    if has_image_shape == has_scale:
        raise ValueError("You must specify either `image_shape` or `scale_factor`, but not both.")
    
    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    
    if has_image_shape:
        # Ensure image_shape is a Sequence of integers
        if not isinstance(size, (Sequence, torch.Size)):
            raise TypeError("`image_shape` must be a sequence of integers or torch.Size.")
        
        if len(size) < 2:
            raise ValueError("`image_shape` must have at least two dimensions for Height and Width.")
        
        # Extract target height and width from the last two dimensions of image_shape
        target_height, target_width = size[-2], size[-1]
        
        # Calculate grid dimensions grid_h and grid_w
        # Assume patches are arranged in row-major order and infer grid size based on target aspect ratio
        aspect_ratio = target_height / target_width
        grid_w = int(round(math.sqrt(n_patches / aspect_ratio)))
        grid_h = n_patches // grid_w
        
        if grid_h * grid_w != n_patches:
            raise ValueError(
                f"Cannot infer a valid grid size based on target size {size[-2:]} and number of patches {n_patches}."
            )
    else:

        original_grid_size = int(math.sqrt(n_patches))
        if original_grid_size ** 2 != n_patches:
            raise ValueError("When using `scale_factor`, the number of patches must be a perfect square.")
        
        grid_h = grid_w = original_grid_size

    # Reshape patches into grid form
    try:
        # Assume patches are arranged in row-major order and first flatten them into a grid
        reshaped_patches = patches.view(-1, n_channels, grid_h, grid_w)
        
        # Move tensor to CPU for deterministic interpolation
        reshaped_patches_cpu = reshaped_patches.cpu()
        
        # Perform interpolation on CPU
        image_cpu = torch.nn.functional.interpolate(
            reshaped_patches_cpu,
            size=size[-2:] if has_image_shape else None,
            scale_factor=scale_factor,
            mode=resize_mode,
        )
        
        # Move the result back to GPU
        image = image_cpu.to(patches.device)
    except Exception as e:
        raise RuntimeError(f"Error resizing image: {e}")

    # Reshape the image to include all original leading dimensions except for patches
    # The new shape will replace the patches dimension with the resized height and width
    # Example: (Batch, C, P) -> (Batch, C, H, W)
    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])
