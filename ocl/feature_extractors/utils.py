"""Utility functions used for feature extractors."""
import abc
import math
from typing import Dict, List, Tuple, Union, Optional

import torch
from torch import nn

import ocl.typing


class FeatureExtractor(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for Feature Extractors.

    We expect that the forward method returns a flattened representation of the features, to make
    outputs consistent and not dependent on equal spacing or the dimensionality of the spatial
    information.
    """

    @abc.abstractmethod
    def forward(self, inputs: ocl.typing.ImageOrVideoFeatures) -> ocl.typing.FeatureExtractorOutput:
        pass


class ImageFeatureExtractor(FeatureExtractor):
    """Base class that allows operation of image based feature extractors on videos.

    This is implemented by reshaping the frame dimesion into the batch dimension and
    inversing the process after extraction of the features.

    Subclasses override the `forward_images` method.
    """

    @abc.abstractmethod
    def forward_images(
        self, images: ocl.typing.ImageData
    ) -> Union[
        Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions],
        Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions, Dict],
    ]:
        """Apply feature extractor to image tensor.

        Returns:
            - `torch.Tensor` of extracted features
            - `torch.Tensor` of spatial positions of extracted features
            - Optional dict with additional auxilliary features or information
                from the feature extractor.
        """
    def forward(self, video: ocl.typing.ImageOrVideoData) -> ocl.typing.FeatureExtractorOutput:
        """Apply subclass image feature extractor to potential video data.

        Args:
            video: 5D tensor for video data or 4D tensor for image data.

        Returns:
            ocl.typing.FeatureExtractorOutput: The extracted features with positiional information
                and potential auxilliary features.
        """
        ndim = video.dim()
        assert ndim == 4 or ndim == 5

        if ndim == 5:
            # Handling video data.
            bs, frames, channels, height, width = video.shape
            images = video.view(bs * frames, channels, height, width).contiguous()
        else:
            images = video

        result = self.forward_images(images)

        if isinstance(result, (Tuple, List)):
            if len(result) == 2:
                features, positions = result
                aux_features = None
            elif len(result) == 3:
                features, positions, aux_features = result
            else:
                raise RuntimeError("Expected either 2 or 3 element tuple from `forward_images`.")
        else:
            # Assume output is simply a tensor without positional information.
            return ocl.typing.FeatureExtractorOutput(result, None, None)

        if ndim == 5:
            features = features.unflatten(0, (bs, frames))
            if aux_features is not None:
                aux_features = {k: f.unflatten(0, (bs, frames)) for k, f in aux_features.items()}

        return ocl.typing.FeatureExtractorOutput(features, positions, aux_features)


def cnn_compute_positions_and_flatten(
    features: ocl.typing.CNNImageFeatures,
) -> Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions]:
    """Flatten CNN features to remove spatial dims and return them with correspoding positions."""
    spatial_dims = features.shape[2:]
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # reorder into format (batch_size, flattened_spatial_dims, feature_dim).
    flattened = torch.permute(features.view(features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return flattened, positions


def transformer_compute_positions(
    features: ocl.typing.TransformerImageFeatures,
    size: Optional[Tuple[int, int]] = None
) -> ocl.typing.Positions:
    """
    Compute the positions of Transformer features, suitable for non-square images.

    Args:
        features (ocl.typing.TransformerImageFeatures): Transformer image features with shape (batch_size, n_tokens, feature_dim).
        size (Optional[Tuple[int, int]]): The size of the image as (height, width). If not provided, assumes a square image and infers size from n_tokens.

    Returns:
        ocl.typing.Positions: Position tensor with shape (n_tokens, 2).
    """
    n_tokens = features.shape[1]
    if size is not None:
        height, width = size
        expected_tokens = height * width
        assert n_tokens == expected_tokens, (
            f"Number of tokens {n_tokens} does not match expected height * width {height} * {width} = {expected_tokens}."
        )
    else:
        image_size = math.sqrt(n_tokens)
        image_size_int = int(image_size)
        assert (
            image_size_int == image_size
        ), "Position computation for Transformers requires a square image when size is not provided."
        height = width = image_size_int

    spatial_dims = (height, width)
    positions = torch.cartesian_prod(
        *[
            torch.linspace(0.0, 1.0, steps=dim, device=features.device) 
            for dim in spatial_dims
        ]
    )
    
    
    return positions
