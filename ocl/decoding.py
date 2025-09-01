"""Implementation of different types of decoders."""
import dataclasses
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

from ocl.neural_networks.convenience import get_activation_fn
from ocl.utils.resizing import resize_patches_to_image


@dataclasses.dataclass
class SimpleReconstructionOutput:
    reconstruction: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class ReconstructionOutput:
    reconstruction: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821
    object_reconstructions: TensorType[
        "batch_size", "n_objects", "channels", "height", "width"  # noqa: F821
    ]
    masks: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class ReconstructionAmodalOutput:
    reconstruction: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821
    object_reconstructions: TensorType[
        "batch_size", "n_objects", "channels", "height", "width"  # noqa: F821
    ]
    masks: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    masks_vis: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    masks_eval: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class PatchReconstructionOutput:
    reconstruction: TensorType["batch_size", "n_patches", "n_patch_features"]  # noqa: F821
    masks: TensorType["batch_size", "n_objects", "n_patches"]  # noqa: F821
    masks_as_image: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    target: Optional[TensorType["batch_size", "n_patches", "n_patch_features"]] = None  # noqa: F821
    slot_features: Optional[TensorType["batch_size", "n_objects", "n_features"]]=None  # noqa: F821


@dataclasses.dataclass
class MRPatchReconstructionOutput:
    reconstruction: TensorType["batch_size", "n_patches", "n_patch_features"]  # noqa: F821
    masks: TensorType["batch_size", "n_objects", "n_patches"]  # noqa: F821
    previous_masks: TensorType["batch_size", "n_objects", "n_patches"]  # noqa: F821
    masks_as_image: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    target: Optional[TensorType["batch_size", "n_patches", "n_patch_features"]] = None  # noqa: F821
    masks_changed: Optional[TensorType["batch_size", "n_objects", "n_patches"]]=None  # noqa: F821
    slot_features: Optional[TensorType["batch_size", "n_objects", "n_features"]]=None  # noqa: F821



@dataclasses.dataclass
class DepthReconstructionOutput(ReconstructionOutput):
    masks_amodal: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    depth_map: Optional[TensorType["batch_size", "height", "width"]] = None  # noqa: F821
    object_depth_map: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    densities: Optional[
        TensorType["batch_size", "n_objects", "n_depth", "height", "width"]  # noqa: F821
    ] = None
    colors: Optional[
        TensorType["batch_size", "n_objects", "n_depth", "channels", "height", "width"]  # noqa: F821
    ] = None


@dataclasses.dataclass
class OpticalFlowPredictionTaskOutput:
    predicted_flow: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821
    object_flows: TensorType["batch_size", "n_objects", "channels", "height", "width"]  # noqa: F821
    masks: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class BBoxOutput:
    bboxes: TensorType["batch_size", "n_objects", "box_dim"]  # noqa: F821
    classes: TensorType["batch_size", "n_objects", "num_classes"]  # noqa: F821
    ori_res_bboxes: TensorType["batch_size", "n_objects", "box_dim"]  # noqa: F821
    inference_obj_idxes: TensorType["batch_size", "n_objects"]  # noqa: F821


def build_grid_of_positions(resolution):
    """Build grid of positions which can be used to create positions embeddings."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    return grid


def get_slotattention_decoder_backbone(object_dim: int, output_dim: int = 4):
    """Get CNN decoder backbone form the original slot attention paper."""
    return nn.Sequential(
        nn.ConvTranspose2d(object_dim, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, output_dim, 3, stride=1, padding=1, output_padding=0),
    )

class SlotAttentionDecoder(nn.Module):
    """Decoder used in the original slot attention paper."""

    def __init__(
        self,
        decoder: nn.Module,
        final_activation: Union[str, Callable] = "identity",
        positional_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.final_activation = get_activation_fn(final_activation)
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.register_buffer("grid", build_grid_of_positions(self.initial_conv_size))

    def forward(self, object_features: torch.Tensor):
        assert object_features.dim() >= 3  # Image or video data.
        initial_shape = object_features.shape[:-1]
        object_features = object_features.flatten(0, -2)

        object_features = (
            object_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.initial_conv_size)
        )
        if self.positional_embedding:
            object_features = self.positional_embedding(object_features, self.grid.unsqueeze(0))

        # Apply deconvolution and restore object dimension.
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        # The decoder is assumed to output tensors in CNN order, i.e. * x C x H x W.
        rgb, alpha = output.split([3, 1], dim=-3)
        rgb = self.final_activation(rgb)
        alpha = alpha.softmax(dim=-4)

        return ReconstructionOutput(
            # Combine rgb weighted according to alpha channel.
            reconstruction=(rgb * alpha).sum(-4),
            object_reconstructions=rgb,
            masks=alpha.squeeze(-3),
        )


class PatchDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes inputs of shape (B * K), P, N, and produce
            outputs of shape (B * K), P, M, where K is the number of objects, N is the number of
            input dimensions and M the number of output dimensions.
        decoder_input_dim: Input dimension to decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        decoder_input_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode

        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = object_dim
        self.slot_dim = decoder_input_dim

        self.decoder = decoder(decoder_input_dim, output_dim + 1)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_input_dim) * 0.02)

    def forward(
        self,
        object_features: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ):
        assert object_features.dim() >= 3  # Image or video data.
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),
                    scale_factor=self.upsample_target,
                    resize_mode=self.resize_mode,
                )
                .flatten(-2, -1)
                .transpose(-2, -1)
            )

        initial_shape = object_features.shape[:-1]
        # original_object_features=object_features
        
        object_features = object_features.flatten(0, -2)


        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        object_features = object_features.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = object_features + self.pos_embed

        output = self.decoder(object_features)

        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)

        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)

        masks = alpha.squeeze(-1)

        if image is not None:
            masks_as_image = resize_patches_to_image(
                masks, size=image.shape, resize_mode="bilinear"
            )

        else:
            masks_as_image = None
        return PatchReconstructionOutput(
            reconstruction=reconstruction,
            masks=masks,
            masks_as_image=masks_as_image,
            target=target if target is not None else None,
            slot_features=object_features,
        )


class AutoregressivePatchDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches autoregressively.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes autoregressive targets of shape B, P, M,
            conditioning of shape B, K, N, masks of shape P, P, and produces outputs of shape
            B, P, M, where K is the number of objects, N is the number of input dimensions and M the
            number of output dimensions.
        decoder_cond_dim: Dimension of conditioning input of decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        decoder_dim: Optional[int] = None,
        decoder_cond_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
        use_decoder_masks: bool = False,
        use_bos_token: bool = True,
        use_input_transform: bool = False,
        use_input_norm: bool = False,
        use_output_transform: bool = False,
        use_positional_embedding: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode
        self.use_decoder_masks = use_decoder_masks

        if decoder_dim is None:
            decoder_dim = output_dim

        self.decoder = decoder(decoder_dim, decoder_dim)
        if use_bos_token:
            self.bos_token = nn.Parameter(torch.randn(1, 1, output_dim) * output_dim**-0.5)
        else:
            self.bos_token = None
        if decoder_cond_dim is not None:
            self.cond_transform = nn.Sequential(
                nn.Linear(object_dim, decoder_cond_dim, bias=False),
                nn.LayerNorm(decoder_cond_dim, eps=1e-5),
            )
            nn.init.xavier_uniform_(self.cond_transform[0].weight)
        else:
            decoder_cond_dim = object_dim
            self.cond_transform = nn.LayerNorm(decoder_cond_dim, eps=1e-5)

        if use_input_transform:
            self.inp_transform = nn.Sequential(
                nn.Linear(output_dim, decoder_dim, bias=False),
                nn.LayerNorm(decoder_dim, eps=1e-5),
            )
            nn.init.xavier_uniform_(self.inp_transform[0].weight)
        elif use_input_norm:
            self.inp_transform = nn.LayerNorm(decoder_dim, eps=1e-5)
        else:
            self.inp_transform = None

        if use_output_transform:
            self.outp_transform = nn.Linear(decoder_dim, output_dim)
            nn.init.xavier_uniform_(self.outp_transform.weight)
            nn.init.zeros_(self.outp_transform.bias)
        else:
            self.outp_transform = None

        if use_positional_embedding:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches, decoder_dim) * decoder_dim**-0.5
            )
        else:
            self.pos_embed = None

        mask = torch.triu(torch.full((num_patches, num_patches), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        object_features: torch.Tensor,
        masks: torch.Tensor,
        target: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        empty_objects: Optional[torch.Tensor] = None,
    ) -> PatchReconstructionOutput:
        assert object_features.dim() >= 3  # Image or video data.
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),
                    scale_factor=self.upsample_target,
                    resize_mode=self.resize_mode,
                )
                .flatten(-2, -1)
                .transpose(-2, -1)
            )
        # Squeeze frames into batch if present.
        object_features = object_features.flatten(0, -3)

        object_features = self.cond_transform(object_features)

        # Squeeze frame into batch size if necessary.
        initial_targets_shape = target.shape[:-2]
        targets = target.flatten(0, -3)
        if self.bos_token is not None:
            bs = len(object_features)
            inputs = torch.cat((self.bos_token.expand(bs, -1, -1), targets[:, :-1].detach()), dim=1)
        else:
            inputs = targets

        if self.inp_transform is not None:
            inputs = self.inp_transform(inputs)

        if self.pos_embed is not None:
            # Simple learned additive embedding as in ViT
            inputs = inputs + self.pos_embed

        if empty_objects is not None:
            outputs = self.decoder(
                inputs,
                object_features,
                self.mask,
                memory_key_padding_mask=empty_objects,
            )
        else:
            outputs = self.decoder(inputs, object_features, self.mask)

        if self.use_decoder_masks:
            decoded_patches, masks = outputs
        else:
            decoded_patches = outputs

        if self.outp_transform is not None:
            decoded_patches = self.outp_transform(decoded_patches)

        decoded_patches = decoded_patches.unflatten(0, initial_targets_shape)

        if image is not None:
            masks_as_image = resize_patches_to_image(
                masks, size=image.shape[-1], resize_mode="bilinear"
            )
        else:
            masks_as_image = None

        return PatchReconstructionOutput(
            reconstruction=decoded_patches, masks=masks, masks_as_image=masks_as_image, target=target
        )
        
class PatchDecoderMLP(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes inputs of shape (B * K), P, N, and produce
            outputs of shape (B * K), P, M, where K is the number of objects, N is the number of
            input dimensions and M the number of output dimensions.
        decoder_input_dim: Input dimension to decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        slot_decoder: Callable[[int, int], nn.Module],
        decoder_input_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode
        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)  
            nn.init.xavier_uniform_(self.inp_transform.weight)  
            nn.init.zeros_(self.inp_transform.bias) 
        else:
            self.inp_transform = None  
            decoder_input_dim = object_dim 
        self.slot_dim = decoder_input_dim
        self.slot_decoder=slot_decoder(decoder_input_dim,1)
        self.decoder = decoder(decoder_input_dim, output_dim + 1)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_input_dim) * 0.02)
            
    def forward(
        self,
        object_features: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ):

        assert object_features.dim() >= 3 
        slot_weights = self.slot_decoder(object_features)
        slot_weights = torch.sigmoid(slot_weights)  # Restrict the output between (0, 1)
        # If an upsample target is set and the target tensor exists, perform upsampling to adjust the target size
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),  # Extract data from the target and change dimensions
                    scale_factor=self.upsample_target,  # Set the upsampling factor
                    resize_mode=self.resize_mode,  # Set the upsampling mode
                )
                .flatten(-2, -1)  # Flatten the tensor to prepare for dimension transformation
                .transpose(-2, -1)  # Transpose again to restore the original shape
            )
        initial_shape = object_features.shape[:-1]  # Save the original shape for final reshaping
        original_object_features = object_features

        object_features = object_features.flatten(0, -2)  # Flatten the object features for processing

        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        # Add a dimension and expand to match the number of image patches
        object_features = object_features.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = object_features + self.pos_embed

        # Process the position-enhanced features using the decoder
        output = self.decoder(object_features)

        output = output.unflatten(0, initial_shape)  # Restore the original batch shape
        # Split out alpha channel and normalize over slots. Separate the alpha channel and normalize it using softmax
        _, alpha = output.split([self.output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)
        masks = alpha.squeeze(-1)                                                                                                                               
        if not self.training:
            slot_weights = torch.where(slot_weights > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            B=slot_weights.shape[0]
            n_pred_clusters=masks.shape[1]
            pred_cluster_ids = torch.argmax(masks,dim=1)
            masks_oh=torch.nn.functional.one_hot(pred_cluster_ids,n_pred_clusters)
            masks_oh = masks_oh.permute(0, 2, 1).to(torch.float64)  
            masks_changed=slot_weights*masks
            binary_map_combined = masks_changed.sum(dim=1, keepdim=True)  
            background = 1 - binary_map_combined
            adjusted_masks = torch.cat([background, masks_changed], dim=1) 
        else:
            masks_changed = slot_weights * masks
            binary_map_combined = masks_changed.sum(dim=1, keepdim=True)  
            background = 1 - binary_map_combined
            adjusted_masks = torch.cat([background, masks_changed], dim=1) 
        if image is not None:
            masks_as_image = resize_patches_to_image(
                adjusted_masks, size=image.shape, resize_mode="bilinear"
            ) 
            binary_map_combined = resize_patches_to_image(
                binary_map_combined, size=image.shape, resize_mode="bilinear"
            )        
            masks_changed = resize_patches_to_image(
                masks_changed, size=image.shape, resize_mode="bilinear"
            ) 
            previous_masks= resize_patches_to_image(
                masks, size=image.shape, resize_mode="bilinear"
            )
        else:
            masks_as_image = None
        return MRPatchReconstructionOutput(
            reconstruction=binary_map_combined, 
            previous_masks=previous_masks,
            masks=adjusted_masks,
            masks_as_image=masks_as_image, 
            target=target if target is not None else None, 
            masks_changed=masks_changed,
            slot_features=original_object_features,
        )
