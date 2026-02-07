# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Net config options for cosmos/tokenizer

ContinuousImageTokenizerConfig
DiscreteImageTokenizerConfig
CausalContinuousVideoTokenizerConfig

"""
from dataclasses import field
from horizon_imagination.models.tokenizer.cosmos.modules import (
    ContinuousFormulation,
    Decoder3DType,
    DecoderType,
    DiscreteQuantizer,
    Encoder3DType,
    EncoderType,
)
from horizon_imagination.utilities.config import BaseConfig
from horizon_imagination.models.tokenizer.cosmos.networks.continuous_image import ContinuousImageTokenizer
from horizon_imagination.models.tokenizer.cosmos.networks.continuous_video import CausalContinuousVideoTokenizer
from horizon_imagination.models.tokenizer.cosmos.networks.discrete_image import DiscreteImageTokenizer
from horizon_imagination.models.tokenizer.cosmos.networks.discrete_video import CausalDiscreteVideoTokenizer


class ContinuousImageTokenizerConfig(BaseConfig):
    # The attention resolution for res blocks.
    attn_resolutions: tuple[int,...] = (32,)
    # The base number of channels.
    channels: int = 128
    # The channel multipler for each resolution.
    channels_mult: tuple[int] = (2, 4, 4)
    dropout: float = 0.0
    in_channels: int = 3
    # The spatial compression ratio, default 8.
    spatial_compression: int = 8
    # The number of layers in each res block.
    num_res_blocks: int = 2
    out_channels: int = 3
    resolution: int = 1024
    patch_size: int = 2
    patch_method: str = "haar"
    # The output latent dimension (channels).
    latent_channels: int = 16
    # The encoder output channels just before sampling.
    # Which is also the decoder's input channels.
    z_channels: int = 16
    # A factor over the z_channels, to get the total channels the encoder should output.
    # For a VAE for instance, we want to output the mean and variance, so we need 2 * z_channels.
    # Since we are using AE formulation, we only need the mean, so z_factor=1.
    z_factor: int = 1
    name: str = "ContinuousImageTokenizer"
    # What formulation to use, either "AE" or "VAE".
    # Chose AE here, since this has been proven to be effective.
    formulation: str = ContinuousFormulation.AE.name
    # Specify type of encoder ["Default", "LiteVAE"]
    encoder: str = EncoderType.Default.name
    # Specify type of decoder ["Default"]
    decoder: str = DecoderType.Default.name


ContinuousImageTokenizerConfig._target_cls = ContinuousImageTokenizer


class DiscreteImageTokenizerConfig(BaseConfig):
    # The attention resolution for res blocks.
    attn_resolutions: tuple[int, ...] = (32,)
    # The base number of channels.
    channels: int = 128
    # The channel multipler for each resolution.
    channels_mult: tuple[int, ...] = (2, 4, 4)
    dropout: float = 0.0
    in_channels: int = 3
    # The spatial compression ratio.
    spatial_compression: int = 16
    # The number of layers in each res block.
    num_res_blocks: int = 2
    out_channels: int = 3
    resolution: int = 1024
    patch_size: int = 2
    patch_method="haar"
    # The encoder output channels just before sampling.
    z_channels: int = 256
    # A factor over the z_channels, to get the total channels the encoder should output.
    # for discrete tokenization, often we directly use the vector, so z_factor=1.
    z_factor: int = 1
    # The quantizer of choice, VQ, LFQ, FSQ, or ResFSQ. Default FSQ.
    quantizer: str = DiscreteQuantizer.FSQ.name
    # The embedding dimension post-quantization, which is also the input channels of the decoder.
    # Which is also the output
    embedding_dim: int = 6
    # The number of levels to use for fine-scalar quantization.
    levels: tuple[int, ...] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    persistent_quantizer=False
    # The number of quantizers to use for residual fine-scalar quantization.
    num_quantizers: int = 4
    name: str = "DiscreteImageTokenizer"
    # Specify type of encoder ["Default", "LiteVAE"]
    encoder: str = EncoderType.Default.name
    # Specify type of decoder ["Default"]
    decoder: str = DecoderType.Default.name


DiscreteImageTokenizerConfig._target_cls = DiscreteImageTokenizer


class CausalContinuousFactorizedVideoTokenizerConfig(BaseConfig):
    # The new causal continuous tokenizer, that is at least 2x more efficient in memory and runtime.
    # - It relies on fully 3D discrete wavelet transform
    # - Uses a layer norm instead of a group norm
    # - Factorizes full convolutions into spatial and temporal convolutions
    # - Factorizes full attention into spatial and temporal attention
    # - Adopts an AE formulation
    # - Strictly causal, with flexible temporal length at inference.
    attn_resolutions=(32,),
    channels=128,
    channels_mult=(2, 4, 4),
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    latent_channels=16,
    z_channels=16,
    z_factor=1,
    num_groups=1,
    # Most of the CV and DV tokenizers trained before September 1, 2024,
    # used temporal upsampling that was not perfectly mirrored with the
    # # encoder's temporal downsampling. Moving forward, new CV/DV tokenizers
    # will use legacy_mode=False, meaning they will adopt mirrored upsampling.
    legacy_mode=False,
    spatial_compression=8,
    temporal_compression=8,
    formulation=ContinuousFormulation.AE.name,
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="CausalContinuousFactorizedVideoTokenizer",


CausalContinuousFactorizedVideoTokenizerConfig._target_cls = CausalContinuousVideoTokenizer


class CausalDiscreteFactorizedVideoTokenizerConfig(BaseConfig):
    # The new causal discrete tokenizer, that is at least 2x more efficient in memory and runtime.
    # - It relies on fully 3D discrete wavelet transform
    # - Uses a layer norm instead of a group norm
    # - Factorizes full convolutions into spatial and temporal convolutions
    # - Factorizes full attention into spatial and temporal attention
    # - Strictly causal, with flexible temporal length at inference.
    attn_resolutions=(32,),
    channels=128,
    channels_mult=(2, 4, 4),
    dropout=0.0,
    in_channels=3,
    num_res_blocks=2,
    out_channels=3,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    # The encoder output channels just before quantization is changed to 256
    # from 16 (old versions). It aligns with the DI that uses 256 channels,
    # making initialization from image tokenizers easier.
    z_channels=256,
    z_factor=1,
    num_groups=1,
    # Most of the CV and DV tokenizers trained before September 1, 2024,
    # used temporal upsampling that was not perfectly mirrored with the
    # # encoder's temporal downsampling. Moving forward, new CV/DV tokenizers
    # will use legacy_mode=False, meaning they will adopt mirrored upsampling.
    legacy_mode=False,
    spatial_compression=16,
    temporal_compression=8,
    quantizer=DiscreteQuantizer.FSQ.name,
    embedding_dim=6,
    levels=(8, 8, 8, 5, 5, 5),
    persistent_quantizer=False,
    encoder=Encoder3DType.FACTORIZED.name,
    decoder=Decoder3DType.FACTORIZED.name,
    name="CausalDiscreteFactorizedVideoTokenizer",


CausalDiscreteFactorizedVideoTokenizerConfig._target_cls = CausalDiscreteVideoTokenizer
