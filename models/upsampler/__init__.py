from typing import Literal, Union

from models.upsampler.sifigan import (
    Generator as SiFiGANGenerator,
    MultiPeriodAndScaleDiscriminator as SiFiGANMultiPeriodAndScaleDiscriminator,
    MultiPeriodAndResolutionDiscriminator as SiFiGANMultiPeriodAndResolutionDiscriminator
)
from models.upsampler.sfregan2 import (
    Generator as SFreGAN2Generator,
    MultiPeriodAndScaleDiscriminator as SFreGAN2MultiPeriodAndScaleDiscriminator,
    MultiPeriodAndResolutionDiscriminator as SFreGAN2MultiPeriodAndResolutionDiscriminator
)
from models.upsampler.config import UpsamplerConfig
from models.upsampler.utils import SignalGenerator


UpsamplerTypeConfig = Literal["sifigan", "sfregan2", "sfregan2m"]
UpsamplerType = Union[SiFiGANGenerator, SFreGAN2Generator]
DiscriminatorType = Union[
    SiFiGANMultiPeriodAndScaleDiscriminator,
    SiFiGANMultiPeriodAndResolutionDiscriminator,
    SFreGAN2MultiPeriodAndScaleDiscriminator,
    SFreGAN2MultiPeriodAndResolutionDiscriminator
]
