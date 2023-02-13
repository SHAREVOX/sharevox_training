from typing import Literal, TypedDict, List, Tuple


class UpsamplerConfig(TypedDict):
    resblock: str
    share_upsamples: bool
    share_downsamples: bool
    upsample_rates: List[int]
    dense_factors: List[int]
    upsample_kernel_sizes:  List[int]
    upsample_initial_channel: int
    source_resblock_dilation_sizes: List[Tuple[int]]
    filter_resblock_kernel_sizes: List[int]
    filter_resblock_dilation_sizes: List[Tuple[int]]
    segment_size: int
    resblock: Literal["1", "2"]
