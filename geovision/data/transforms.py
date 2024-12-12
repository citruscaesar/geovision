from typing import Any, Sequence

import logging

from torch import Tensor
from torchvision.transforms.v2 import Transform, ToDtype

logger = logging.getLogger(__name__)

# TODO:
# Write custom Compose which takes in (image, mask), applies all transforms to the image, but does not apply
# any of the radiometric, i.e. Color and ToDtype transfroms to Maks


class SegmentationCompose(Transform):
    """applies a sequence of transforms while skipping over pixel(color) transforms, as they can mess up segmentation masks"""

    def __init__(self, transforms: Sequence[Transform]):
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of transforms")
        elif len(transforms) < 2:
            raise ValueError("Pass at least two transforms")
        self.transforms = transforms

    def forward(self, *inputs) -> Any:
        for transform in self.transforms:

            if self.is_color_transform(transform):
                continue

            if isinstance(transform, ToDtype):
                image = transform(inputs[0])
                transform.scale = False
                mask = transform(inputs[1])
                inputs = (image, mask)
                continue

            inputs = transform(inputs)
        return inputs 

    def extra_repr(self) -> str:
        format_string = []
        for transform in self.transforms:
            if self.is_color_transform(transform):
                format_string.append(f"   *{transform}")
            else:
                format_string.append(f"    {transform}")
        return "\n".join(format_string)
    
    def is_color_transform(self, transform) -> bool:
        "_color" in str(type(transform))
