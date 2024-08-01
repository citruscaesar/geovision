from typing import Sequence, Any
import torchvision.transforms.v2 as T

class Compose(T.Transform):
    """applies a sequence of transforms while skipping over pixel(color) transforms, as they can mess up segmentation masks"""

    def __init__(self, transforms: Sequence[T.Transform]):
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of transforms")
        elif len(transforms) < 2:
            raise ValueError("Pass at least two transforms")
        self.transforms = transforms

    def forward(self, inputs: tuple[Any, Any]) -> Any:
        for transform in self.transforms:
            if self.is_color_transform(transform):
                continue
            outputs = transform(*inputs)
            inputs = outputs
        return outputs

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