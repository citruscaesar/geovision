import torch
from torchvision.transforms import v2 as T # type: ignore

class TransformParser:
    ref = {
        "to_image": T.ToImage(),
        "to_float32_scaled": T.ToDtype(torch.float32, scale = True),
        "to_float32_unscaled": T.ToDtype(torch.float32, scale = False),
    }

    @classmethod
    def parse(cls, transforms: list[str]):
        parsed_transforms: list[T.Transform]
        for transform in transforms[1:]:
            pass


# transform: [composition, t1, t2, t3, ...]
# apply_composition: str, list -> composed

    @classmethod
    def compose(cls, composition: str, transforms: list[T.Transform]):
        p_str = composition.split('_')[-1]
        if p_str != '':
            p = float(p_str) / 100
            if "random_apply" in composition:
                return T.RandomApply(transforms, p) 
            elif "random_choice" in composition:
                return T.RandomChoice(transforms, p)
        elif composition == "compose":
            return T.Compose(transforms)
        elif composition == "random_order":
            return T.RandomOrder(transforms)
        else:
            raise ValueError(
                f""":key should be one of [compose, random_order, random_apply_p, random_choice_p], 
                got {composition}"""
            )