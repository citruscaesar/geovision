from typing import Literal, Optional

import torch
from geovision.io.local import FileSystemIO as fs

def get_state_dict(weights_init: Literal["torchvision", "torchgeo", "url", "path"], weights_param: Optional[str] = None):
    valid_weights_inits = ("torchvision", "torchgeo", "url", "path") 
    assert weights_init in valid_weights_inits, f"config error, expected :weights to be one of {valid_weights_inits}, got {weights_init}"

    if weights_init in ("torchvision", "torchgeo"):
        assert weights_param is not None, f"config error, did not expect :weights_param be None when :weights_init is {weights_init}"
        if weights_init == "torchvision":
            import torchvision
            weights = torchvision.models.get_weight(weights_param)
        elif weights_init == "torchgeo":
            raise NotImplementedError("loading torchgeo weights is not implemented yet, need to fix poetry for that")
            # import torchgeo.models
            # weights = torchgeo.models.get_weight(weights_param)
        return weights.get_state_dict()

    elif weights_init == "path":
        weights = fs.get_valid_file_err(weights_param)
        return torch.load(weights, weights_only=True)

    elif weights_init == "url":
        assert isinstance(weights_param, str), f"config error, expected :weights_param to be a valid url when :weights is url, got {weights_param}"
        raise NotImplementedError("loading weights from random URLs is not implemented yet")