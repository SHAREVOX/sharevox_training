import os
import glob
from typing import Optional, Tuple

import torch
from torch import nn, optim, Tensor
from tqdm import tqdm


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    printf=tqdm.write,
) -> Tuple[nn.Module, optim.Optimizer, Tensor, int, int]:
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    reset = "G_0" in checkpoint_path or "D_0" in checkpoint_path
    if reset:
        iteration = 1
    else:
        iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and not reset:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if reset and "extractor" in k:
            new_state_dict[k] = v
            continue
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            printf("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    printf(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    step = int(str(os.path.basename(checkpoint_path)).split(".")[0].split("_")[1])
    return model, optimizer, learning_rate, iteration, step


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, learning_rate: float, iteration: int, checkpoint_path: str, printf=tqdm.write,
):
    printf(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )
    printf("Saved!")


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x
