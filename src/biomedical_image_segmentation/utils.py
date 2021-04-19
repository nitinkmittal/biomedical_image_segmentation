from typing import Any, List, Tuple


from numpy import ndarray
import numpy as np
from pickle import load
import os
import torch

def split(
    a: List[Any],
    ratio: Tuple[int, int, int] = (2 / 3, 1 / 6, 1 / 6),
    replace: bool = False,
    seed: int = 40,
):
    """Split input into train, valid and test set."""
    
    random = np.random.RandomState(seed)
    total = len(a)
    train = random.choice(
        a, replace=replace, size=int(total * ratio[0])
    ).tolist()
    valid = random.choice(
        [b for b in a if b not in train],
        replace=replace,
        size=int(total * ratio[1]),
    ).tolist()
    test = random.choice(
        [c for c in a if c not in train + valid],
        replace=replace,
        size=int(total * ratio[2]),
    ).tolist()
    return train, valid, test

def load_pickle(filename: str) -> Any:
    """Load and return pickle."""
    with open(filename, "rb") as f:
        data = load(f)
    return data


def create_dir(path: str, verbose: bool=False):
    """Create folder if does not exists."""
    if not os.path.exists(path):
        try: 
            os.mkdir(path) 
        except OSError as error: 
            if verbose: print(error)
                
def empty_dir(path: str, verbose: bool=False):
    """Delete content of a directory recursively."""
    assert os.path.exists(path)
    for f in os.listdir(path):
        try:
            os.remove(os.path.join(path, f))
        except Exception as error:
            if verbose: print(error)
                