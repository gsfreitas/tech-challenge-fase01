import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Define seeds globais para reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
