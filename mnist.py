# /// script
# dependencies = [
#   "numpy",
#   "wandb",
#   "torch",
#   "torchvision",
# ]
# [tool.uv]
# exclude-newer = "2024-02-20T00:00:00Z"
# ///

import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from datetime import datetime
import time
import torch.profiler
import hashlib
import os
import math
import socket
from datetime import datetime
from zoneinfo import ZoneInfo
import wandb