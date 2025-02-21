# /// script
# dependencies = [
#   "python-dotenv",
#   "numpy",
#   "wandb",
#   "torch",
#   "torchvision",
# ]
# [tool.uv]
# exclude-newer = "2024-02-20T00:00:00Z"
# ///

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
from zoneinfo import ZoneInfo
import wandb
from dotenv import dotenv_values
config = dotenv_values(".env")


def create_papertrail_logger(config):
    log_name = config.get("LOG_NAME", "MNIST")
    timezone = config.get("TIMEZONE", "UTC")
    papertrail_host = config.get("PAPERTRAIL_HOST")
    papertrail_port = config.get("PAPERTRAIL_PORT")

    def papertrail(message):
        timestamp = datetime.now(ZoneInfo(timezone))
        if papertrail_host and papertrail_port:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    priority = 22
                    hostname = ""
                    syslog_message = (
                        f"<{priority}>{timestamp.strftime("%b %d %H:%M:%S")} "
                        f"{hostname} {log_name}: {message}"
                    )
                    sock.sendto(
                        syslog_message.encode("utf-8"),
                        (papertrail_host, int(papertrail_port)),
                    )
            except:
                pass
        print(f'{timestamp.strftime("%H:%M:%S")} {message}', flush=True)

    return papertrail


log = create_papertrail_logger(config)
log("Test message")