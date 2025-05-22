# /// script
# dependencies = [
#   "python-dotenv",
#   "numpy",
#   "wandb",
#   "torch",
#   "torchvision",
#   "IPython",
#   "python-telegram-bot",
#   "asyncio",
# ]
# [tool.uv]
# exclude-newer = "2024-02-20T00:00:00Z"
# ///
# pip install wandb python-dotenv python-telegram-bot asyncio

import math
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
import os
import socket
from zoneinfo import ZoneInfo
import wandb
import ast

import torch.profiler
############################ CONFIG ########################

from dotenv import dotenv_values
config = { **dotenv_values(".env"), **os.environ }

# PASS_INPUT_TO_ALL_LAYERS=1 C_INIT="XAVIER_U" C_SPARSITY=10 G_SPARSITY=1 OPT_GATE16_CODEPATH=3 KINETO_LOG_LEVEL=99 GATE_ARCHITECTURE="[2000,2000]" INTERCONNECT_ARCHITECTURE="[]" PRINTOUT_EVERY=211 EPOCHS=300 uv run mnist.py
# CONNECTIVITY_GAIN=1 LEARNING_RATE=0.001 PASS_RESIDUAL=0 PASS_INPUT_TO_ALL_LAYERS=0 C_INIT="NORMAL" C_SPARSITY=100 C_INIT_PARAM=0 GATE_ARCHITECTURE="[250,250,250,250,250,250]"PRINTOUT_EVERY=55 VALIDATE_EVERY=211 EPOCHS=200 uv run mnist.py
# LEARNING_RATE=0.075 C_SPARSITY=1 NO_SOFTMAX=1 SCALE_TARGET=1.5 SCALE_LOGITS="ADAVAR" TAU_LR=.03 MANUAL_GAIN=1 PASS_RESIDUAL=0 PASS_INPUT_TO_ALL_LAYERS=0 KINETO_LOG_LEVEL=99 GATE_ARCHITECTURE="[8000]" INTERCONNECT_ARCHITECTURE="[]" PRINTOUT_EVERY=211 VALIDATE_EVERY=1055 IMG_WIDTH=28 EPOCHS=30 uv run mnist.py

LOG_TAG = config.get("LOG_TAG", "MNIST")
TIMEZONE = config.get("TIMEZONE", "UTC")
PAPERTRAIL_HOST = config.get("PAPERTRAIL_HOST")
PAPERTRAIL_PORT = config.get("PAPERTRAIL_PORT")
WANDB_KEY = config.get("WANDB_KEY")
WANDB_PROJECT = config.get("WANDB_PROJECT", "mnist_project")

BINARIZE_IMAGE_TRESHOLD = float(config.get("BINARIZE_IMAGE_TRESHOLD", 0.75))
IMG_WIDTH = int(config.get("IMG_WIDTH", 28)) # previous 16 which was suitable for Tiny Tapeout
INPUT_SIZE = IMG_WIDTH * IMG_WIDTH
DATA_SPLIT_SEED = int(config.get("DATA_SPLIT_SEED", 42))
TRAIN_FRACTION = float(config.get("TRAIN_FRACTION", 0.99)) # previous 0.9, NOTE: since we are not using VALIDATION subset in VALIDATION pass, we can boost accuracy a bit by training on the whole dataset
NUMBER_OF_CATEGORIES = int(config.get("NUMBER_OF_CATEGORIES", 10))
ONLY_USE_DATA_SUBSET = config.get("ONLY_USE_DATA_SUBSET", "0").lower() in ("true", "1", "yes")

SEED = config.get("SEED", random.randint(0, 1024*1024))
LOG_NAME = f"{LOG_TAG}_{SEED}"
GATE_ARCHITECTURE = ast.literal_eval(config.get("GATE_ARCHITECTURE", "[8000,8000]")) # previous: [1300,1300,1300] resnet95: [2000, 2000] conn_gain96: [2250, 2250, 2250] power_law_fixed97: [8000,8000,8000, 8000,8000,8000] auto_scale_logits_97_5 [8000]
INTERCONNECT_ARCHITECTURE = ast.literal_eval(config.get("INTERCONNECT_ARCHITECTURE", "[[],['k',5]]")) # previous: [[32, 325], [26, 52], [26, 52]])) resnet95, conn_gain96, auto_scale_logits_97_5: [] power_law_fixed97: [[],[-1],[-1], [-1],[-1],[-1]] p
if not INTERCONNECT_ARCHITECTURE or INTERCONNECT_ARCHITECTURE == []:
    INTERCONNECT_ARCHITECTURE = [[] for g in GATE_ARCHITECTURE]
assert len(GATE_ARCHITECTURE) == len(INTERCONNECT_ARCHITECTURE)
BATCH_SIZE = int(config.get("BATCH_SIZE", 256))

EPOCHS = int(config.get("EPOCHS", 30)) # previous: 50
EPOCH_STEPS = math.floor((60_000 * TRAIN_FRACTION) / BATCH_SIZE) # MNIST consists of 60K images
TRAINING_STEPS = EPOCHS*EPOCH_STEPS
PRINTOUT_EVERY = int(config.get("PRINTOUT_EVERY", EPOCH_STEPS))
VALIDATE_EVERY = int(config.get("VALIDATE_EVERY", EPOCH_STEPS * 5))

LEARNING_RATE = float(config.get("LEARNING_RATE", 0.075)) # conn_gain96, power_law_fixed_conn97: 0.03

TG_TOKEN = config.get("TG_TOKEN")
TG_CHATID = config.get("TG_CHATID")

SUPPRESS_PASSTHROUGH = config.get("SUPPRESS_PASSTHROUGH", "0").lower() in ("true", "1", "yes")
SUPPRESS_CONST = config.get("SUPPRESS_CONST", "0").lower() in ("true", "1", "yes")
TENSION_REGULARIZATION = float(config.get("TENSION_REGULARIZATION", -1))

PROFILE = config.get("PROFILE", "0").lower() in ("true", "1", "yes")
if PROFILE: prof = torch.profiler.profile(schedule=torch.profiler.schedule(skip_first=10, wait=3, warmup=1, active=1, repeat=1000), record_shapes=True, with_flops=True) #, with_stack=True, with_modules=True)
PROFILER_ROWS = int(config.get("PROFILER_ROWS", 20))

FORCE_CPU = config.get("FORCE_CPU", "0").lower() in ("true", "1", "yes")
COMPILE_MODEL = config.get("COMPILE_MODEL", "0").lower() in ("true", "1", "yes")

C_INIT = config.get("C_INIT", "NORMAL") # NORMAL, UNIFORM, EXP_U, LOG_U, XAVIER_N, XAVIER_U, KAIMING_OUT_N, KAIMING_OUT_U, KAIMING_IN_N, KAIMING_IN_U
G_INIT = config.get("G_INIT", "NORMAL") # NORMAL, UNIFORM, PASSTHROUGH, XOR
C_INIT_PARAM = float(config.get("C_INIT_PARAM", -1.0))
C_SPARSITY = float(config.get("C_SPARSITY", 1.0)) # NOTE: 1.0 works well only for SHALLOW nets, 3.0 for deeper is necessary to binarize well
G_SPARSITY = float(config.get("G_SPARSITY", 1.0))

PASS_INPUT_TO_ALL_LAYERS = config.get("PASS_INPUT_TO_ALL_LAYERS", "0").lower() in ("true", "1", "yes") # previous: 1
PASS_RESIDUAL = config.get("PASS_RESIDUAL", "0").lower() in ("true", "1", "yes")

MANUAL_GAIN = float(config.get("MANUAL_GAIN", 1.0))

NO_SOFTMAX = config.get("NO_SOFTMAX", "1").lower() in ("true", "1", "yes") # previous: 0
SCALE_LOGITS = config.get("SCALE_LOGITS", "ADAVAR") # TANH, ARCSINH, BN, MINMAX, ADATAU, ADAVAR, AUTOAU, ADAVAR_ASH, AUTOAU_ASH
SCALE_TARGET = float(config.get("SCALE_TARGET", 1.5)) # NOTE: 1.5 works well only for SHALLOW nets, 1.0 is better for deeper networks combined with C_SPARCITY=3
if SCALE_TARGET == 0:
    SCALE_LOGITS = "0"
TAU_LR = float(config.get("TAU_LR", 0.03))
if TAU_LR == 0 and SCALE_LOGITS.startswith("ADATAU"):
    SCALE_LOGITS = "0"

DROPOUT = float(config.get("DROPOUT", 0.0))

config_printout_keys = ["LOG_NAME", "TIMEZONE", "WANDB_PROJECT",
               "BINARIZE_IMAGE_TRESHOLD", "IMG_WIDTH", "INPUT_SIZE", "DATA_SPLIT_SEED", "TRAIN_FRACTION", "NUMBER_OF_CATEGORIES", "ONLY_USE_DATA_SUBSET",
               "SEED", "GATE_ARCHITECTURE", "INTERCONNECT_ARCHITECTURE", "BATCH_SIZE",
               "EPOCHS", "EPOCH_STEPS", "TRAINING_STEPS", "PRINTOUT_EVERY", "VALIDATE_EVERY",
               "LEARNING_RATE",
               "C_INIT", "C_INIT_PARAM", "G_INIT", "C_SPARSITY", "G_SPARSITY",
               "PASS_INPUT_TO_ALL_LAYERS", "PASS_RESIDUAL",
               "NO_SOFTMAX",
               "MANUAL_GAIN",
               "SCALE_LOGITS", "SCALE_TARGET", "TAU_LR",
               "DROPOUT",
               "SUPPRESS_PASSTHROUGH", "SUPPRESS_CONST", "TENSION_REGULARIZATION",
               "PROFILE", "FORCE_CPU", "COMPILE_MODEL"]
config_printout_dict = {key: globals()[key] for key in config_printout_keys}

# Making sure sensitive configs are not logged
assert "TG_TOKEN" not in config_printout_dict.keys()
assert "TG_CHATID" not in config_printout_dict.keys()
assert "PAPERTRAIL_HOST" not in config_printout_dict.keys()
assert "PAPERTRAIL_PORT" not in config_printout_dict.keys()
assert "WANDB_KEY" not in config_printout_dict.keys()

if WANDB_KEY is not None:
    wandb.login(key=WANDB_KEY)
    wandb_run = wandb.init(project=WANDB_PROJECT, name=LOG_NAME, config=config_printout_dict)
    script_path = os.path.abspath(__file__)
    artifact = wandb.Artifact("source_code", type="code")
    artifact.add_file(script_path)
    wandb.log_artifact(artifact)

############################ LOG ########################

def create_papertrail_logger(log_tag, timezone, papertrail_host, papertrail_port):
    def papertrail(message):
        timestamp = datetime.now(ZoneInfo(timezone))
        if (papertrail_host is not None) and (papertrail_port is not None):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    priority = 22
                    hostname = ""
                    syslog_message = (
                        f"<{priority}>{timestamp.strftime('%b %d %H:%M:%S')} "
                        f"{hostname} {log_tag}: {message}"
                    )
                    sock.sendto(
                        syslog_message.encode("utf-8"),
                        (papertrail_host, int(papertrail_port)),
                    )
            except:
                pass
        print(f'{timestamp.strftime("%H:%M:%S")} {message}', flush=True)
    return papertrail

log = create_papertrail_logger(LOG_TAG, TIMEZONE, PAPERTRAIL_HOST, PAPERTRAIL_PORT)

if WANDB_KEY is None:
    log("-"*80)
    for k in config_printout_dict.keys():
        log(f"{k}={config_printout_dict[k]}")
    log("-"*80)

############################ DEVICE ########################

try:
    device = torch.device(
                    "cuda" if torch.cuda.is_available()         and not FORCE_CPU else 
                    "mps"  if torch.backends.mps.is_available() and not FORCE_CPU else 
                    "cpu")
except:
    device = torch.device("cpu")
WANDB_KEY and wandb.log({"device": str(device)})

#################### TENSOR BINARIZATION ##################

def binarize_inplace(x, dim=-1, bin_value=1):
    ones_at = torch.argmax(x, dim=dim)
    x.data.zero_()
    x.data.scatter_(dim=dim, index=ones_at.unsqueeze(dim), value=bin_value)

############################ MODEL ########################
class Dropout01(nn.Module):
    def __init__(self, p: float = 0.5):
        super(Dropout01, self).__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in the range [0, 1).")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        pred = torch.rand_like(x) > self.p
        mask = pred.float()
        zero_one = (torch.rand_like(x) < 0.5).float()

        # return mask * x + (1 - mask) * zero_one
        return torch.where(pred, x, zero_one)

    def __repr__(self):
        return f"Dropout01(p={self.p})"

class DropoutFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super(DropoutFlip, self).__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in the range [0, 1).")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        return torch.where(torch.rand_like(x) > self.p, x, 1-x)

    def __repr__(self):
        return f"DropoutFlip(p={self.p})"

class FixedPowerLawInterconnect(nn.Module):
    def __init__(self, inputs, outputs, alpha, x_min=1.0, name=''):
        super(FixedPowerLawInterconnect, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.alpha = alpha

        max_length = inputs
        size = outputs
        r = torch.rand(size)
        if alpha > 1:
            magnitudes = x_min * (1 - r) ** (-1 / (alpha - 1))          # Power law distribution
            signs = torch.randint(low=0, high=2, size=(size,)) * 2 - 1  # -1 or +1
            offsets = magnitudes * signs * max_length
        else:
            offsets = r * max_length
        indices = torch.arange(start=0, end=size) + offsets.long()
        indices = indices % max_length
        self.register_buffer("indices", indices)

        self.binarized = False

        # self.batch_indices = indices.unsqueeze(0)

    @torch.profiler.record_function("mnist::Fixed::FWD")
    def forward(self, x):
        return x[:, self.indices] if not self.binarized else torch.matmul(x, self.c)

        # Performance comparison
        # 1) x[:, self.indices]
        # MPS: 4.29 ms per iteration [300,300], tiny bit faster
        # MPS: 9.93 ms per iteration [3000,3000]
        # Takes significantly less memory though!

        # 2)
        # batch_size = x.shape[0]
        # if self.batch_indices.shape[0] != batch_size:
        #     self.batch_indices = self.indices.repeat(batch_size, 1)
        # return torch.gather(x, dim=1, index=self.batch_indices) 
        # MPS: 4.57 ms per iteration [300,300]
        # MPS: 9.32 ms per iteration [3000,3000], tiny bit faster

    def binarize(self, bin_value=1):
        with torch.no_grad():
            c = torch.zeros((self.inputs, self.outputs), dtype=torch.float32, device=device)
            c.scatter_(dim=0, index=self.indices.unsqueeze(0), value=bin_value)
            self.register_buffer("c", c)
            self.binarized = True

    def __repr__(self):
        with torch.no_grad():
            i = self.indices.view(self.outputs // 2, 2) # [batch_size, number_of_gates, 2]
            A = i[:,0]
            B = i[:,1]

            d = torch.abs(A-B)
            d = torch.minimum(d, self.inputs - d)

            fanout = torch.bincount(self.indices).max().item()
            return f"FixedPowerLawInterconnect({self.inputs} -> {self.outputs // 2}x2, α={self.alpha}, mean={d.float().mean().long()} median={d.float().median().long()} fanout={fanout})"

class SparseInterconnect(nn.Module):
    def __init__(self, inputs, outputs, name=''):
        super(SparseInterconnect, self).__init__()
        self.c = nn.Parameter(torch.zeros((inputs, outputs), dtype=torch.float32))
        if   C_INIT == "XAVIER_N":
            nn.init.xavier_normal_(self.c, gain=nn.init.calculate_gain('sigmoid'))
        elif C_INIT == "XAVIER_U":
            nn.init.xavier_uniform_(self.c, gain=nn.init.calculate_gain('sigmoid'))
        elif C_INIT == "KAIMING_OUT_N":
            nn.init.kaiming_normal_(self.c, mode="fan_out", nonlinearity='sigmoid')
        elif C_INIT == "KAIMING_OUT_U":
            nn.init.kaiming_uniform_(self.c, mode="fan_out", nonlinearity='sigmoid')
        elif C_INIT == "KAIMING_IN_N":
            nn.init.kaiming_normal_(self.c, mode="fan_in", nonlinearity='sigmoid')
        elif C_INIT == "KAIMING_IN_U":
            nn.init.kaiming_uniform_(self.c, mode="fan_in", nonlinearity='sigmoid')
        elif C_INIT == "EXP_U":
            nn.init.uniform_(self.c, a=0.0, b=1.0)
            with torch.no_grad(): self.c.data = torch.exp(self.c)
        elif C_INIT == "LOG_U":
            nn.init.uniform_(self.c, a=0.0, b=1.0)
            with torch.no_grad(): self.c.data = torch.log(self.c)
        # ChatGPT: Orthogonal initialization tends to spread rows out in the space, and may help ensure different dominant directions per row.
        elif C_INIT == "ORTHO":
            torch.nn.init.orthogonal_(self.c, gain=nn.init.calculate_gain('sigmoid'))
        # ChatGPT: Orthogonal initialization tends to spread rows out in the space, and may help ensure different dominant directions per row.
        elif C_INIT == "QR":
            with torch.no_grad():
                max_dim = max(self.c.shape[0], self.c.shape[1])
                cc = torch.normal(mean=0.0, std=1, size=(max_dim, max_dim))
                cc, _ = torch.linalg.qr(cc)
                self.c.data = cc[:self.c.shape[0], :self.c.shape[1]]
        # ChatGPT: A Dirichlet-distributed row will have a single dominant component if the concentration parameter is low.
        elif C_INIT == "DIRICHLET":
            with torch.no_grad():
                alpha = (0.1 if C_INIT_PARAM < 0 else C_INIT_PARAM) * torch.ones(self.c.shape[1])
                self.c.data = torch.distributions.Dirichlet(alpha).sample((self.c.shape[0],))
        # ChatGPT: Start with uniform or Gaussian, apply a sharpened softmax, then back to logit-space.
        elif C_INIT == "SOFTMAX_SHARPEN_N":
            nn.init.normal_(self.c, mean=0.0, std=1)
            with torch.no_grad():
                T = 0.5 if C_INIT_PARAM < 0 else C_INIT_PARAM   # temperature
                W = torch.nn.functional.softmax(self.c / T, dim=0)
                self.c.data = torch.log(W)
        elif C_INIT == "SOFTMAX_SHARPEN_U":
            nn.init.uniform_(self.c, a=0.0, b=1.0)
            with torch.no_grad():
                T = 0.5 if C_INIT_PARAM < 0 else C_INIT_PARAM   # temperature
                W = torch.nn.functional.softmax(self.c / T, dim=0)
                self.c.data = torch.log(W)
        elif C_INIT == "UNIFORM":
            nn.init.uniform_(self.c, a=0.0, b=1.0)
        else:
            nn.init.normal_(self.c, mean=0.0, std=1)
        self.name = name
        self.binarized = False
    
    @torch.profiler.record_function("mnist::Sparse::FWD")
    def forward(self, x):
        connections = F.softmax(self.c * C_SPARSITY, dim=0) if not self.binarized else self.c
        return torch.matmul(x, connections)

    def binarize(self, bin_value=1):
        binarize_inplace(self.c, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"SparseInterconnect({self.c.shape[0]} -> {self.c.shape[1] // 2}x2)"

class BlockSparseInterconnect(nn.Module):
    def __init__(self, layer_inputs, layer_outputs, granularity, name=''):
        super(BlockSparseInterconnect, self).__init__()
        self.layer_inputs = layer_inputs
        self.layer_outputs = layer_outputs
        # simple:
        # 1024:[32x32]x32 ==transpose=> 32x[32x32]:1024
        # 1280:[32x32]x40 ==transpose=> 32x[40x40]:1280
        # 1300:[25x25]x52 ==transpose=> 25x[52x52]:1300                                                                         32500+67600=100100 vs 1300x1300 ~~ 5.93%
        # 1300:[50x50]x26 ==transpose=> 50x[26x26]:1300                                                                         65000+33800=98800 vs 1300x1300  ~~ 5.93%
        #   IN:[GGxGG]xII ==transpose=> GGx[IIxYY]:OUT   where GG=granularity, II=IN//granularity, YY=OUT/granularity
        # complex:
        #  256:[64x325]x4 ==transpose=> 325x[4x4]:1300                                                                          83200+5200=88400 vs 256x1300    ~~ 26.56%
        #   IN:[GGxBB]xAA ==transpose=> BBx[AAxAA]:OUT   where GG=granularity, AA=IN//granularity, BB=OUT/AA
        self.n_blocks_in_sub_layer_1 = layer_inputs // granularity
        self.inputs_per_block_in_sub_layer_1  = layer_inputs // self.n_blocks_in_sub_layer_1
        self.inputs_per_block_in_sub_layer_2  = self.n_blocks_in_sub_layer_1
        # self.n_blocks_in_sub_layer_2 = granularity # simple
        # self.outputs_per_block_in_sub_layer_1 = granularity # simple
        # self.outputs_per_block_in_sub_layer_2 = layer_outputs // granularity # simple
        self.n_blocks_in_sub_layer_2 = layer_outputs // self.n_blocks_in_sub_layer_1 # complex
        self.outputs_per_block_in_sub_layer_1 = self.n_blocks_in_sub_layer_2 # complex
        self.outputs_per_block_in_sub_layer_2 = layer_outputs // self.n_blocks_in_sub_layer_2 # complex

        self.name = name
        self.binarized = False
        
        assert layer_inputs  == self.n_blocks_in_sub_layer_1 * self.inputs_per_block_in_sub_layer_1,  f"name={self.name}, sub(1): inputs={layer_inputs} n_blocks={self.n_blocks_in_sub_layer_1} inputs_per_block={self.inputs_per_block_in_sub_layer_1}"
        assert layer_outputs == self.n_blocks_in_sub_layer_2 * self.outputs_per_block_in_sub_layer_2, f"name={self.name}, sub(2): outputs={layer_outputs//2} n_blocks={self.n_blocks_in_sub_layer_2} outputs_per_block={self.outputs_per_block_in_sub_layer_2//2}"
        assert self.n_blocks_in_sub_layer_1 * self.outputs_per_block_in_sub_layer_1 == self.n_blocks_in_sub_layer_2 * self.inputs_per_block_in_sub_layer_2, f"name={self.name}, sub(1): n_blocks={self.n_blocks_in_sub_layer_1} outputs_per_block={self.outputs_per_block_in_sub_layer_1}, " + \
                                                                                                                                                                              f"sub(2): n_blocks={self.n_blocks_in_sub_layer_2}  inputs_per_block={self.inputs_per_block_in_sub_layer_2}"
        self.c_sub_layer_1 = nn.Parameter(torch.zeros((self.n_blocks_in_sub_layer_1, self.inputs_per_block_in_sub_layer_1, self.outputs_per_block_in_sub_layer_1), dtype=torch.float32))
        self.c_sub_layer_2 = nn.Parameter(torch.zeros((self.n_blocks_in_sub_layer_2, self.inputs_per_block_in_sub_layer_2, self.outputs_per_block_in_sub_layer_2), dtype=torch.float32))
        if C_INIT == "UNIFORM":
            nn.init.uniform_(self.c_sub_layer_1, a=0.0, b=1)
            nn.init.uniform_(self.c_sub_layer_2, a=0.0, b=1)
        else:
            nn.init.normal_(self.c_sub_layer_1, mean=0.0, std=1)
            nn.init.normal_(self.c_sub_layer_2, mean=0.0, std=1)

    
    @torch.profiler.record_function("mnist::BlockSparse::FWD")
    def forward(self, x):
        conn_1 = F.softmax(self.c_sub_layer_1 * C_SPARSITY, dim=1) if not self.binarized else self.c_sub_layer_1
        conn_2 = F.softmax(self.c_sub_layer_2 * C_SPARSITY, dim=1) if not self.binarized else self.c_sub_layer_2

        x_reshaped = x.view(-1, self.n_blocks_in_sub_layer_1, self.inputs_per_block_in_sub_layer_1)
        output = torch.einsum("bni,nim,mno->bmo", x_reshaped, conn_1, conn_2)
        return output.reshape(x.shape[0], self.layer_outputs)

    def binarize(self, bin_value=1):
        binarize_inplace(self.c_sub_layer_1, dim=1, bin_value=bin_value)
        binarize_inplace(self.c_sub_layer_2, dim=1, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        fc_params = self.layer_inputs * self.layer_outputs
        params  = self.n_blocks_in_sub_layer_1 * self.inputs_per_block_in_sub_layer_1 * self.outputs_per_block_in_sub_layer_1 + \
                  self.n_blocks_in_sub_layer_2 * self.inputs_per_block_in_sub_layer_2 * self.outputs_per_block_in_sub_layer_2
        percent = params * 100 / fc_params
        return f"BlockSparseInterconnect({self.layer_inputs} -> {self.layer_outputs // 2}x2 @ {percent:.1f}%)"

# EPOCHS=30  C_SPARSITY=1 LEARNING_RATE=0.075 TAU_LR=.03 SCALE_TARGET=1.5 SCALE_LOGITS="ADAVAR" MANUAL_GAIN=1 GATE_ARCHITECTURE="[1000]"
# F1000, α=1   ->   acc_bin_train=84.29%        train_acc_diff= 0.93% acc_train=85.22%   acc_bin_TEST=84.40% SCALE_LOGITS="TAU" MANUAL_GAIN=3
# K1000, K=5   ->   acc_bin_train=89.67%        train_acc_diff= 5.05% acc_train=94.72%   acc_bin_TEST=88.02%
# K1000, K=25  ->   acc_bin_train=94.22%        train_acc_diff= 3.81% acc_train=98.03%   acc_bin_TEST=92.53%    --- top 25 is not enough for the input layer!
# L1000        ->   acc_bin_train=97.21% <<<<<  train_acc_diff= 2.36% acc_train=99.56% < acc_bin_TEST=94.66%    --- winner
# L1000        ->   acc_bin_train=97.21% <<<<<  train_acc_diff= 2.36% acc_train=99.56% < acc_bin_TEST=94.66%    --- winner on BIN

# EPOCHS=30  C_SPARSITY=1 LEARNING_RATE=0.075 SCALE_LOGITS="TAU" MANUAL_GAIN=3 GATE_ARCHITECTURE="[1000]"
# F1000, α=1   ->   acc_bin_train=84.29%        train_acc_diff= 0.93% acc_train=85.22%   acc_bin_TEST=84.40% 
# K1000, K=5   ->   acc_bin_train=92.46%        train_acc_diff= 0.54% acc_train=93.01%   acc_bin_TEST=92.18%
# K1000, K=16  ->   acc_bin_train=95.06%        train_acc_diff= 0.57% acc_train=95.64%   acc_bin_TEST=94.23%
# K1000, K=25  ->   acc_bin_train=95.81%        train_acc_diff= 0.44% acc_train=96.25%   acc_bin_TEST=95.15%
# K1000, K=32  ->   acc_bin_train=96.02%        train_acc_diff= 0.56% acc_train=96.59%   acc_bin_TEST=95.49%
# B980,  B=28  ->   acc_bin_train=95.13%        train_acc_diff= 1.83% acc_train=96.96%   acc_bin_TEST=94.48%
# B980,  B=16  ->   acc_bin_train=95.54%        train_acc_diff= 1.23% acc_train=96.77%   acc_bin_TEST=94.45%
# B980,  B=16  ->   acc_bin_train=95.74%        train_acc_diff= 1.26% acc_train=97.00%   acc_bin_TEST=94.65%
# L1000        ->   acc_bin_train=97.33% <<<<<  train_acc_diff= 0.40% acc_train=97.73% < acc_bin_TEST=96.16%    --- winner
# L1000        ->   acc_bin_train=97.60% <<<<<  train_acc_diff= 0.65% acc_train=98.26% < acc_bin_TEST=96.13%    --- winner [MANUAL_GAIN=2.5]
# FFFF1000_1160_1350_1580, α=1
#              ->   acc_bin_train=95.84%        train_acc_diff= 0.72% acc_train=96.56%   acc_bin_TEST=94.86%
# FFFFF1000_1160_1350_1580_1830, α=1
#              ->   acc_bin_train=96.83%        train_acc_diff= 0.45% acc_train=97.28%   acc_bin_TEST=95.09%

# EPOCHS=30  C_SPARSITY=1 LEARNING_RATE=0.075 SCALE_LOGITS="TAU" MANUAL_GAIN=3 GATE_ARCHITECTURE="[1000,1000]"
# LL1000       ->   acc_bin_train=96.77%        train_acc_diff= 1.80% acc_train=98.57% < acc_bin_TEST=96.82%


# TODO: try PyTorch COO sparse tensor functionality and compare the speed
class TopKSparseInterconnect(nn.Module):
    def __init__(self, inputs, outputs, topk, name=''):
        super(TopKSparseInterconnect, self).__init__()
        with torch.no_grad():
            cc = torch.normal(mean=0.0, std=1, size=(inputs, outputs))
            top_indices = torch.zeros((topk, outputs), dtype=torch.long)
        self.top_c = nn.Parameter(torch.zeros((topk, outputs), dtype=torch.float32))
        with torch.no_grad():
            torch.topk(cc, topk, dim=0, largest=True, sorted=False, out=(self.top_c, top_indices))
        self.register_buffer("top_indices", top_indices)
        self.inputs = inputs
        self.name = name
        self.binarized = False
    
    @torch.profiler.record_function("mnist::TopKSparse::FWD")
    def forward(self, x):
        if self.binarized:
            return torch.matmul(x, self.c)

        x = x[:, self.top_indices]
        top_connections = F.softmax(self.top_c * C_SPARSITY, dim=0)
        return (x * top_connections).sum(dim=1)

    def binarize(self, bin_value=1):
        with torch.no_grad():
            outputs = self.top_c.shape[1]
            top1 = torch.argmax(self.top_c, dim=0)
            write_indices = self.top_indices[top1, torch.arange(outputs)]
            c = torch.zeros((self.inputs, outputs), dtype=torch.float32, device=device)
            c.scatter_(dim=0, index=write_indices.unsqueeze(0), value=bin_value)
            self.register_buffer("c", c)
            self.binarized = True

    def __repr__(self):
        percent = self.top_c.shape[0] * 100 / self.inputs
        return f"TopKSparseInterconnect({self.inputs} -> {self.top_c.shape[1] // 2}x2 @ {percent:.2f}%)"

class LearnableGate16Array(nn.Module):
    def __init__(self, number_of_gates, name=''):
        super(LearnableGate16Array, self).__init__()
        self.number_of_gates = number_of_gates
        self.number_of_inputs = number_of_gates * 2
        self.name = name
        self.w = nn.Parameter(torch.zeros((16, self.number_of_gates), dtype=torch.float32)) # [16, W]
        self.zeros = torch.empty(0)
        self.ones = torch.empty(0)
        self.binarized = False
        nn.init.normal_(self.w, mean=0, std=1)
        if G_INIT == "UNIFORM":
            nn.init.uniform_(self.w, a=0.0, b=1.0)
        elif G_INIT == "PASS" or G_INIT == "PASSTHROUGH":
            with torch.no_grad(): self.w[3, :] = 5 # 5 will roughly result in a 95% percent once softmax() applied
        elif G_INIT == "XOR":
            with torch.no_grad(): self.w[6, :] = 5 # 5 will roughly result in a 95% percent once softmax() applied
        else:
            nn.init.normal_(self.w, mean=0.0, std=1)

        # g0  = 
        # g1  =         AB
        # g2  =   A    -AB
        # g3  =   A
        # g4  =      B -AB
        # g5  =      B
        # g6  =   A  B-2AB
        # g7  =   A  B -AB
        # g8  = 1-A -B  AB
        # g9  = 1-A -B 2AB
        # g10 = 1   -B
        # g11 = 1   -B  AB
        # g12 = 1-A
        # g13 = 1-A     AB
        # g14 = 1      -AB
        # g15 = 1

        W = torch.zeros(4, 16, device=device)
        # 1 weights
        W[0, 8:16] =            1
        # A weights
        W[1, [2, 3,  6,  7]] =  1
        W[1, [8, 9, 12, 13]] = -1
        # B weights
        W[2, [4, 5,  6,  7]] =  1
        W[2, [8, 9, 10, 11]] = -1
        # A*B weights
        W[3, 1] =               1
        W[3, 6] =              -2
        W[3, [2, 4, 7,14]] =   -1
        W[3, [1, 8,11,13]] =    1
        W[3, 9] =               2 

        self.W16_to_4 = W


    @torch.profiler.record_function("mnist::LearnableGate16::FWD")
    def forward(self, x):
        # batch_size = x.shape[0]
        # x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        # A = x[:,:,0] # [batch_size, number_of_gates]
        # B = x[:,:,1] # [batch_size, number_of_gates]
        
        # weights_t = F.softmax(self.w * G_SPARSITY, dim=0).transpose(0,1) if not self.binarized else self.w.transpose(0,1)
        # weights = torch.matmul(weights_t, self.W16_to_4.transpose(0,1))  # [number_of_gates, 4]
        # result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=2)
        # return result.sum(dim=2)

        batch_size = x.shape[0]
        x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        A = x[:,:,0]          # [batch_size, number_of_gates]
        B = x[:,:,1]          # [batch_size, number_of_gates]
        
        weights = F.softmax(self.w * G_SPARSITY, dim=0) if not self.binarized else self.w
        weights = torch.matmul(self.W16_to_4, weights) # [4, number_of_gates]
        result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=1)
        return result.sum(dim=1)

    def binarize(self, bin_value=1):
        binarize_inplace(self.w, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"LearnableGate16Array({self.number_of_gates})"

class Model(nn.Module):
    def __init__(self, seed, gate_architecture, interconnect_architecture, number_of_categories, input_size):
        super(Model, self).__init__()
        self.gate_architecture = gate_architecture
        self.interconnect_architecture = interconnect_architecture
        self.first_layer_gates = self.gate_architecture[0]
        self.last_layer_gates = self.gate_architecture[-1]
        self.number_of_categories = number_of_categories
        self.input_size = input_size
        self.seed = seed
        self.adatau = 0
        
        self.outputs_per_category = self.last_layer_gates // self.number_of_categories
        assert self.last_layer_gates == self.number_of_categories * self.outputs_per_category

        layers_ = []
        layer_inputs = input_size
        R = [input_size]
        for layer_idx, (layer_gates, interconnect_params) in enumerate(zip(gate_architecture, interconnect_architecture)):
            if   len(interconnect_params) == 2 and (interconnect_params[0] == "k" or interconnect_params[0] == "top_k"):
                interconnect = TopKSparseInterconnect       (layer_inputs, layer_gates*2, topk=        interconnect_params[1],  name=f"i_{layer_idx}")
            elif len(interconnect_params) == 1 and interconnect_params[0] > 0:
                interconnect = BlockSparseInterconnect      (layer_inputs, layer_gates*2, granularity= interconnect_params[0],  name=f"i_{layer_idx}")
            elif len(interconnect_params) == 1 and interconnect_params[0] < 0:
                interconnect = FixedPowerLawInterconnect    (layer_inputs, layer_gates*2, alpha=      -interconnect_params[0],  name=f"i_{layer_idx}")
            else:
                interconnect = SparseInterconnect           (layer_inputs, layer_gates*2,                                       name=f"i_{layer_idx}")
            layers_.append(interconnect)
            layers_.append(LearnableGate16Array(layer_gates, f"g_{layer_idx}"))
            layer_inputs = layer_gates
            R.append(layer_gates)
            if PASS_INPUT_TO_ALL_LAYERS:
                layer_inputs += input_size
            if PASS_RESIDUAL and (layer_idx > 0 or not PASS_INPUT_TO_ALL_LAYERS):
                layer_inputs += R[-2]
        self.layers = nn.ModuleList(layers_)

        if DROPOUT > 0.0001:
            self.dropout = Dropout01(p=DROPOUT)
        elif DROPOUT < -0.0001:
            self.dropout_last = Dropout01(p=-DROPOUT)

    @torch.profiler.record_function("mnist::Model::FWD")
    def forward(self, X):
        with torch.no_grad():
            S = torch.sum(X, dim=-1)
            self.log_input_mean = torch.mean(S).item()
            self.log_input_std = torch.std(S).item()
            self.log_input_norm = torch.norm(S).item()
        I = X
        R = [I]
        for layer_idx in range(0, len(self.layers)):
            X = self.layers[layer_idx](X)
            if type(self.layers[layer_idx]) is LearnableGate16Array:
                R.append(X)
                # TODO: fix unreadable logic with layer_idx
                # NOTE: ugly layer_idx > 1 which differ from layer_idx > 0 in the Model constructor, but has the same meaning
                if PASS_INPUT_TO_ALL_LAYERS and layer_idx < len(self.layers)-2:
                    X = torch.cat([X, I], dim=-1)
                if PASS_RESIDUAL and (layer_idx > 1 or not PASS_INPUT_TO_ALL_LAYERS) and layer_idx < len(self.layers)-2:
                    X = torch.cat([X, R[-2]], dim=-1)
                if hasattr(self, 'dropout'):
                    X = self.dropout(X)

        if hasattr(self, 'dropout_last'):
            X = self.dropout_last(X)
        X = X.view(X.size(0), self.number_of_categories, self.outputs_per_category).sum(dim=-1)
        if not self.training:   # INFERENCE ends here! Everything past this line will only concern training
            return X            # Finishing inference here is both:
                                # 1) an OPTIMISATION and
                                # 2) it ensures no discrepancy between VALIDATION step during training vs STANDALONE inference

        gain = MANUAL_GAIN
        with torch.no_grad():
            self.log_applied_gain = gain
            self.log_pregain_mean = torch.mean(X).item()
            self.log_pregain_std = torch.std(X).item()
            self.log_pregain_min = torch.min(X).item()
            self.log_pregain_max = torch.max(X).item()
            self.log_pregain_norm = torch.norm(X).item()
        X = X / gain

        if SCALE_LOGITS == "TANH":
            mu = torch.mean(X, dim=0)
            X = SCALE_TARGET * torch.tanh((X - mu)/SCALE_TARGET)
        elif SCALE_LOGITS == "ARCSINH":
            mu = torch.mean(X, dim=0)
            k = 2
            X = SCALE_TARGET * torch.arcsinh((X - mu)/SCALE_TARGET*k) / np.arcsinh(k)
        elif SCALE_LOGITS == "MINMAX":
            rng = torch.max(X).item() - torch.min(X).item()
            f = self.outputs_per_category / rng
            X = X / f / SCALE_TARGET
            self.log_applied_gain = f * SCALE_TARGET
        elif SCALE_LOGITS == "ADATAU": # TAU_LR=0.001 SCALE_TARGET 0.75 -> 96.6 LLLLx1000
            rng = torch.max(X).item() - torch.min(X).item()
            t = rng / self.outputs_per_category
            if self.adatau < 0.1:
                self.adatau = 1 / t
            elif t < SCALE_TARGET:
                self.adatau *= 1.0+TAU_LR
            else:
                self.adatau *= 1.0-TAU_LR*100 # faster falloff
            self.adatau = min(self.adatau, 100)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
        elif SCALE_LOGITS == "ADATAU2":
            rng = torch.max(X).item() - torch.min(X).item()
            t = rng / self.outputs_per_category
            if self.adatau < 0.1:
                self.adatau = 1 / t
            else:
                self.adatau = self.adatau * (1-TAU_LR) + (1 / t) * TAU_LR
            self.adatau = min(self.adatau, 100)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
        elif SCALE_LOGITS == "ADATAU4": # SCALE_TARGET=.7 TAU_LR=.001
            rng = torch.max(X).item() - torch.min(X).item()
            t = rng / self.outputs_per_category
            self.adatau += (SCALE_TARGET-t)*TAU_LR
            self.adatau = max(self.adatau, .1)
            self.adatau = min(self.adatau, 100)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
        elif SCALE_LOGITS == "ADAVAR0":
            # std = torch.std(X, dim=0)
            std = torch.std(X).item()
            t = std / (4.0 * SCALE_TARGET)
            if self.adatau < 0.1:
                self.adatau = t
            else:
                self.adatau = self.adatau * (1-TAU_LR) + t * TAU_LR
            self.adatau = min(self.adatau, self.outputs_per_category / 6)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
        elif SCALE_LOGITS == "ADAVAR":      # SCALE_TARGET=1.5 TAU_LR=.03 C_SPARSITY=1 ->   99.96/97.57% Lx8000 (30epochs)
                                            # SCALE_TARGET=1.5 TAU_LR=.03 C_SPARSITY=1 ->   99.99/97.73% LFFx8000 (30epochs)
                                            # SCALE_TARGET=1.5 TAU_LR=.03 C_SPARSITY=1 ->   99.86/96.79% LFFFFFx8000 (30epochs)
                                            # SCALE_TARGET=1   TAU_LR=.03 C_SPARSITY=1 ->   99.91/97.47% LFFFFFx8000 (30epochs)
                                            # SCALE_TARGET=1.5 TAU_LR=.03 C_SPARSITY=1 ->   99.48/96.73% Lx2000 (30epochs)
                                            # SCALE_TARGET=1   TAU_LR=.03 C_SPARSITY=1 ->   98.63/97.04% Lx2000 (30epochs)
                                            # SCALE_TARGET=1   TAU_LR=.01              ->   97.62/96.32% LLLLx1000 (50 epochs)
            std = torch.std(X).item()
            t = std / (4.0 * SCALE_TARGET)
            if self.adatau < 0.1:
                self.adatau = np.sqrt(self.outputs_per_category)
            else:
                self.adatau = self.adatau * (1-TAU_LR) + t * TAU_LR
            self.adatau = min(self.adatau, self.outputs_per_category / 6)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
        elif SCALE_LOGITS == "ADAVAR_TANH": # SCALE_TARGET=1 TAU_LR=.01               ->    98.71/97.44% Lx8000 (20epochs)
            std = torch.std(X).item()
            t = std / (4.0 * SCALE_TARGET)
            if self.adatau < 0.1:
                self.adatau = np.sqrt(self.outputs_per_category)
            else:
                self.adatau = self.adatau * (1-TAU_LR) + t * TAU_LR
            self.adatau = min(self.adatau, self.outputs_per_category / 6)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
            mu = torch.mean(X, dim=0)
            std = torch.std(X).item()
            X = (3.0 * std) * torch.tanh((X - mu)/(3.0 * std))
        elif SCALE_LOGITS == "ADAVAR_ARCSINH" or \
             SCALE_LOGITS == "ADAVAR_ASH":  # SCALE_TARGET=1 TAU_LR=.01 ->                  99.53/97.58% Lx8000 (20epochs)
                                            #                                               98.09/96.28% LLLLx1000 (50 epochs)
                                            #                                               98.72/96.56% LLLx1000+Lx8000 (20 epochs)
            std = torch.std(X).item()
            t = std / (4.0 * SCALE_TARGET)
            if self.adatau < 0.1:
                self.adatau = np.sqrt(self.outputs_per_category)
            else:
                self.adatau = self.adatau * (1-TAU_LR) + t * TAU_LR
            self.adatau = min(self.adatau, self.outputs_per_category / 6)
            X = X / self.adatau
            self.log_applied_gain = self.adatau
            k = 2
            mu = torch.mean(X, dim=0)
            std = torch.std(X).item()
            X = (2.0 * std) * torch.arcsinh((X - mu)/(2.0 * std)*k) / np.arcsinh(k)
        elif SCALE_LOGITS == "AUTOTAU":
            tau = np.sqrt(self.outputs_per_category / (4.0 * SCALE_TARGET))
            X = X / tau
            self.log_applied_gain = tau
        elif SCALE_LOGITS == "AUTOTAU_TANH":
            tau = np.sqrt(self.outputs_per_category / (6.0 * SCALE_TARGET))
            X = X / tau
            self.log_applied_gain = tau
            mu = torch.mean(X, dim=0)
            X = tau * torch.tanh((X - mu)/tau)
        elif SCALE_LOGITS == "AUTOTAU_ARCSINH" or \
             SCALE_LOGITS == "AUTOTAU_ASH":         # SCALE_TARGET=1 TAU_LR=.01 ->          ??.??/(?)97.55% Lx8000 (20epochs)
                                                    #                                       ??.??/(?)95.93% LLLLx1000 (50 epochs)
            tau = np.sqrt(self.outputs_per_category / (6.0 * SCALE_TARGET))
            X = X / tau
            self.log_applied_gain = tau
            k = 2
            mu = torch.mean(X, dim=0)
            std = torch.std(X).item()
            X = (2.0 * std) * torch.arcsinh((X - mu)/(2.0 * std)*k) / np.arcsinh(k)
            # X = tau * torch.arcsinh((X - mu)/tau*k) / np.arcsinh(k)


        with torch.no_grad():
            self.log_logits_mean = torch.mean(X).item()
            self.log_logits_std = torch.std(X).item()
            self.log_logits_norm = torch.norm(X).item()

        if not NO_SOFTMAX:
            X = F.softmax(X, dim=-1)
        return X

    def clone_and_binarize(self, device, bin_value=1):
        model_binarized = Model(self.seed, self.gate_architecture, self.interconnect_architecture, self.number_of_categories, self.input_size).to(device)
        model_binarized.load_state_dict(self.state_dict())
        for layer in model_binarized.layers:
            if hasattr(layer, 'binarize') and callable(layer.binarize):
                layer.binarize(bin_value)
        return model_binarized

    def get_passthrough_fraction(self):
        pass_fraction_array = []
        indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                pass_fraction_array.append(pass_weight / total_weight)
        return pass_fraction_array
    
    def get_unique_fraction(self):
        unique_fraction_array = []
        for model_layer in self.layers:
            # TODO: better to measure only unique indices that point to the previous layer and ignore skip connections:
            # ... = sum(unique_indices < previous_layer.c.shape[1])
            if hasattr(model_layer, 'c'):
                unique_indices = torch.unique(torch.argmax(model_layer.c, dim=0)).numel()
                unique_fraction_array.append(unique_indices / model_layer.c.shape[0])
            elif hasattr(model_layer, 'top_c') and hasattr(model_layer, 'top_indices'):
                outputs = model_layer.top_c.shape[1]
                top1 = torch.argmax(model_layer.top_c, dim=0)
                indices = model_layer.top_indices[top1, torch.arange(outputs)]
                max_inputs = indices.max().item() # approximation
                unique_indices = torch.unique(indices).numel()
                unique_fraction_array.append(unique_indices / max_inputs)
                # unique_fraction_array.append(unique_indices / 100)
            elif hasattr(model_layer, 'indices'):
                max_inputs = model_layer.indices.max().item() # approximation
                unique_indices = torch.unique(model_layer.indices).numel()
                unique_fraction_array.append(unique_indices / max_inputs)
        return unique_fraction_array

    def compute_selected_gates_fraction(self, selected_gates):
        gate_fraction_array = []
        indices = torch.tensor(selected_gates, dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                gate_fraction_array.append(pass_weight / total_weight)
        return torch.mean(torch.tensor(gate_fraction_array)).item()


### INSTANTIATE THE MODEL AND MOVE TO GPU ###

log(f"PREPARE MODEL on device={device}")
random.seed(SEED)
torch.manual_seed(SEED)
model = Model(SEED, GATE_ARCHITECTURE, INTERCONNECT_ARCHITECTURE, NUMBER_OF_CATEGORIES, INPUT_SIZE).to(device)
if COMPILE_MODEL:
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
log(f"model={model}")

############################ DATA ########################

### GENERATORS
def transform():    
    def binarize_image_with_histogram(image):
        threshold = torch.quantile(image, BINARIZE_IMAGE_TRESHOLD)
        return (image > threshold).float()

    return transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
        transforms.Lambda(lambda x: binarize_image_with_histogram(x))
    ])

import inspect
import hashlib
import re
    
def get_code_hash(fn, verbose=False) -> str:
    def replace_globals_in_string(s: str) -> str:
        for name, value in globals().items():
            if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismodule(value):
                continue
            try:
                pattern = re.escape(name)
                replacement = str(value)
                s = re.sub(rf'\b{pattern}\b', replacement, s)
            except Exception:
                continue
        return s
    src = inspect.getsource(fn)
    src = replace_globals_in_string(src)
    if verbose:
        print(src)
    return hashlib.sha1(src.encode('utf-8')).hexdigest()[:10]

hash_transform_fn = get_code_hash(transform)#, verbose=True)
log(f"READ DATA")

def load_or_build_cached_mnist(root, train, transform, hash):
    split = 'train' if train else 'test'
    path = f'{root}/cached_mnist_{split}_{hash}.pt' #  get_cache_path(config, split)
    try:
        cached = torch.load(path)
        print(f"Loaded cached MNIST {split} from {path}")
        return torch.utils.data.TensorDataset(cached['data'], cached['labels'])
    except FileNotFoundError:
        print(f"No cache found for hash {hash}. Building new cache at {path}...")
        # dataset = CIFAR10(root='./data', train=(split == 'train'), download=True, transform=transform)
        dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            download=True
        )

        inputs, labels = [], []
        # for x, y in tqdm(dataset, desc=f"Preprocessing MNIST {split}"):
        for x, y in dataset:
            inputs.append(x.unsqueeze(0))
            labels.append(torch.tensor(y).unsqueeze(0))

        data_tensor = torch.cat(inputs, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        torch.save({'data': data_tensor, 'labels': label_tensor, 'hash': hash}, path)
        return torch.utils.data.TensorDataset(data_tensor, label_tensor)

# train_dataset = torchvision.datasets.MNIST(
train_dataset = load_or_build_cached_mnist(
    root="./data",
    train=True,
    transform=transform(),
    hash=hash_transform_fn
    # download=True
)

# test_dataset = torchvision.datasets.MNIST(
test_dataset = load_or_build_cached_mnist(
    root="./data",
    train=False,
    transform=transform(),
    hash=hash_transform_fn
    # download=True
)

### SPLIT TRAIN DATASET ###

log(f"UPLOAD DATA")
train_size = int(TRAIN_FRACTION * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(DATA_SPLIT_SEED))

if ONLY_USE_DATA_SUBSET:
    train_dataset = torch.utils.data.Subset(train_dataset, range(1024))
    val_dataset = torch.utils.data.Subset(val_dataset, range(1024))

### MOVE TRAIN DATASET TO GPU ###

train_dataset_samples = len(train_dataset)
train_images = torch.empty((train_dataset_samples, INPUT_SIZE), dtype=torch.float32)
train_labels = torch.empty((train_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

train_labels_ = torch.empty((train_dataset_samples), dtype=torch.long)
for i, (image, label) in enumerate(train_dataset):
    train_images[i] = image
    train_labels_[i] = label
train_labels = torch.nn.functional.one_hot(train_labels_, num_classes=NUMBER_OF_CATEGORIES)
train_labels = train_labels.type(torch.float32)

train_images = train_images.to(device)
train_labels = train_labels.to(device)

### MOVE VAL DATASET TO GPU ###

val_dataset_samples = len(val_dataset)

val_images = torch.empty((val_dataset_samples, INPUT_SIZE), dtype=torch.float32)
val_labels = torch.empty((val_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

val_labels_ = torch.empty((val_dataset_samples), dtype=torch.long)
for i, (image, label) in enumerate(val_dataset):
    val_images[i] = image
    val_labels_[i] = label
val_labels = torch.nn.functional.one_hot(val_labels_, num_classes=NUMBER_OF_CATEGORIES)
val_labels = val_labels.type(torch.float32)

val_images = val_images.to(device)
val_labels = val_labels.to(device)

### MOVE TEST DATASET TO GPU ###

test_dataset_samples = len(test_dataset)

test_images = torch.empty((test_dataset_samples, INPUT_SIZE), dtype=torch.float32)
test_labels = torch.empty((test_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

test_labels_ = torch.empty((test_dataset_samples), dtype=torch.long)
for i, (image, label) in enumerate(test_dataset):
    test_images[i] = image
    test_labels_[i] = label
test_labels = torch.nn.functional.one_hot(test_labels_, num_classes=NUMBER_OF_CATEGORIES)
test_labels = test_labels.type(torch.float32)

test_images = test_images.to(device)
test_labels = test_labels.to(device)

### VALIDATE ###

def get_validate(default_model):
    def validate(dataset="val", model=default_model):
        if dataset == "val":
            number_of_samples = val_dataset_samples
            sample_images = val_images
            sample_labels = val_labels
        elif dataset == "test":
            number_of_samples = test_dataset_samples
            sample_images = test_images
            sample_labels = test_labels
        elif dataset == "train":
            number_of_samples = train_dataset_samples
            sample_images = train_images
            sample_labels = train_labels
        else:
            raise IOError(f"Unknown dataset {dataset}")
        val_loss = 0.0
        val_steps = 0
        correct = 0
        for start_idx in range(0, number_of_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, number_of_samples)
            val_indices = torch.arange(start_idx, end_idx, device=device)    
            x_val = sample_images[val_indices]
            y_val = sample_labels[val_indices]
            with torch.no_grad():
                val_output = model(x_val)
                val_loss += F.cross_entropy(val_output, y_val, reduction="sum").item()
                correct += (val_output.argmax(dim=1) == y_val.argmax(dim=1)).sum().item()
            val_steps += len(x_val)
        val_loss /= val_steps
        val_accuracy = correct / val_steps
        return val_loss, val_accuracy
    return validate

def get_binarized_model(model=None, bin_value=1):
    return model.clone_and_binarize(device, bin_value)

def l1_topk(weights_after_softmax, k=4, special_dim=0): # but goes to 1 when binarized; 0 when uniform
    # test
    # t1 = torch.zeros([8,32,75]); l1_topk(F.softmax(t1,dim=1),special_dim=1) # should get zero
    # t1 = torch.rand(8, 32, 75); l1_topk(F.softmax(t1,dim=1),special_dim=1) # should get almost zero
    # ones_at = torch.argmax(t1, dim=1); t1.zero_(); t1.scatter_(dim=1, index=ones_at.unsqueeze(1), value=100); l1_topk(F.softmax(t1,dim=1),special_dim=1) # should get 1
    other_dims_prod = torch.prod(torch.tensor([x for i, x in enumerate(weights_after_softmax.shape) if i != special_dim]))
    normalization_factor = (weights_after_softmax.shape[special_dim]-k)/weights_after_softmax.shape[special_dim] * other_dims_prod # in case of a uniform tensor
    top_k_values, _ = torch.topk(weights_after_softmax, k, dim=special_dim)
    top_k_sum = top_k_values.sum(dim=special_dim, keepdim=True)
    non_top_k_sum = (1 - top_k_sum).sum()
    return 1. - non_top_k_sum / normalization_factor

### TRAIN ###

validate = get_validate(model)
val_loss, val_accuracy = validate(dataset="val")
passthrough_log = ", ".join([f"{value * 100:4.1f}%" for value in model.get_passthrough_fraction()])
unique_log = ", ".join([f"{value * 100:4.1f}%" for value in model.get_unique_fraction()])
log(f"INIT VAL loss={val_loss:6.3f} acc={val_accuracy*100:6.2f}%                 - Pass {passthrough_log} | Connectivity {unique_log}")
WANDB_KEY and wandb.log({"init_val": val_accuracy*100})

log(f"EPOCH_STEPS={EPOCH_STEPS}, will train for {EPOCHS} EPOCHS")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0) # if weight decay encourages uniform distribution
time_start = time.time()

if PROFILE: prof.start()
for i in range(TRAINING_STEPS):
    indices = torch.randint(0, train_dataset_samples, (BATCH_SIZE,), device=device)
    x = train_images[indices]
    y = train_labels[indices]
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        for l in model.layers:
            if hasattr(l,'w'):
                if SUPPRESS_CONST:
                    for const_gate_ix in [0,15]:
                        l.w.data[const_gate_ix, :] = 0
                if SUPPRESS_PASSTHROUGH:
                    for pass_gate_ix in [3, 5, 10, 12]:
                        l.w.data[pass_gate_ix, :] = 0

        model_output = model(x)
        loss_ce = F.cross_entropy(model_output, y)
        
        tension_loss = 0
        if TENSION_REGULARIZATION > 0:
            for model_layer in model.layers:
                if hasattr(model_layer, 'c'):
                    conn_weights_after_softmax = F.softmax(model_layer.c, dim=0)
                    tension_loss += torch.sum((1 - conn_weights_after_softmax) * conn_weights_after_softmax)
                if hasattr(model_layer, 'c_sub_layer_1') and hasattr(model_layer, 'c_sub_layer_2'):
                    conn_1 = F.softmax(model_layer.c_sub_layer_1, dim=1)
                    conn_2 = F.softmax(model_layer.c_sub_layer_2, dim=1)
                    tension_loss += torch.sum((1 - conn_1) * conn_1)
                    tension_loss += torch.sum((1 - conn_2) * conn_2)
                if hasattr(model_layer, 'w'):
                    gate_weights_after_softmax = F.softmax(model_layer.w, dim=0)
                    tension_loss += torch.sum((1 - gate_weights_after_softmax) * gate_weights_after_softmax)
        regularization_loss = tension_loss * TENSION_REGULARIZATION * (float(i) / float(TRAINING_STEPS))

        loss = loss_ce + regularization_loss
        loss.backward()
        optimizer.step()

    # TODO: model.eval here perhaps speeds everything up?
    if (i + 1) % PRINTOUT_EVERY == 0:
        passthrough_log = " ".join([f"{value * 100:2.0f}" for value in model.get_passthrough_fraction()])
        unique_log = " ".join([f"{value * 100:2.0f}" for value in model.get_unique_fraction()])
        # grad_log = ", ".join([f"{value.grad.std():.0e}" for value in model.parameters()])
        # grad_log = ", ".join([f"{value.grad.std():2.3e}" for value in model.parameters()])
        grad_log = " ".join([f"{torch.log10(value.grad.std())*10:2.0f}" for value in model.parameters()])
        # inputs_log = f"μ{model.log_input_mean:.2f}±{model.log_input_std:.2f}"
        # self.log_input_norm = torch.norm(X).item()
        # logits_log = f"μ{model.log_pregain_mean:.0f}±{model.log_pregain_std:.1f}*{1.0/model.log_applied_gain:0.3f} v{model.log_pregain_min:.0f}^{model.log_pregain_max:.0f}= μ{model.log_logits_mean:.0f}±{model.log_logits_std:.1f} ‖{model.log_logits_norm:.0f}‖"
        logits_log = f"{model.log_pregain_min:.0f}..{model.log_pregain_max:.0f} μ{model.log_pregain_mean:.0f}±{model.log_pregain_std:.1f}/{model.log_applied_gain:.1f} = ±{model.log_logits_std:.1f} ‖{model.log_logits_norm:.0f}‖"
        log(f"Iteration {i + 1:10} - Loss {loss:6.3f} - RegLoss {(1-loss_ce/loss)*100:2.0f}% - Pass (%) {passthrough_log} | Conn (%) {unique_log} | e-∇x10 {grad_log} | Logits {logits_log}")
        WANDB_KEY and wandb.log({"training_step": i, "loss": loss, 
            "regularization_loss_fraction":(1-loss_ce/loss)*100, 
            "tension_loss":tension_loss, })
    if (i + 1) % VALIDATE_EVERY == 0 or (i + 1) == TRAINING_STEPS:
        current_epoch = (i+1) // EPOCH_STEPS

        train_loss, train_acc = validate('train')
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     TRN loss={train_loss:.3f} acc={train_acc*100:.2f}%")
        model_binarized = get_binarized_model(model)
        _, bin_train_acc = validate(dataset="train", model=model_binarized)
        train_acc_diff = train_acc-bin_train_acc
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS} BIN TRN            acc={bin_train_acc*100:.2f}%, train_acc_diff={train_acc_diff*100:.2f}%")

        test_loss, test_acc = validate('test')
        test_acc_diff = train_acc-test_acc
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     TST            acc={test_acc*100:.2f}%, test_acc_diff= {test_acc_diff*100:.2f}%, loss={test_loss:.3f}")
        
        top1w = torch.tensor(0., device=device)
        top2w = torch.tensor(0., device=device)
        top4w = torch.tensor(0., device=device)
        top8w = torch.tensor(0., device=device)
        top1c = torch.tensor(0., device=device)
        top2c = torch.tensor(0., device=device)
        top4c = torch.tensor(0., device=device)
        top8c = torch.tensor(0., device=device)
        for model_layer in model.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                top1w += l1_topk(weights_after_softmax,k=1)
                top2w += l1_topk(weights_after_softmax,k=2)
                top4w += l1_topk(weights_after_softmax,k=4)
                top8w += l1_topk(weights_after_softmax,k=8)
            if hasattr(model_layer, 'c'):
                weights_after_softmax = F.softmax(model_layer.c, dim=1)
                top1c += l1_topk(weights_after_softmax,k=1,special_dim=1)
                top2c += l1_topk(weights_after_softmax,k=2,special_dim=1)
                top4c += l1_topk(weights_after_softmax,k=4,special_dim=1)
                top8c += l1_topk(weights_after_softmax,k=8,special_dim=1)
            if hasattr(model_layer, 'c_sub_layer_1') and hasattr(model_layer, 'c_sub_layer_2'):
                conn_1 = F.softmax(model_layer.c_sub_layer_1, dim=1)
                conn_2 = F.softmax(model_layer.c_sub_layer_2, dim=1)
                top1c += 0.5*l1_topk(conn_1,k=1,special_dim=1)
                top2c += 0.5*l1_topk(conn_1,k=2,special_dim=1)
                top4c += 0.5*l1_topk(conn_1,k=4,special_dim=1)
                top8c += 0.5*l1_topk(conn_1,k=8,special_dim=1)
                top1c += 0.5*l1_topk(conn_2,k=1,special_dim=1)
                top2c += 0.5*l1_topk(conn_2,k=2,special_dim=1)
                top4c += 0.5*l1_topk(conn_2,k=4,special_dim=1)
                top8c += 0.5*l1_topk(conn_2,k=8,special_dim=1)
        top1w /= len(model.layers)
        top2w /= len(model.layers)
        top4w /= len(model.layers)
        top8w /= len(model.layers)
        top1c /= len(model.layers)
        top2c /= len(model.layers)
        top4c /= len(model.layers)
        top8c /= len(model.layers)

        WANDB_KEY and wandb.log({"epoch": current_epoch, 
            "train_loss": train_loss, "train_acc": train_acc*100,
            # "val_loss": val_loss, "val_acc": val_acc*100,
            "bin_train_acc": bin_train_acc*100, "train_acc_diff": train_acc*100-bin_train_acc*100,
            # "bin_val_acc": bin_val_acc*100, "val_acc_diff": val_acc*100-bin_val_acc*100,
             "top1w":top1w, "top2w":top2w, "top4w":top4w, "top8w":top8w,
             "top1c":top1c, "top2c":top2c, "top4c":top4c, "top8c":top8c,
             "gate_perc_pass": model.compute_selected_gates_fraction([3, 5, 10, 12])*100.,
             "gate_perc_const": model.compute_selected_gates_fraction([0, 15])*100.,
             "gate_perc_and": model.compute_selected_gates_fraction([1, 14])*100.,
             "gate_perc_aImplies": model.compute_selected_gates_fraction([2, 13])*100.,
             "gate_perc_bImplies": model.compute_selected_gates_fraction([4, 11])*100.,
             "gate_perc_xor": model.compute_selected_gates_fraction([6, 9])*100.,
             "gate_perc_or": model.compute_selected_gates_fraction([7, 8])*100.,
            })
    if PROFILE:
        torch.cpu.synchronize()
        if   device.type == "cuda": torch.cuda.synchronize()
        elif device.type == "mps":  torch.mps.synchronize()
    if PROFILE: prof.step()
if PROFILE: prof.stop()


time_end = time.time()
training_total_time = time_end - time_start 
log(f"Training took {training_total_time:.2f} seconds, per iteration: {(training_total_time) / TRAINING_STEPS * 1000:.2f} milliseconds")

test_loss, test_acc = validate('test')
log(f"    TEST loss={test_loss:.3f} acc={test_acc*100:.2f}%")




model_binarized = get_binarized_model(model)
bin_test_loss, bin_test_acc = validate(dataset="test", model=model_binarized)
log(f"BIN TEST loss={bin_test_loss:.3f} acc={bin_test_acc*100:.2f}%")

model_filename = (
    f"{datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y%m%d-%H%M%S')}"
    f"_binTestAcc{round(bin_test_acc * 10000)}"
    f"_seed{SEED}_epochs{EPOCHS}_{len(GATE_ARCHITECTURE)}x{GATE_ARCHITECTURE[0]}"
    f"_b{BATCH_SIZE}_lr{LEARNING_RATE * 1000:.0f}"
    f"_interconnect.pth"
)
torch.save(model.state_dict(), model_filename) #!!!
log(f"Saved to {model_filename}")

WANDB_KEY and wandb.log({
            "final_train_loss": train_loss, "final_train_acc": train_acc*100,
            # "final_val_loss": val_loss, "final_val_acc": val_acc*100,
            "final_test_loss": test_loss, "final_test_acc": test_acc*100,
            "final_bin_test_loss": bin_test_loss, "final_bin_test_acc": bin_test_acc*100,
            "training_total_time": training_total_time,
    })

from telegram import Bot
import asyncio
(TG_TOKEN and TG_CHATID) and asyncio.run(Bot(token=TG_TOKEN).send_message(chat_id=int(TG_CHATID), text=LOG_NAME))

WANDB_KEY and wandb.finish()

if PROFILE:
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=PROFILER_ROWS))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=PROFILER_ROWS))
    if device.type == "cuda":
        print("-"*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=PROFILER_ROWS))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=PROFILER_ROWS))
    prof.export_chrome_trace(f"{LOG_NAME}_profile.json")
