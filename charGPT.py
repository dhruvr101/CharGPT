import math, random, argparse, json, time, sys, os, io
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class Cfg:  

    block_size: int = 64
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    eval_interval: int = 500
    precision: str = "fp32"           


    temperature: float = 1.0
    top_k: int = 50
    max_tokens: int = 256


    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    out_dir: Path = Path("checkpoints")
    dataset_path: Path = Path("tiny_shakespeare.txt")
    vocab_size: int = -1  

cfg = Cfg()
torch.manual_seed(cfg.seed)

def build_vocab(text: str):
    chars = list(sorted(set(text.encode('utf-8'))))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    cfg.vocab_size = len(chars)
    return stoi, itos

def encode(text: str, stoi: dict):
    return [stoi[b] for b in text.encode('utf-8')]
