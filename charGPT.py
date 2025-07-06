
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

def decode(tokens: List[int], itos: dict):
    return bytes([itos[t] for t in tokens]).decode('utf-8', errors='ignore')

class CharDataset(Dataset):
    def __init__(self, data: List[int]):
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return self.data.size(0) - cfg.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + cfg.block_size]
        y = self.data[idx + 1 : idx + 1 + cfg.block_size]
        return x, y
# ────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2)
        div = torch.exp(two_i * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, cfg.d_model * 3, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.proj_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, cfg.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:T, :T] == 0, -float("inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.out_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention()
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.block_size)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)  # B T C
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        model = self
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / cfg.temperature
            if cfg.top_k:
                v, _ = torch.topk(logits, cfg.top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def save_checkpoint(model, optim, step):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "cfg": vars(cfg),
        "step": step,
    }
    torch.save(ckpt, cfg.out_dir / "latest.pt")

def load_checkpoint(model, optim):
    path = cfg.out_dir / "latest.pt"
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    if "cfg" in ckpt:
        vars(cfg).update(ckpt["cfg"])
    return ckpt.get("step", 0)

def get_lr(step):
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup_steps) / (cfg.epochs * len_train - cfg.warmup_steps)))


def prepare_data():
    if not cfg.dataset_path.exists():
        # Minimal fallback dataset (fits in file)
        cfg.dataset_path.write_text(
            "Shall I compare thee to a summer's day?\n"
            "Thou art more lovely and more temperate:\n"
            "Rough winds do shake the darling buds of May,\n"
            "And summer's lease hath all too short a date:\n"
        )
    text = cfg.dataset_path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    data_ids = encode(text, stoi)
    split = int(0.9 * len(data_ids))
    train_ids, val_ids = data_ids[:split], data_ids[split:]
    return CharDataset(train_ids), CharDataset(val_ids), itos, stoi

def evaluate(model, val_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            _, loss = model(x, y)
            losses.append(loss.item())
            if len(losses) > 100:  # cap the time spent
                break
    model.train()
    return sum(losses) / len(losses)

def train():
    global len_train
    train_ds, val_ds, itos, _ = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    len_train = cfg.epochs * len(train_loader)
    model = TransformerLM().to(cfg.device)

    scaler = GradScaler(enabled=cfg.precision == "amp")
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    start_step = load_checkpoint(model, optim)
    sched = CosineAnnealingLR(optim, T_max=len_train, eta_min=1e-5)
    train_iter = cycle(train_loader)

    pbar = tqdm(range(start_step, len_train), initial=start_step, total=len_train, desc="steps")
    for step in pbar:
        x, y = next(train_iter)
        x, y = x.to(cfg.device), y.to(cfg.device)

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.precision == "amp"):
            logits, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optim)
        scaler.update()
        sched.step()

        if step % cfg.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_loader)
            pbar.set_postfix(train=f"{loss.item():.3f}", val=f"{val_loss:.3f}")
            save_checkpoint(model, optim, step)

    torch.save({"model": model.state_dict(), "itos": itos}, cfg.out_dir / "final.pt")

def generate_text(prompt: str):
    ckpt = torch.load(cfg.out_dir / "final.pt", map_location=cfg.device)
    itos = ckpt["itos"]; stoi = {c: i for i, c in enumerate(itos)}
    model = TransformerLM().to(cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = torch.tensor([[stoi.get(b, 0) for b in prompt.encode('utf-8')]], dtype=torch.long).to(cfg.device)
    idx = model.generate(idx, cfg.max_tokens)
    print(decode(idx[0].tolist(), itos))

def parse_cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=cfg.epochs)
    t.add_argument("--precision", choices=["fp32", "amp", "bf16"], default=cfg.precision)
    g = sub.add_parser("generate")
    g.add_argument("--prompt", type=str, default="ROMEO:")
    g.add_argument("--tokens", type=int, default=cfg.max_tokens)
    args = p.parse_args()
    cfg.epochs = getattr(args, "epochs", cfg.epochs)
    cfg.precision = getattr(args, "precision", cfg.precision)
    cfg.max_tokens = getattr(args, "tokens", cfg.max_tokens)
    return args

# entryp lmao
if __name__ == "__main__":
    args = parse_cli()
    if args.cmd == "train":
        train()
    elif args.cmd == "generate":
        generate_text(args.prompt)
