"""
ddpm_utils.py
=============
Shared components for IFN680 Project 7 - "Improving DDPM" (linear vs. cosine
beta schedule).

Imported by both:
  - DDPM_CosineSchedule.ipynb  (dataset setup, model definition, training)
  - main_report.ipynb          (loads checkpoints, runs inference / evaluation)

Keeping every schedule-, model- and sampling-related function here (instead
of duplicating code in both notebooks) avoids the two notebooks drifting out
of sync and keeps the "linear" and "cosine" runs strictly comparable, since
they always share the exact same U-Net backbone, forward process and
sampler implementation.

Architecture follows the 9.3 Tutorial's conditional U-Net as closely as
possible; the only structural change is that mnist_custom.pt stores 20x20
images (not 28x28 like torchvision's MNIST), which the model below accepts
via the `image_size` / `channels` arguments.
"""

import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import utils as vutils
from tqdm.auto import tqdm  # single updating bar in Jupyter, plain bar in terminals
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MNISTCustomDataset(Dataset):
    """Wraps the tensors stored in mnist_custom.pt.

    The raw tensors are float32 pixel values in [0, 1]. DDPM's forward
    process assumes data centred at 0, so we rescale to [-1, 1] exactly as
    the 9.3 Tutorial does for torchvision MNIST.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.float() * 2.0 - 1.0
        self.labels = labels.long()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_mnist_custom(path: str):
    """Loads mnist_custom.pt and returns (train_dataset, test_dataset).

    The file is a dict with keys 'train_images', 'train_labels',
    'test_images', 'test_labels' (images: [N, 1, 20, 20] float32 in [0, 1],
    labels: [N] int64 in {0, ..., 9}).
    """
    data = torch.load(path, map_location="cpu")
    train_ds = MNISTCustomDataset(data["train_images"], data["train_labels"])
    test_ds = MNISTCustomDataset(data["test_images"], data["test_labels"])
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear variance schedule, as defined in the 9.3 Tutorial."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine variance schedule.

    Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
    Models", ICML 2021 (Eq. 17). Unlike the linear schedule, alpha_bar(t)
    follows a squared-cosine curve, which removes the abrupt information
    destruction the linear schedule causes near t=T and instead spreads the
    noise increase more evenly across the whole trajectory.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alpha_bars = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, min=1e-4, max=0.999)


class NoiseSchedule:
    """Bundles betas / alphas / alpha_bars for one schedule.

    Using an explicit object (instead of module-level globals, as in the
    9.3 Tutorial) lets us keep the linear and cosine schedules alive at the
    same time without either one silently overwriting the other -- both
    notebooks train / evaluate two models side by side.
    """

    def __init__(self, betas: torch.Tensor, device):
        self.timesteps = betas.shape[0]
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.device = device

    @classmethod
    def linear(cls, timesteps, device, beta_start=1e-4, beta_end=0.02):
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        return cls(betas, device)

    @classmethod
    def cosine(cls, timesteps, device, s=0.008):
        betas = cosine_beta_schedule(timesteps, s)
        return cls(betas, device)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """Gathers per-timestep coefficients and reshapes for broadcasting
    against an image batch of shape x_shape = [B, C, H, W]."""
    batch_size = t.shape[0]
    out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule, noise: torch.Tensor = None):
    """Forward diffusion: samples x_t directly from x0 (closed-form)."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(extract(schedule.alpha_bars, t, x0.shape))
    sqrt_one_minus_alpha_bar = torch.sqrt(extract(1 - schedule.alpha_bars, t, x0.shape))
    return sqrt_alpha_bar * x0 + noise * sqrt_one_minus_alpha_bar


# ---------------------------------------------------------------------------
# Conditional U-Net (identical backbone for both schedules)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetCond(nn.Module):
    """Conditional noise-prediction network, following the 9.3 Tutorial's
    `UNet_cond` (2-level encoder/decoder with additive timestep + class
    conditioning at every resolution). Generalised to accept any square
    image size; mnist_custom.pt uses 20x20 instead of 28x28, and 20 is
    still divisible by 4 so the same two pooling/upsampling stages apply.
    """

    def __init__(self, channels: int = 1, ch: int = 32, n_classes: int = 10, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps

        self.enc1 = ConvBlock(channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.bot = ConvBlock(ch * 2, ch * 2)
        self.dec2 = ConvBlock(ch * 2 + ch * 2, ch)
        self.dec1 = ConvBlock(ch + ch, ch)
        self.final = nn.Conv2d(ch, channels, 1)

        self.time_mlp = nn.Sequential(nn.Linear(1, ch), nn.ReLU(), nn.Linear(ch, ch))
        self.class_emb = nn.Embedding(n_classes, ch)

        self.to_e1 = nn.Conv2d(ch, ch, 1)
        self.to_e2 = nn.Conv2d(ch, ch * 2, 1)
        self.to_b = nn.Conv2d(ch, ch * 2, 1)
        self.to_d2 = nn.Conv2d(ch, ch, 1)
        self.to_d1 = nn.Conv2d(ch, ch, 1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / self.timesteps).unsqueeze(-1).unsqueeze(-1)
        y_emb = self.class_emb(y).unsqueeze(-1).unsqueeze(-1)
        cond_emb = t_emb + y_emb  # (B, ch, 1, 1)

        e1 = self.enc1(x) + self.to_e1(cond_emb)
        e2 = self.enc2(self.pool(e1)) + self.to_e2(cond_emb)

        b = self.bot(self.pool(e2)) + self.to_b(cond_emb)

        d2 = self.upsample(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2) + self.to_d2(cond_emb)

        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1) + self.to_d1(cond_emb)

        return self.final(d1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_ddpm(model: nn.Module, schedule: NoiseSchedule, dataloader, epochs: int, lr: float,
                device, run_name: str = "model"):
    """Trains `model` to predict the noise added by `schedule`'s forward
    process, following Algorithm 1 of the DDPM paper (Ho et al., 2020).

    Returns:
        loss_history: list of mean training loss per epoch.
        training_time: wall-clock training time in seconds.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(dataloader, unit="batch", desc=f"[{run_name}] epoch {epoch + 1}/{epochs}", colour="GREEN")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_size = imgs.shape[0]
            t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(imgs)
            x_t = q_sample(imgs, t, schedule, noise=noise)

            pred_noise = model(x_t, t, labels)
            loss = loss_fn(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        loss_history.append(sum(epoch_losses) / len(epoch_losses))

    training_time = time.time() - start_time
    return loss_history, training_time


# ---------------------------------------------------------------------------
# Sampling (reverse process), with optional step-count acceleration
# ---------------------------------------------------------------------------
@torch.no_grad()
def p_sample_step(model: nn.Module, x_t: torch.Tensor, t_index: int, schedule: NoiseSchedule,
                   y: torch.Tensor, orig_t_index: int = None):
    """One reverse step of ancestral (DDPM) sampling (Algorithm 2, Ho et al. 2020).

    t_index      : index into `schedule`'s own betas/alphas/alpha_bars.
    orig_t_index : the ORIGINAL timestep (0..T_train-1) to feed the model's
                   time embedding with. Needed for accelerated / respaced
                   sampling, where `schedule` only covers a subset of the
                   timesteps the model was trained on. Defaults to t_index
                   (i.e. no respacing).
    """
    if orig_t_index is None:
        orig_t_index = t_index

    batch = x_t.shape[0]
    t_model = torch.full((batch,), orig_t_index, device=x_t.device, dtype=torch.long)
    t_local = torch.full((batch,), t_index, device=x_t.device, dtype=torch.long)

    beta = extract(schedule.betas, t_local, x_t.shape)
    alpha = extract(schedule.alphas, t_local, x_t.shape)
    alpha_bar = extract(schedule.alpha_bars, t_local, x_t.shape)
    sigma = torch.sqrt(beta)

    pred_noise = model(x_t, t_model, y)

    z = torch.randn_like(x_t) if t_index > 0 else 0
    x_tm1 = 1.0 / torch.sqrt(alpha) * (x_t - beta / torch.sqrt(1 - alpha_bar) * pred_noise) + sigma * z
    return x_tm1


def build_respaced_schedule(schedule: NoiseSchedule, num_steps: int):
    """Builds an accelerated sampling schedule that reuses a model trained
    with `schedule.timesteps` steps, but evaluates it at only `num_steps` of
    the original timesteps -- a "respaced" DDPM sampler (Nichol & Dhariwal,
    2021, Sec. 4). This is what lets us test the project's hypothesis: can a
    schedule still produce good samples if we take fewer reverse-process
    steps than it was trained with?

    Returns:
        respaced      : a NoiseSchedule over the num_steps subsequence.
        orig_indices  : LongTensor mapping respaced index -> original
                         timestep (needed for the model's time embedding).
    """
    T = schedule.timesteps
    num_steps = min(num_steps, T)
    orig_indices = torch.linspace(0, T - 1, num_steps, device=schedule.device).round().long().unique()

    alpha_bars_subset = schedule.alpha_bars[orig_indices]
    prev_alpha_bars = torch.cat([torch.ones(1, device=schedule.device), alpha_bars_subset[:-1]])
    betas_respaced = 1 - (alpha_bars_subset / prev_alpha_bars)
    betas_respaced = torch.clamp(betas_respaced, min=1e-5, max=0.999)

    respaced = NoiseSchedule(betas_respaced, schedule.device)
    return respaced, orig_indices


@torch.no_grad()
def sample_ddpm(model: nn.Module, schedule: NoiseSchedule, labels: torch.Tensor, image_size: int,
                 channels: int = 1, num_steps: int = None, device="cpu", seed: int = 1337,
                 show_progress: bool = True):
    """Conditional ancestral sampling.

    If num_steps is None (default), every one of schedule.timesteps steps is
    used (standard DDPM sampling). If num_steps < schedule.timesteps, an
    accelerated "respaced" sampler is used instead (see
    build_respaced_schedule). Passing num_steps == schedule.timesteps is
    equivalent to the default and reproduces standard sampling exactly.
    """
    model.eval()
    torch.manual_seed(seed)
    if num_steps is None:
        num_steps = schedule.timesteps

    local_schedule, orig_indices = build_respaced_schedule(schedule, num_steps)

    n = labels.shape[0]
    x = torch.randn(n, channels, image_size, image_size, device=device)
    K = local_schedule.timesteps
    iterator = reversed(range(K))
    if show_progress:
        iterator = tqdm(iterator, total=K, desc=f"Sampling ({K} steps)", colour="BLUE")
    for i in iterator:
        orig_t = int(orig_indices[i].item())
        x = p_sample_step(model, x, i, local_schedule, labels, orig_t_index=orig_t)
    return x.clamp(-1, 1)


def make_class_conditioned_labels(n_classes: int, samples_per_class: int, device) -> torch.Tensor:
    """Builds a label tensor [0,0,...,0, 1,1,...,1, ..., 9,9,...,9] with
    `samples_per_class` repeats each, for conditional sample grids."""
    return torch.arange(n_classes, device=device).repeat_interleave(samples_per_class)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_ddpm_checkpoint(path: str, model: nn.Module, schedule_name: str, schedule_kwargs: dict,
                          timesteps: int, ch: int, n_classes: int, image_size: int):
    """Saves everything needed to reconstruct the model AND its exact
    training schedule (so main_report.ipynb can rebuild `NoiseSchedule`
    without re-deriving betas by hand)."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "schedule_name": schedule_name,       # "linear" or "cosine"
        "schedule_kwargs": schedule_kwargs,    # e.g. {"beta_start": 1e-4, "beta_end": 0.02} or {"s": 0.008}
        "timesteps": timesteps,
        "ch": ch,
        "n_classes": n_classes,
        "image_size": image_size,
        "channels": 1,
    }, path)


def load_ddpm_checkpoint(path: str, device):
    """Recreates a UNetCond model + its NoiseSchedule from a checkpoint
    saved by save_ddpm_checkpoint()."""
    ckpt = torch.load(path, map_location=device)

    model = UNetCond(channels=ckpt["channels"], ch=ckpt["ch"], n_classes=ckpt["n_classes"],
                      timesteps=ckpt["timesteps"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    if ckpt["schedule_name"] == "linear":
        schedule = NoiseSchedule.linear(ckpt["timesteps"], device, **ckpt["schedule_kwargs"])
    elif ckpt["schedule_name"] == "cosine":
        schedule = NoiseSchedule.cosine(ckpt["timesteps"], device, **ckpt["schedule_kwargs"])
    else:
        raise ValueError(f"Unknown schedule_name in checkpoint: {ckpt['schedule_name']}")

    return model, schedule, ckpt


# ---------------------------------------------------------------------------
# Lightweight auxiliary classifier (quantitative evaluation only)
# ---------------------------------------------------------------------------
class DigitClassifier(nn.Module):
    """Small CNN used only to quantitatively score generated samples: does a
    generated image actually look like the digit it was conditioned on?
    This is NOT part of the diffusion model -- it is purely an evaluation
    tool trained once on the real mnist_custom.pt training set.
    """

    def __init__(self, channels: int = 1, n_classes: int = 10, image_size: int = 20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * reduced * reduced, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_classifier(train_ds: Dataset, device, epochs: int = 5, batch_size: int = 256, lr: float = 1e-3,
                      image_size: int = 20):
    """Trains the auxiliary DigitClassifier on real mnist_custom.pt images
    (already rescaled to [-1, 1] by MNISTCustomDataset, matching the
    convention used for diffusion samples)."""
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    clf = DigitClassifier(image_size=image_size).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        clf.train()
        pbar = tqdm(loader, desc=f"[classifier] epoch {epoch + 1}/{epochs}", colour="MAGENTA")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(clf(imgs), labels)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
    return clf


@torch.no_grad()
def classifier_accuracy(clf: nn.Module, images: torch.Tensor, labels: torch.Tensor, device) -> float:
    """Fraction of `images` (in [-1, 1]) the classifier assigns to `labels`."""
    clf.eval()
    preds = clf(images.to(device)).argmax(dim=1).cpu()
    return (preds == labels.cpu()).float().mean().item()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_grid(images: torch.Tensor, nrow: int, title: str = None, figsize=(16, 8), save_path: str = None):
    """Displays (and optionally saves) a grid of generated/real images.
    `images` are expected in [-1, 1]."""
    grid = vutils.make_grid((images.clamp(-1, 1) + 1) / 2, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
