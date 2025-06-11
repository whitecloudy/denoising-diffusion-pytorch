import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter, writer
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

import complex.complex_module as cm
import numpy as np
from Five_G_dataset import Five_G_dataset


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def divisible_by(numer, denom):
    return (numer % denom) == 0

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


# TODO: Replace MLP with nn.Embedding
class MLPConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, hidden_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            cm.ComplexLinear(cond_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim*4, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim*4, hidden_dim, bias=True),
        )
        self.apply(init_weight_norm)

    def forward(self, c):
        return self.projection(c)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x):
        x = self.projection(x)
        return cm.complex_mul(x, self.embedding.to(x.device))

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.s_attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.normc = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.x_attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, *block_kwargs)
        self.norm3 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, t, c):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            t).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.s_attn(mod_x, mod_x, mod_x)
        x = x + self.x_attn(queries=self.normc(c),
                            keys=self.norm2(x), values=self.norm2(x))
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.apply(init_weight_zero)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = x + modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class SpatialDiffusion(nn.Module):
    """
    Process each sample of a sequence.
    Take CSI diffusion as an example.
    - Input:\\
      x, [B, S, A, 2], \\
      t, [B], \\
      c, [B, C, 2], \\
    - Output:
      n, [B, S*A, 2]
    """

    def __init__(self, params):
        super().__init__()
        self.learn_tfdiff = params.learn_tfdiff
        self.num_block = params.num_spatial_block
        self.input_dim = params.extra_dim[-1]  # A
        self.input_len = params.extra_dim[-2]  # S
        self.output_dim = self.input_dim * self.input_len  # S*A
        self.hidden_dim = params.spatial_hidden_dim  # D
        self.num_heads = params.num_heads  # H
        self.max_step = params.max_step  # T
        self.embed_dim = params.embed_dim  # E
        self.cond_dim = params.cond_dim[-1]  # C
        self.dropout = params.dropout
        self.task_id = params.task_id
        self.mlp_ratio = params.mlp_ratio
        self.p_embed = PositionEmbedding(
            self.input_len, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            self.max_step, self.embed_dim, self.hidden_dim)
        self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
        # A series of concatenated DiA blocks.
        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(self.num_block)
        ])
        self.adaMLP = nn.Sequential(
            # Flatten [B, S, A, 2] to [B, S*A, 2]
            nn.Flatten(start_dim=1, end_dim=-2),
            cm.ComplexLinear(self.input_len*self.hidden_dim, self.output_dim),
            cm.ComplexSiLU(),
            cm.ComplexLinear(self.output_dim, self.output_dim),
        )
        self.adaMLP.apply(init_weight_xavier)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        c = self.c_embed(c)
        for block in self.blocks:
            x = block(x, t, c)
        x = self.adaMLP(x)
        return x


class TimeFrequencyDiffusion(nn.Module):
    """
    Process the whole sequence.
    Take CSI diffusion as an example.
    - Input:\\
      x, [B, N, S*A, 2], \\
      t, [B], \\
      c, [B, N, C, 2], \\
    - Output:
      n, [B, N, S*A, 2]
    """

    def __init__(self, params):
        super().__init__()
        self.learn_tfdiff = params.learn_tfdiff
        self.num_block = params.num_tf_block
        self.input_dim = np.prod(params.extra_dim)  # S*A
        self.input_len = params.sample_rate  # N
        self.output_dim = self.input_dim  # S*A
        self.hidden_dim = params.tf_hidden_dim  # D
        self.num_heads = params.num_heads  # H
        self.max_step = params.max_step  # T
        self.embed_dim = params.embed_dim  # E
        self.cond_dim = np.prod(params.cond_dim)  # C
        self.dropout = params.dropout
        self.task_id = params.task_id
        self.mlp_ratio = params.mlp_ratio
        self.p_embed = PositionEmbedding(
            self.input_len, self.input_dim, self.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            self.max_step, self.embed_dim, self.hidden_dim)
        self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(self.num_block)
        ])
        self.final_layer = FinalLayer(
            self.hidden_dim, self.output_dim)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        c = c.reshape([-1, self.input_len, 2496, 2])
        c = self.c_embed(c)
        for block in self.blocks:
            x = block(x, t, c)
        x = self.final_layer(x, t)
        return x


class tfdiff_mimo(nn.Module):
    """
    Signal Modulation and Augmentation via Generative Diffusion Model.
    Take CSI diffusion as an example.
    - Input:\\
      x, [B, N, S, A, 2], \\
      t, [B], \\
      c, [B, N, C, 2], \\
    - Output:
      n, [B, N, S, A, 2]
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.task_id = params.task_id
        self.sample_rate = params.sample_rate
        self.extra_dim = params.extra_dim
        self.cond_dim = params.cond_dim
        self.spatial_dim = np.prod(self.extra_dim)
        # N parallel SpatialDiffusion blocks.
        self.spatial_block = SpatialDiffusion(self.params)
        self.tf_block = TimeFrequencyDiffusion(self.params)

    def forward(self, x, t, c):
        t = t-1
        x_s = x.reshape([-1]+self.extra_dim+[2])  # [B*N, S, A, 2] 
        c_s = c.reshape([-1]+self.cond_dim+[2])  # [B*N, [C], 2]
        x_s = self.spatial_block(x_s, t.repeat(
            self.sample_rate), c_s)  # [B*N, S*A, 2]
        x = x_s.reshape([-1, self.sample_rate] +
                        [self.spatial_dim, 2])  # [B, N, S*A, 2]
        x = self.tf_block(x, t, c)  # [B, N, S*A, 2]
        x = x.reshape([-1, self.sample_rate] +
                      self.extra_dim+[2])  # [B, N, S, A, 2]
        return x


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.index_select(0, t)
    new_shape = out.shape + (1,) * (len(x_shape) - len(out.shape))
    return out.reshape(new_shape)

def linear_beta_schedule(timesteps):
    # TODO: Forcing our scheduler for now
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        data_shape,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        tqdm_disable = False,
    ):
        super().__init__()

        self.model = model
        self.data_shape = data_shape

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.tqdm_disable = tqdm_disable

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def model_forward(self, x, t, classes):
        model_out = self.model.forward(x, t, classes)
        return model_out


    def model_predictions(self, x, t, classes, clip_x_start = False):
        model_output = self.model_forward(x, t, classes)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, clip_denoised = True):
        preds = self.model_predictions(x, t, classes)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, clip_denoised = True):
        b, device = x.shape[0], x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, disable=self.tqdm_disable):
            img, x_start = self.p_sample(img, t, classes)

        return img

    @torch.inference_mode()
    def ddim_sample(self, classes, shape, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=self.tqdm_disable):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    @torch.inference_mode()
    def sample_with_class(self, classes):
        batch_size, data_shape = classes.shape[0], self.data_shape
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, *data_shape))

    # # sample with random classes
    # @torch.inference_mode()
    # def sample(self, batch_size = 16, cond_scale = 6., rescaled_phi = 0.7):
    #     classes = torch.randint(0, self.model.num_classes, (batch_size,), device = self.device)
    #     return self.sample_with_class(classes, cond_scale, rescaled_phi)

    @torch.inference_mode()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t, disable=self.tqdm_disable):
            img, _ = self.p_sample(img, i, classes)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        model_out = self.model_forward(x, t, classes)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        shape, device, data_shape = img.shape, img.device, self.data_shape
        b = shape[0]
        assert shape[1] == data_shape[0] and shape[2] == data_shape[1] and shape[3] == data_shape[2], f'height and width of image must be {data_shape}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, *args, **kwargs)


@torch.inference_mode()
def get_kernel(blur_kernel, input_dim):
    samples = torch.arange(0, input_dim) # [N]
    gaussian_kernel = torch.exp(-((samples - input_dim // 2)**2) / (2 * blur_kernel))
    # gaussian_kernel = torch.exp(-((samples - self.input_dim // 2)**2) / (2 * var_kernel)) / torch.sqrt(2 * torch.pi * var_kernel) # G_t, [T, N]
    # gaussian_kernel = self.input_dim * gaussian_kernel / torch.sum(gaussian_kernel, dim=1, keepdim=True) # Normalized G_t, [T, N]
    return gaussian_kernel

@torch.inference_mode()
def get_sigma_bar_weights(input_dim: int, diffusion_steps: int, gamma_weights, sigma_weights):
    noise_weights = []
    noise_weight_square = torch.zeros(input_dim) # [N]

    for t in range(diffusion_steps):
        noise_weight_square *= (gamma_weights[t] ** 2)
        noise_weight_square += (torch.ones(input_dim) * sigma_weights[t] ** 2)
        noise_weights.append(torch.sqrt(noise_weight_square).clone().detach())

    return torch.stack(noise_weights, dim=0) # [T, N] 



class SignalDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        data_shape,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        tqdm_disable = False,
        freq_blur = None,
    ):
        super().__init__()

        self.model = model
        self.data_shape = data_shape

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.tqdm_disable = tqdm_disable

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
       
        input_time_dim = data_shape[0]

        freq_blur = torch.tensor(default(freq_blur, torch.zeros(timesteps, dtype=torch.float32)+torch.inf), dtype=torch.float32)
        freq_blur_bar = torch.cumsum(freq_blur, dim=0)

        freq_kernel = (input_time_dim / freq_blur).unsqueeze(1)
        freq_kernel_bar = (input_time_dim / freq_blur_bar).unsqueeze(1)

        gaussian_kernel = get_kernel(freq_kernel, input_time_dim)
        gaussian_kernel_bar = get_kernel(freq_kernel_bar, input_time_dim)

        gamma_weights = gaussian_kernel * torch.sqrt(alphas).unsqueeze(1)
        gamma_weights_bar = gaussian_kernel_bar * torch.sqrt(alphas_cumprod).unsqueeze(1)
        gamma_weights_bar_prev = torch.cat([torch.ones_like(gamma_weights_bar[0]).unsqueeze(0), (gamma_weights_bar[:-1])], dim=0)

        sigma_weights = torch.sqrt(betas).unsqueeze(1)
        sigma_weights_bar = get_sigma_bar_weights(input_dim=input_time_dim, 
                                                  diffusion_steps=timesteps, 
                                                  gamma_weights=gamma_weights, 
                                                  sigma_weights=sigma_weights)
        sigma_weights_bar_prev = torch.cat([torch.zeros_like(sigma_weights_bar[0]).unsqueeze(0), (sigma_weights_bar[:-1])], dim=0)
        
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)

        register_buffer('gamma_weights', gamma_weights)
        register_buffer('gamma_weights_bar', gamma_weights_bar)
        register_buffer('gamma_weights_bar_prev', gamma_weights_bar_prev)

        register_buffer('sigma_weights', sigma_weights)
        register_buffer('sigma_weights_bar', sigma_weights_bar)
        register_buffer('sigma_weights_bar_prev', sigma_weights_bar_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('recip_gamma_weights', 1. / gamma_weights)
        register_buffer('recip_gamma_weights_bar', 1. / gamma_weights_bar)
        register_buffer('sigma_weights_bar_recip_gamma_weights_bar', sigma_weights_bar / gamma_weights_bar)

        register_buffer('recip_gamma_weights_bar_sqr_plus_sigma_weights_bar_sqr',
                        1. / (gamma_weights_bar**2 + sigma_weights_bar**2))

        #TODO : REPLACE
        # register_buffer('alphas_cumprod', alphas_cumprod)
        # register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        #TODO : REPLACE
        # register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (sigma_weights * sigma_weights_bar_prev / sigma_weights_bar)**2

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef_x_t', (gamma_weights * sigma_weights_bar_prev**2) / (sigma_weights_bar**2))
        register_buffer('posterior_mean_coef_x_0', (gamma_weights_bar_prev * sigma_weights**2) / (sigma_weights_bar**2))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = (gamma_weights_bar / sigma_weights_bar)**2

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.recip_gamma_weights_bar, t, x_t.shape) * x_t -
            extract(self.sigma_weights_bar_recip_gamma_weights_bar, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (x_t - extract(self.gamma_weights_bar, t, x_t.shape) * x0) / \
            extract(self.sigma_weights_bar, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.gamma_weights_bar, t, x_start.shape) * noise -
            extract(self.sigma_weights_bar, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            (extract(self.gamma_weights_bar, t, x_t.shape) * x_t -
            extract(self.sigma_weights_bar, t, x_t.shape) * v) / 
            extract(self.recip_gamma_weights_bar_sqr_plus_sigma_weights_bar_sqr, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef_x_t, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef_x_0, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    

    def model_forward(self, x, t, classes):
        model_out = self.model.forward(x, t, classes)
        return model_out


    def model_predictions(self, x, t, classes, clip_x_start = False):
        model_output = self.model_forward(x, t, classes)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start
            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, clip_denoised = True):
        preds = self.model_predictions(x, t, classes)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, clip_denoised = True):
        b, device = x.shape[0], x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape):
        batch, device = shape[0], self.betas.device

        x_T = torch.randn(shape, device=device)
        batch_max = ((self.sampling_timesteps-1)*torch.ones(batch, dtype=torch.int64)).to(device)
        
        inf_weight = extract(self.sigma_weights_bar, batch_max, x_T.shape) + \
                     extract(self.gamma_weights_bar, batch_max, x_T.shape)
        x_T = x_T * inf_weight
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps, disable=self.tqdm_disable):
            x_T, x_start = self.p_sample(x_T, t, classes)

        return x_T

    #TODO : Need to fix for SignalDiffusion
    @torch.inference_mode()
    def ddim_sample(self, classes, shape, clip_denoised = True):
        assert False, "DDIM sampling is not implemented for SignalDiffusion yet."
    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     img = torch.randn(shape, device = device)

    #     x_start = None

    #     for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=self.tqdm_disable):
    #         time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
    #         pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, clip_x_start = clip_denoised)

    #         if time_next < 0:
    #             img = x_start
    #             continue

    #         alpha = self.gamma_weights_bar[time]
    #         alpha_next = self.gamma_weights_bar[time_next]

    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()

    #         noise = torch.randn_like(img)

    #         img = x_start * alpha_next.sqrt() + \
    #               c * pred_noise + \
    #               sigma * noise

    #     return img

    @torch.inference_mode()
    def sample_with_class(self, classes):
        batch_size, data_shape = classes.shape[0], self.data_shape
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, *data_shape))

    # # sample with random classes
    # @torch.inference_mode()
    # def sample(self, batch_size = 16, cond_scale = 6., rescaled_phi = 0.7):
    #     classes = torch.randint(0, self.model.num_classes, (batch_size,), device = self.device)
    #     return self.sample_with_class(classes, cond_scale, rescaled_phi)

    @torch.inference_mode()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t, disable=self.tqdm_disable):
            img, _ = self.p_sample(img, i, classes)

        return img

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.gamma_weights_bar, t, x_start.shape) * x_start +
            extract(self.sigma_weights_bar, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, *, classes, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step
        model_out = self.model_forward(x, t, classes)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b t ... -> b t', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        shape, device, data_shape = img.shape, img.device, self.data_shape
        b = shape[0]
        assert shape[1] == data_shape[0] and shape[2] == data_shape[1] and shape[3] == data_shape[2], f'height and width of image must be {data_shape}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, *args, **kwargs)