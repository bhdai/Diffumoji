import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="sigmoid",
    ):
        super().__init__()

        self.model = model
        self.channels = 3
        self.image_size = image_size
        self.objective = objective
        assert objective in ["pred_noise", "pred_x_start"], (
            "Objective must be either pred_noise (predict noise) or pred_x_start(predict image start)"
        )

        # a helper function to reister some constants as buffers to ensure that
        # they are on the same device as model parameters
        # can be access as self.name
        register_buffer = lambda name, val: self.register_buffer(name, val.float())
        betas = get_beta_schedule(beta_schedule, timesteps)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        # precompure the coeffes for the closed-form "jump ahead" formula
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # details about those coeffes please refer to https://arxiv.org/abs/2006.11239
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
        register_buffer("posterior_std", posterior_std)

        snr = alphas_cumprod / (1 - alphas_cumprod)  # signal to nosie ratio
        loss_weights = torch.ones_like(snr) if objective == "pred_noise" else snr
        register_buffer("loss_weights", loss_weights)

    def q_sample(self, x_start, t, noise=None):
        """Forward process: jumping directly to any noisy time step t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, context):
        """
        x_start is the clean image from the dataloader
        context is the pre-computed text embedding

        this should compute the training loss
        """
        batch_size = x_start.shape[0]
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=x_start.device
        ).long()
        noise = torch.randn_like(x_start)  # generate noise
        x_t = self.q_sample(x_start, t, noise)  # create the nozy image x_t
        model_out = self.model(x_t, t, context)

        # determine the target
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x_start":
            target = x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = loss.mean(dim=[1, 2, 3])  # mean over spatial dimensions

        loss_weights_t = extract(self.loss_weights, t, loss.shape)
        loss = loss * loss_weights_t
        return loss.mean()

    def predict_start_from_noise(self, x_t, t, noise):
        """predict x_start from predicted noise"""
        sqrt_recip_alphas_cumprod = extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod = extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    @torch.no_grad()
    def p_sample(self, x, t, context):
        """should be used in inference/sampling, sample x_{t-1} from the model at timestep t"""
        batch_size = x.shape[0]
        model_out = self.model(x, t, context)

        if self.objective == "pred_noise":
            x_start_pred = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == "pred_x_start":
            x_start_pred = model_out
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        x_start_pred = x_start_pred.clamp(-1, 1)  # clamp predicted x_start to [-1, 1]

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start_pred
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_std = extract(self.posterior_std, t, x.shape)

        # sample from posterior
        noise = torch.randn_like(x)
        # only add noise if t > 0
        nonzero_mask = (t > 0).float().reshape(batch_size, *((1,) * (len(x.shape) - 1)))

        return posterior_mean + nonzero_mask * posterior_std * noise


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # basically a[t] but for a batch of indices
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def get_beta_schedule(schedule_name, num_timesteps):
    """Generate beta schedule for diffusion process"""
    if schedule_name == "linear":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_name == "cosine":
        return cosine_beta_schedule(num_timesteps)
    elif schedule_name == "sigmoid":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return (
            torch.sigmoid(torch.linspace(-6, 6, num_timesteps))
            * (beta_end - beta_start)
            + beta_start
        )
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
