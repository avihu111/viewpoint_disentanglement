from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from dalle_pytorch import distributed_utils
from dalle_pytorch.vae import OpenAIDiscreteVAE
from dalle_pytorch.vae import VQGanVAE1024
from dalle_pytorch.transformer import Transformer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def is_empty(t):
    return t.nelement() == 0

def masked_mean(t, mask, dim = 1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim = 1) / mask.sum(dim = 1)[..., None]

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# discrete vae class

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = ((0.5,) * 3, (0.5,) * 3)
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

        self._register_external_parameters()
        self.iterations = 0

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(
        self,
        img,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)
        if self.iterations < 1000:

            loss = recon_loss
        else:
            loss = recon_loss + kl_div_loss_weight * kl_div
        self.iterations += 1
        if not return_recons:
            return loss

        return recon_loss, kl_div_loss_weight * kl_div, out

# main classes

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        num_visual_tokens = 512,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_image_size = 256,
        visual_patch_size = 32,
        channels = 3
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
        self.text_transformer = Transformer(causal = False, seq_len = text_seq_len, dim = dim_text, depth = text_enc_depth, heads = text_heads)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2

        self.visual_patch_size = visual_patch_size
        self.to_visual_embedding = nn.Linear(patch_dim, dim_image)
        self.visual_pos_emb = nn.Embedding(num_patches, dim_image)
        self.visual_transformer = Transformer(causal = False, seq_len = num_patches, dim = dim_image, depth = visual_enc_depth, heads = visual_heads)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.tensor(1.))

    def forward(
        self,
        text,
        image,
        text_mask = None,
        return_loss = False
    ):
        b, device, p = text.shape[0], text.device, self.visual_patch_size

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        image_emb = self.to_visual_embedding(image_patches)
        image_emb += self.visual_pos_emb(torch.arange(image_emb.shape[1], device = device))

        enc_text = self.text_transformer(text_emb, mask = text_mask)
        enc_image = self.visual_transformer(image_emb)

        if exists(text_mask):
            text_latents = masked_mean(enc_text, text_mask, dim = 1)
        else:
            text_latents = enc_text.mean(dim = 1)

        image_latents = enc_image.mean(dim = 1)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim

        sim = einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

# main DALL-E class

class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_frames = 10000,
        text_seq_len = 16,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        num_videos=10
    ):
        super().__init__()
        assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024)), 'vae must be an instance of DiscreteVAE'
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2
        num_embedding_tokens = num_videos + num_frames + text_seq_len
        # reserve unique padding tokens for each position (text seq len)
        self.per_frame_emb = nn.Embedding(num_frames + 1, dim)
        self.per_video_emb = nn.Embedding(num_videos, dim * (text_seq_len -1))
        self.image_emb = nn.Embedding(num_image_tokens, dim)
        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size))
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len + 1
        self.image_seq_len = image_seq_len
        self.dim = dim
        seq_len = self.text_seq_len + self.image_seq_len

        self.total_seq_len = seq_len
        self.frame_emb_regularization = 0.001 # as in lord
        self.noise_std = 1 # as in lord
        self.vae = vae
        set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        frame_index,
        video_index,
        *,
        filter_thres = 0.5,
        temperature = 1.
    ):
        vae, text_seq_len, image_seq_len, = self.vae, self.text_seq_len, self.image_seq_len
        device = frame_index.device
        out = torch.zeros((len(frame_index), self.image_seq_len), dtype=frame_index.dtype).cuda(device)
        for cur_len in range(image_seq_len):
            logits = self(image=out[:cur_len + text_seq_len], frame_indices=frame_index,
                          video_indices=video_index)[:, cur_len + text_seq_len - 1, :]
            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            out[:, cur_len] = sample[:,0]

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        return images

    def forward(
        self,
        image=None, video_indices=None, frame_indices=None, return_loss = False):
        device = image.device
        bs = len(video_indices)
        bos_index = torch.full_like(frame_indices, self.per_frame_emb.num_embeddings - 1)
        bos_embs = self.per_frame_emb(bos_index).cuda(device).unsqueeze(1)
        video_embs = self.per_video_emb(video_indices).cuda(device).view(bs, -1, self.dim)
        frame_embs = self.per_frame_emb(frame_indices).cuda(device).unsqueeze(1)
        noisy_embs = frame_embs + torch.normal(0, self.noise_std, size=frame_embs.shape).cuda(device)
        tokens = torch.cat([bos_embs, video_embs, noisy_embs], dim=1).cuda(device)
        tokens += self.text_pos_emb(torch.arange(self.text_seq_len).cuda(device)).unsqueeze(0)
        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                assert tuple(image.shape[1:]) == (3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)
            image_emb += self.image_pos_emb(image_emb)
            tokens = torch.cat((tokens, image_emb), dim = 1)
            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained
        if tokens.shape[1] > self.total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        out = self.transformer(tokens)
        logits = self.to_logits(out)
        if not return_loss:
            return logits
        labels = image
        assert exists(image), 'when training, image must be supplied'
        logits = rearrange(logits, 'b n c -> b c n')
        loss_img = F.cross_entropy(logits[..., self.text_seq_len - 1: -1], labels)
        frame_emb_loss = (frame_embs ** 2).mean()
        return loss_img, self.frame_emb_regularization * frame_emb_loss
