from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from dalle_pytorch.vae import OpenAIDiscreteVAE
from dalle_pytorch.vae import VQGanVAE1024, VQGanVAE16384
from dalle_pytorch.transformer import Transformer
from dalle_pytorch.dalle_pytorch import eval_decorator, exists, always, is_empty, top_k, set_requires_grad, DiscreteVAE
from training.networks import Generator

class StyleGANGenerator(nn.Module):
    def __init__(
            self,
            *,
            dim,
            vae,
            num_frames=10000,
            text_seq_len=16,
            depth,
            heads=8,
            dim_head=64,
            reversible=False,
            attn_dropout=0.,
            ff_dropout=0,
            sparse_attn=False,
            attn_types=None,
            num_videos=10, sigma=1, regularization=0.001
    ):
        super().__init__()
        assert isinstance(vae, (DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE1024, VQGanVAE16384)), 'vae must be an instance of DiscreteVAE'

        self.n_adain_layers = 4
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2
        channels_max = 1024
        self.generator = Generator(z_dim=dim,  # Input latent (Z) dimensionality.
                      c_dim=dim,  # Conditioning label (C) dimensionality.
                      w_dim=dim,  # Intermediate latent (W) dimensionality.
                      img_resolution=image_fmap_size,  # Output resolution.
                      img_channels=1,  # Number of output color channels.
                      num_image_tokens=num_image_tokens,
                      mapping_kwargs={},  # Arguments for MappingNetwork.
                      synthesis_kwargs={'channel_max': channels_max},  # Arguments for SynthesisNetwork.
                                   # 'channel_max':1024, 'architecture': 'skip'
                      )

        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len + 1
        self.image_seq_len = image_seq_len
        self.dim = dim
        self.class_dim = dim
        self.content_dim = dim
        seq_len = self.text_seq_len + self.image_seq_len

        self.total_seq_len = seq_len
        self.frame_emb_regularization = regularization
        self.noise_std = sigma
        self.vae = vae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained

        self.per_frame_emb = RegularizedEmbedding(num_frames, self.content_dim, sigma)
        self.per_video_emb = nn.Embedding(num_videos, self.class_dim)

    def forward(
            self,
            image=None, video_indices=None, frame_indices=None, return_loss=False, rel_codes=None, sigma=None):

        content_code = self.per_frame_emb(frame_indices)
        class_code = self.per_video_emb(video_indices)
        # class_adain_params = self.modulation(class_code)
        logits = self.generator(content_code, class_code)
        # print(logits.shape)
        if not return_loss:
            return logits
        if rel_codes is not None:
            labels = rel_codes
        else:
            labels = self.vae.get_codebook_indices(image)
        assert exists(image), 'when training, image must be supplied'
        logits = logits.flatten(2)
        # logits = rearrange(logits, 'b n c -> b c n')
        loss_img = F.cross_entropy(logits, labels)
        dim_size = content_code.shape[-1]
        raw_embedding = self.per_frame_emb.embedding(frame_indices)
        frame_emb_loss = (raw_embedding ** 2).mean() * dim_size
        return loss_img, self.frame_emb_regularization * frame_emb_loss

    @torch.no_grad()
    @eval_decorator
    def generate_images(
            self,
            frame_index,
            video_index,
            *,
            filter_thres=0.5,
            temperature=1.
    ):
        out_im_codes_probs = self(image=None, video_indices=video_index, frame_indices=frame_index, return_loss=False)
        out_im_codes = torch.argmax(out_im_codes_probs, dim=1).flatten(1, 2)

        images = self.vae.decode(out_im_codes)

        return images

    @torch.no_grad()
    @eval_decorator
    def interpolate_images(
            self,
            frame_index1, frame_index2,
            video_index,
            *,
            filter_thres=0.5,
            temperature=1., interpolation_num=10
    ):
        vae, text_seq_len, image_seq_len, = self.vae, self.text_seq_len, self.image_seq_len
        device = frame_index1.device
        bs = len(video_index)

        frame_emb1 = self.per_frame_emb(frame_index1).unsqueeze(1)
        frame_emb2 = self.per_frame_emb(frame_index2).unsqueeze(1)
        video_emb = self.per_video_emb(video_index).view(bs, -1, self.class_dim)

        interpolator = (torch.arange(interpolation_num) / (interpolation_num - 1)).view(-1, 1, 1).cuda(device)
        content_code = interpolator * frame_emb1 + (1 - interpolator) * frame_emb2
        class_code = torch.ones_like(interpolator) * video_emb
        # class_adain_params = self.modulation(class_code)
        out_im_codes_probs = self.generator(content_code, class_code)
        out_im_codes = torch.argmax(out_im_codes_probs, dim=1).flatten(1, 2)

        images = self.vae.decode(out_im_codes)

        return images

    @torch.no_grad()
    @eval_decorator
    def sample_random_image(
            self,
            video_index, sigma, num_images,
            *,
            filter_thres=0.5,
            temperature=1., interpolation_num=10
    ):
        device = video_index.device
        bs = len(video_index)
        video_emb = self.per_video_emb(video_index).view(bs, -1, self.class_dim)
        video_embs = torch.ones(size=(num_images, 1, 1)).to(device) * video_emb
        frame_embs = torch.normal(mean=0, std=sigma, size=(num_images, 1, self.content_dim)).to(device)

        out_im_codes_probs = self.generator(frame_embs, video_embs)
        out_im_codes = torch.argmax(out_im_codes_probs, dim=1).flatten(1, 2)

        images = self.vae.decode(out_im_codes)

        return images


class RegularizedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, stddev):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.stddev = stddev

    def forward(self, x):
        x = self.embedding(x)

        if self.training and self.stddev != 0:
            noise = torch.zeros_like(x)
            noise.normal_(mean=0, std=self.stddev)

            x = x + noise

        return x
