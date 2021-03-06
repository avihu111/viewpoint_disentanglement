from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from dalle_pytorch.vae import OpenAIDiscreteVAE
from dalle_pytorch.vae import VQGanVAE1024, VQGanVAE16384
from dalle_pytorch.transformer import Transformer
from dalle_pytorch.dalle_pytorch import eval_decorator, exists, always, is_empty, top_k, set_requires_grad, DiscreteVAE


class LordGenerator(nn.Module):
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
        # self.content_dim = 512
        # self.class_dim = 512
        # self.adain_dim = 512
        self.content_dim = 128
        self.class_dim = 256
        self.adain_dim = 256
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len + 1
        self.image_seq_len = image_seq_len
        self.dim = dim
        seq_len = self.text_seq_len + self.image_seq_len

        self.total_seq_len = seq_len
        self.frame_emb_regularization = regularization
        self.noise_std = sigma
        self.vae = vae
        set_requires_grad(self.vae, False)  # freeze VAE from being trained

        self.per_frame_emb = RegularizedEmbedding(num_frames, self.content_dim, sigma)
        self.per_video_emb = nn.Embedding(num_videos, self.class_dim)
        self.modulation = Modulation(self.class_dim, self.n_adain_layers, self.adain_dim)
        self.generator = Generator(self.content_dim, self.n_adain_layers, self.adain_dim,
                                   (image_fmap_size, image_fmap_size, vae.num_tokens))

    def forward(
            self,
            image=None, video_indices=None, frame_indices=None, return_loss=False, rel_codes=None, sigma=None):

        content_code = self.per_frame_emb(frame_indices)
        class_code = self.per_video_emb(video_indices)
        class_adain_params = self.modulation(class_code)
        logits = self.generator(content_code, class_adain_params)

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
        class_adain_params = self.modulation(class_code)
        out_im_codes_probs = self.generator(content_code, class_adain_params)
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

        class_adain_params = self.modulation(video_embs)
        out_im_codes_probs = self.generator(frame_embs, class_adain_params)
        out_im_codes = torch.argmax(out_im_codes_probs, dim=1).flatten(1, 2)

        images = self.vae.decode(out_im_codes)

        return images


class Modulation(nn.Module):

    def __init__(self, code_dim, n_adain_layers, adain_dim):
        super().__init__()

        self.__n_adain_layers = n_adain_layers
        self.__adain_dim = adain_dim

        self.adain_per_layer = nn.ModuleList([
            nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
            for _ in range(n_adain_layers)
        ])

    def forward(self, x):
        adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
        adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

        return adain_params


class Generator(nn.Module):

    def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
        super().__init__()

        self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
        self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
        self.__adain_dim = adain_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=content_dim,
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
                out_features=self.__initial_height * self.__initial_width * adain_dim
            ),

            nn.LeakyReLU()
        )

        self.adain_conv_layers = nn.ModuleList()
        for i in range(n_adain_layers):
            self.adain_conv_layers += [
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
                nn.LeakyReLU(),
                AdaptiveInstanceNorm2d(adain_layer_idx=i)
            ]

        self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)
        # more capacity
        self.last_conv_layers = nn.Sequential(
            # nn.Conv2d(in_channels=adain_dim, out_channels=256, padding=2, kernel_size=5),
            nn.Conv2d(in_channels=adain_dim, out_channels=2048, padding=1, kernel_size=3),
            nn.LeakyReLU(),

            # nn.Conv2d(in_channels=256, out_channels=img_shape[2], padding=3, kernel_size=7),
            nn.Conv2d(in_channels=2048, out_channels=img_shape[2], padding=1, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=img_shape[2], padding=0, kernel_size=1),
            # nn.Softmax(dim=1)
        )

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.adain_layer_idx, :, 0]
                m.weight = adain_params[:, m.adain_layer_idx, :, 1]

    def forward(self, content_code, class_adain_params):
        self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

        return x


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, adain_layer_idx):
        super().__init__()
        self.weight = None
        self.bias = None
        self.adain_layer_idx = adain_layer_idx

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]

        x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
        weight = self.weight.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = F.batch_norm(
            x_reshaped, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True
        )

        out = out.view(b, c, *x.shape[2:])
        return out


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
