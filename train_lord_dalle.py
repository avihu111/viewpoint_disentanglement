import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils
import wandb  # Quit early if user doesn't have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE, VQGanVAE16384
from dalle_pytorch.lord_generator import LordGenerator
from dalle_pytorch.stylegan_generator import StyleGANGenerator
from cars import Cars3D
from ffhq import FFHQ
# argument parsing
from real_estate import RealEstate10K
from multi_dataloader import MultiEpochsDataLoader
import sys


torch.hub.set_dir('/cs/labs/daphna/avihu.dekel/.cache/')
VAE_MAPPER = {'taming': VQGanVAE1024, 'geo': VQGanVAE16384, 'openai': OpenAIDiscreteVAE}

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--vae_path', type=str, default=None,
                       help='path to your trained discrete VAE')
    group.add_argument('--dalle_path', type=str,
                       default=None,
                       help='path to your partially trained DALL-E')
    parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=0.75,
                        help='Random resized crop lower ratio')
    parser.add_argument('--vae_type', dest='vae_type', default='taming', help='taming|geo|openai')
    parser.add_argument('--fp16', action='store_true',
                        help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')
    parser.add_argument('--wandb_name', default='dalle_train_transformer',
                        help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')
    parser.add_argument('--dalle_output_file_name', type=str, default="dalle_small_taming",
                        help='output_file_name')
    parser.add_argument('--dataset', type=str, default="realestate",
                        help='dataset')
    train_group = parser.add_argument_group('Training settings')
    train_group.add_argument('--epochs', default=500, type=int, help='Number of epochs')
    train_group.add_argument('--save_every_n_steps', default=5000, type=int, help='Save a checkpoint every n steps')
    train_group.add_argument('--batch_size', default=4, type=int, help='Batch size')
    train_group.add_argument('--ga_steps', default=1, type=int,
                             help='Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')
    train_group.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')

    train_group.add_argument('--clip_grad_norm', default=0.5, type=float, help='Clip gradient norm')
    train_group.add_argument('--lr_decay', dest='lr_decay', action='store_true')
    model_group = parser.add_argument_group('Model settings')
    train_group.add_argument('--extract_codes', dest='extract_codes', action='store_true')
    model_group.add_argument('--dim', default=512, type=int, help='Model dimension')
    model_group.add_argument('--text_seq_len', default=20, type=int, help='Text sequence length')
    model_group.add_argument('--depth', default=4, type=int, help='Model depth')
    model_group.add_argument('--heads', default=8, type=int, help='Model number of heads')
    model_group.add_argument('--dim_head', default=64, type=int, help='Model head dimension')
    model_group.add_argument('--reversible', dest='reversible', action='store_true')
    model_group.add_argument('--generator', default='transformer', choices=['transformer', 'lord', 'stylegan'])
    model_group.add_argument('--cheating', dest='cheating', action='store_true')
    model_group.add_argument('--attn_types', default='full', type=str,
                             help='comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')
    model_group.add_argument('--device', default=0, type=int, help='device')
    model_group.add_argument('--sigma', default=1, type=float, help='noise injected in training')
    model_group.add_argument('--regularization', default=0.001, type=float, help='regularization')
    return parser.parse_args()


def exists(val):
    return val is not None


def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


def is_debug_mode():
    return sys.gettrace() is not None


def save_model(path, dalle_params, vae_params, dalle):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
    }

    save_obj = {
        **save_obj,
        'weights': dalle.state_dict()
    }

    torch.save(save_obj, path)


def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def reconstruct(im, vae, args):
    codes = vae.get_codebook_indices(im.to(args.device).unsqueeze(0))
    new_im = vae.decode(codes)[0].detach().cpu().permute(1, 2, 0)
    return new_im

def log_outputs(video_indices, frame_indices, generator, vae, log, images, epoch, args, rel_codes):
    # generate random image:
    # sampled_images = generator.sample_random_image(video_indices[:1], sigma=2, num_images=9)
    # log['samples'] = wandb.Image(torchvision.utils.make_grid(sampled_images, nrow=3))
    # generate crossovers
    fr = torch.cat([frame_indices[:2], frame_indices[:2]], dim=0)
    vd = torch.cat([video_indices[:2], torch.flip(video_indices[:2], (0,))], dim=0)
    recon = generator.generate_images(frame_index=fr,
                                      video_index=vd,
                                      filter_thres=0.99).cpu()  # topk sampling at 0.9
    dalle_recon = recon[:2]
    crossovers = recon[2:]
    if args.dataset == 'realestate':
        crossovers = torch.flip(crossovers, dims=(0,))

    vae_recon = vae.decode(rel_codes[:2]).detach().cpu()
    grid = torchvision.utils.make_grid(
        torch.cat([images[:2].cpu(), vae_recon, dalle_recon, crossovers], dim=0), nrow=2)
    log['all'] = wandb.Image(grid)
    log['vae'] = wandb.Image(torchvision.utils.make_grid(vae_recon, nrow=1))
    log['dalle'] = wandb.Image(torchvision.utils.make_grid(dalle_recon, nrow=1))
    log['image'] = wandb.Image(torchvision.utils.make_grid(images[:2], nrow=1))

    # if ((epoch + 1) % 20) == 0:
    #     INTERPOLATION_NUM = 20
    #     first_index = video_indices[:1] * 0
    #     interpolations = generator.interpolate_images(first_index,
    #                                                   first_index + 5,
    #                                               video_indices[:1], filter_thres=0.99,
    #                                                   interpolation_num=INTERPOLATION_NUM)
    #     interpolations = (interpolations * 255).cpu().to(torch.uint8)
    #     forward_backward_iterpolated = torch.cat([interpolations, torch.flip(interpolations, dims=(0,))], dim=0)
    #     log['interpolate'] = wandb.Video(forward_backward_iterpolated)

    return log


def main():
    args = parse_args()
    if exists(args.dalle_path):
        dalle_path = Path('checkpoints/' + args.dalle_path)

        assert dalle_path.exists(), 'DALL-E model file does not exist'
        loaded_obj = torch.load(str(dalle_path), map_location='cpu')

        dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
        dalle_params['regularization'] = args.regularization
        dalle_params['sigma'] = args.sigma
        if vae_params is not None:
            vae = DiscreteVAE(**vae_params)
        else:
            vae_klass = VAE_MAPPER[args.vae_type]
            vae = vae_klass()

        dalle_params = dict(
            **dalle_params
        )

    else:
        if exists(args.vae_path):
            vae_path = Path(args.vae_path)
            assert vae_path.exists(), 'VAE model file does not exist'
            loaded_obj = torch.load(str(vae_path), map_location=f'cuda:{args.device}')
            vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']
            vae = DiscreteVAE(**vae_params)
            vae.load_state_dict(weights)
        else:
            print('using pretrained VAE for encoding images to tokens')
            vae_params = None
            vae_klass = VAE_MAPPER[args.vae_type]
            vae = vae_klass()
        vae.cuda(args.device)

        dalle_params = dict(
            text_seq_len=args.text_seq_len,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            dim_head=args.dim_head,
            reversible=args.reversible,
            attn_types=tuple(args.attn_types.split(',')),
            sigma=args.sigma,
            regularization=args.regularization
        )
    # configure OpenAI VAE for float16s
    if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
        vae.enc.blocks.output.conv.use_float16 = True

    # helpers

    # create dataset and dataloader
    is_shuffle = not args.extract_codes
    if args.dataset == 'realestate':
        ds = RealEstate10K(
            root='./dataset',
            transform=T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                 T.CenterCrop(360),
                                 T.Resize(vae.image_size),
                                 # T.RandomResizedCrop(IMAGE_SIZE, scale=(args.resize_ratio, 1.), ratio=(1., 1.)),
                                 T.ToTensor()
                                 ]),
            split='train')
    elif args.dataset == 'cars':
        ds = Cars3D(
            root='./',
            transform=T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                 T.Resize(vae.image_size),
                                 # T.RandomResizedCrop(IMAGE_SIZE, scale=(args.resize_ratio, 1.), ratio=(1., 1.)),
                                 T.ToTensor()
                                 ]))
    elif 'ffhq' in args.dataset:
        small = 'small' in args.dataset
        ds = FFHQ(root='/cs/labs/peleg/avivga/data/images/ffhq/',
                  transform=T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                       T.Resize(vae.image_size),
                                       T.ToTensor()
                                       ]),
                  small=small
                  )
    else:
        raise ValueError('unknown dataset')

    # im = ds[0][0]
    # vqgan = VQGanVAE1024().to(args.device)
    # openai_vae = OpenAIDiscreteVAE().to(args.device)
    # openai_recon = reconstruct(im, openai_vae, args)
    # vqgan_recon = reconstruct(im, vqgan, args)
    #
    # fig, axes = plt.subplots(ncols=3)
    # for ax, name, img in zip(axes, ['orig', 'openai', 'taming'], [im, openai_recon, vqgan_recon]):
    #     ax.imshow(im)
    #     ax.set_axis_off()
    #     ax.set_title(name)
    # fig.suptitle(args.vae_type)
    # plt.tight_layout()
    # plt.show()
    # exit()


    dalle_params['num_frames'] = len(ds)
    dalle_params['num_videos'] = ds.num_classes
    assert len(ds) > 0, 'dataset is empty'
    print(f'{len(ds)} image-text pairs found for training')
    num_workers = 0 if is_debug_mode() else 10
    dl = MultiEpochsDataLoader(ds, batch_size=args.batch_size, shuffle=is_shuffle, drop_last=False, sampler=None,
                               pin_memory=True, num_workers=num_workers)
    print(f'using {num_workers} workers')
    # initialize DALL-E
    if args.generator == 'lord':
        generator = LordGenerator(vae=vae, **dalle_params)
    elif args.generator == 'transformer':
        generator = DALLE(vae=vae, **dalle_params)
    elif args.generator == 'stylegan':
        generator = StyleGANGenerator(vae=vae, **dalle_params)
    print(f"the generator has {count_parameters(generator):,} parameters")

    if args.fp16:
        generator = generator.half()
    generator = generator.cuda(args.device)
    if exists(args.dalle_path):
        # no_last_conv = {k: v for k, v in weights.items() if 'last_conv' not in k}
        generator.load_state_dict(weights, strict=False)
    # optimizer
    generator_opt = Adam(get_trainable_params(generator.generator), lr=args.learning_rate)
    latents_opt = Adam(get_trainable_params(generator.per_video_emb) + get_trainable_params(generator.per_frame_emb),
                       lr=args.learning_rate * 10)

    if args.lr_decay:
        generator_scheduler = ReduceLROnPlateau(
            generator_opt,
            mode="min",
            factor=0.5,
            patience=20,
            cooldown=20,
            min_lr=1e-6,
            verbose=True,
        )
        latents_scheduler = ReduceLROnPlateau(
            latents_opt,
            mode="min",
            factor=0.5,
            patience=20,
            cooldown=20,
            min_lr=1e-6 * 10,
            verbose=True,
        )
    else:
        # keeping the LR constant, with the same API
        generator_scheduler = ExponentialLR(generator_opt, gamma=1)
        latents_scheduler = ExponentialLR(latents_opt, gamma=1)

    if not args.extract_codes:
        # experiment tracker
        model_config = vars(args)

        run = wandb.init(
            project=args.wandb_name,  # 'dalle_train_transformer' by default
            resume=False,
            config=model_config,
        )

    codes_filename = f'codes_tensor_{args.vae_type}_{args.dataset}.pt'
    if not args.extract_codes:
        loaded_codes = torch.load(codes_filename, map_location=f'cpu')
    PRINT_TIME = 500
    # training
    CODESIZE_MAPPER = {'openai': 1024, 'taming': 256, 'geo': 256}
    NUM_CODES = CODESIZE_MAPPER[args.vae_type]
    all_codes = torch.zeros(len(ds), NUM_CODES, dtype=torch.long)

    for epoch in range(args.epochs):
        for i, (images, video_indices, frame_indices) in enumerate(dl):
            if i % PRINT_TIME == 0:
                t = time.time()
            if args.fp16:
                images = images.half()
            images = images.cuda(args.device)
            video_indices = video_indices.cuda(args.device).long()
            if args.extract_codes:
                codebook_indices = vae.get_codebook_indices(images).detach().cpu()
                all_codes[frame_indices] = codebook_indices
                print(f'finished is {i * args.batch_size} frames out of {len(ds)}')
                # if i > 10:
                #     break
                continue
            # text, images = map(lambda t: t.cuda(DEVICE), (text, images))
            rel_codes = loaded_codes[frame_indices].cuda(args.device)
            if args.cheating: # in cars, you can use the actual labels as latents
                frame_indices = torch.tensor(ds.contents[frame_indices.cpu()]).cuda()
            frame_indices = frame_indices.cuda(args.device)
            ce_loss, reg_loss = generator(images, video_indices, frame_indices, return_loss=True, rel_codes=rel_codes)
            loss = ce_loss + reg_loss

            loss.backward()
            clip_grad_norm_(generator.parameters(), args.clip_grad_norm)
            generator_opt.step()
            generator_opt.zero_grad()
            latents_opt.step()
            latents_opt.zero_grad()
            log = {}

            if i % PRINT_TIME == 0:
                print(epoch, i, f'loss - {loss.item():.3f}')
                lr = generator_opt.param_groups[0]['lr']
                log = {
                    **log,
                    'epoch': epoch,
                    'iter': i,
                    'loss': loss.item(),
                    'ce_loss': ce_loss.item(),
                    'reg_loss': reg_loss.item(),
                    'lr': lr
                }

            if ((epoch + 1) % 50) == 0 and i == 0:
                save_model('checkpoints/' + args.dalle_output_file_name + ".pt", dalle_params, vae_params, generator)

            if ((epoch + 1) % 10) == 0 and i == 0:
                log = log_outputs(video_indices, frame_indices, generator, vae, log, images, epoch, args, rel_codes)

                # check classification_accuracy
                if args.dataset == 'cars':
                    log = check_disentanglement(args, ds, generator, log)

            if i % PRINT_TIME == 9:
                sample_per_sec = args.batch_size * 10 / (time.time() - t)
                log["sample_per_sec"] = sample_per_sec
                print(epoch, i + 1, f'sample_per_sec - {sample_per_sec:.3f}')
            log = {
                **log,
            }
            wandb.log(log)

        if args.extract_codes:
            torch.save(all_codes, codes_filename)
            exit()
        if args.lr_decay:
            # Scheduler is automatically progressed after the step when
            # using DeepSpeed.
            generator_scheduler.step(loss)
            latents_scheduler.step(loss)
        # save_model('checkpoints/' + args.dalle_output_file_name + ".pt", dalle_params, vae_params, generator)
    save_model('checkpoints/' + args.dalle_output_file_name + ".pt", dalle_params, vae_params, generator)
    wandb.save('checkpoints/' + args.dalle_output_file_name + ".pt")
    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file('checkpoints/' + args.dalle_output_file_name + ".pt")
    run.log_artifact(model_artifact)
    wandb.finish()


def check_disentanglement(args, ds, generator, log):
    num_samples = len(ds)
    if args.cnn_generator:
        X = generator.per_frame_emb.embedding(torch.arange(num_samples).to(args.device)).detach().cpu().numpy()
    else:
        X = generator.per_frame_emb(torch.arange(num_samples).to(args.device)).detach().cpu().numpy()

    y = ds.targets
    n = len(X)
    perm = np.random.permutation(n)
    train, test = perm[n//2:], perm[:n//2]
    import sklearn.linear_model
    logistic = sklearn.linear_model.LogisticRegression(random_state=0, n_jobs=4).fit(X[train], y[train])
    perceptron = sklearn.linear_model.Perceptron(random_state=0, n_jobs=4).fit(X[train], y[train])
    log['accuracy_logistic'] = (logistic.predict(X[test]) == y[test]).mean()
    log['accuracy_perceptron'] = (perceptron.predict(X[test]) == y[test]).mean()
    return log


if __name__ == '__main__':
    main()
