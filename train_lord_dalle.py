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
from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE1024, DiscreteVAE, DALLE
from dalle_pytorch.lord_generator import LordGenerator
from cars import Cars3D
from ffhq import FFHQ
# argument parsing
from real_estate import RealEstate10K
from multi_dataloader import MultiEpochsDataLoader
import sys

torch.hub.set_dir('/cs/labs/daphna/avihu.dekel/.cache/')


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
    parser.add_argument('--taming', dest='taming', action='store_true')
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
    train_group.add_argument('--batch_size', default=32, type=int, help='Batch size')
    train_group.add_argument('--ga_steps', default=1, type=int,
                             help='Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')
    train_group.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
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


def log_outputs(video_indices, frame_indices, dalle, vae, log, images, epoch):
    fr = torch.cat([frame_indices[:2], torch.flip(frame_indices[:2], (0,))], dim=0)
    vd = torch.cat([video_indices[:2], video_indices[:2]], dim=0)
    recon = dalle.generate_images(frame_index=fr,
                                  video_index=vd,
                                  filter_thres=0.99).cpu()  # topk sampling at 0.9
    dalle_recon = recon[:2]
    crossovers = recon[2:]

    codes = vae.get_codebook_indices(images[:2])
    vae_recon = vae.decode(codes).detach().cpu()


    grid = torchvision.utils.make_grid(
        torch.cat([images[:2].cpu(), vae_recon, dalle_recon, crossovers], dim=0), nrow=2)
    log['all'] = wandb.Image(grid)
    log['vae'] = wandb.Image(torchvision.utils.make_grid(vae_recon, nrow=1))
    log['dalle'] = wandb.Image(torchvision.utils.make_grid(dalle_recon, nrow=1))
    log['image'] = wandb.Image(torchvision.utils.make_grid(images[:2], nrow=1))
    # plot tsne
    # vid1_id, vid2_id = video_indices[:2].cpu().numpy()
    # frame1_id, frame2_id = frame_indices[:2].cpu().numpy()
    # frame_embs = list(dalle.per_frame_emb.parameters())[0].detach().cpu().numpy()[:-1]
    # frame_embs = frame_embs / np.linalg.norm(frame_embs, axis=1, keepdims=True)
    # r = np.random.RandomState(10)
    # rand_subset = r.permutation(len(frame_embs))[:10000]
    # tsne = MulticoreTSNE.MulticoreTSNE(n_jobs=10)
    # from sklearn.decomposition import PCA
    # tsne_emb = tsne.fit_transform(frame_embs[rand_subset])
    # pca_emb = PCA().fit_transform(frame_embs[rand_subset])
    # for emb_name, emb in [('Tsne', tsne_emb), ('PCA', pca_emb)]:
    #     for c_name, c in [('class', ds.classes), ('content', ds.contents)]:
    #         fig, ax = plt.subplots()
    #         ax.scatter(emb[:,0], emb[:,1], c=c[rand_subset], cmap='hsv', alpha=0.5)
    #         ax.set_title(f'{emb_name} - colored by {c_name}')
    #         fig.show()
    # wandb.log({'colorByClass': class_fig, 'colorByContent': content_fig})

    # if (epoch % 20) == 0:
    #     INTERPOLATION_NUM = 20
    #     first_index = video_indices[:1] * 0
    #     interpolations = dalle.interpolate_images(first_index,
    #                                               first_index + 5,
    #                                               video_indices[:1], filter_thres=0.99,
    #                                               interpolation_num=INTERPOLATION_NUM)
    #     interpolations = (interpolations * 255).cpu().to(torch.uint8)
    #     forward_backward_iterpolated = torch.cat([interpolations, torch.flip(interpolations, dims=(0,))], dim=0)
    #     log['interpolate'] = wandb.Video(forward_backward_iterpolated)


def main():
    args = parse_args()
    if exists(args.dalle_path):
        dalle_path = Path(args.dalle_path)

        assert dalle_path.exists(), 'DALL-E model file does not exist'
        loaded_obj = torch.load(str(dalle_path), map_location='cpu')

        dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
        dalle_params['regularization'] = args.regularization
        dalle_params['sigma'] = args.sigma
        if vae_params is not None:
            vae = DiscreteVAE(**vae_params)
        else:
            vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
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
            vae_klass = OpenAIDiscreteVAE if not args.taming else VQGanVAE1024
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
    elif args.dataset == 'ffhq':
        ds = FFHQ(root='/cs/labs/peleg/avivga/data/images/ffhq/imgs-x256',
                  transform=T.Compose([T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                       T.Resize(vae.image_size),
                                       T.ToTensor()
                                       ])
                  )
    else:
        raise ValueError('unknown dataset')

    dalle_params['num_frames'] = len(ds)
    dalle_params['num_videos'] = ds.num_classes
    assert len(ds) > 0, 'dataset is empty'
    print(f'{len(ds)} image-text pairs found for training')
    num_workers = 0 if is_debug_mode() else 10
    dl = MultiEpochsDataLoader(ds, batch_size=args.batch_size, shuffle=is_shuffle, drop_last=False, sampler=None, pin_memory=True,
                    num_workers=num_workers)
    print(f'using {num_workers} workers')
    # initialize DALL-E
    dalle = DALLE(vae=vae, **dalle_params)
    dalle = LordGenerator(vae=vae, **dalle_params)
    if args.fp16:
        dalle = dalle.half()
    dalle = dalle.cuda(args.device)
    if exists(args.dalle_path):
        dalle.load_state_dict(weights)
    # optimizer
    opt = Adam(get_trainable_params(dalle), lr=args.learning_rate)

    if args.lr_decay:
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
        )
    else:
        # keeping the LR constant, with the same API
        scheduler = ExponentialLR(opt, gamma=1)

    if not args.extract_codes:
        # experiment tracker

        model_config = vars(args)

        run = wandb.init(
            project=args.wandb_name,  # 'dalle_train_transformer' by default
            resume=False,
            config=model_config,
        )

    codes_filename = f'codes_tensor_taming_{args.dataset}.pt'
    if not args.extract_codes:
        loaded_codes = torch.load(codes_filename, map_location=f'cuda:{args.device}')

    # training
    NUM_CODES = 256
    all_codes = torch.zeros(len(ds), NUM_CODES, dtype=torch.long)

    for epoch in range(args.epochs):
        cur_epoch_codes = []
        for i, (images, video_indices, frame_indices) in enumerate(dl):
            if i % 10 == 0:
                t = time.time()
            if args.fp16:
                images = images.half()
            images = images.cuda(args.device)
            video_indices = video_indices.cuda(args.device)
            frame_indices = frame_indices.cuda(args.device)

            if args.extract_codes:
                codebook_indices = vae.get_codebook_indices(images).detach().cpu()
                all_codes[frame_indices] = codebook_indices
                print(f'finished is {i * args.batch_size} frames out of {len(ds)}')
                # if i > 10:
                #     break
                continue
            # text, images = map(lambda t: t.cuda(DEVICE), (text, images))
            rel_codes = loaded_codes[frame_indices].cuda(args.device)
            ce_loss, reg_loss = dalle(images, video_indices, frame_indices, return_loss=True, rel_codes=rel_codes)
            loss = ce_loss + reg_loss

            loss.backward()
            clip_grad_norm_(dalle.parameters(), args.clip_grad_norm)
            opt.step()
            opt.zero_grad()

            # Collective loss, averaged
            avg_loss = loss

            log = {}

            if i % 10 == 0:
                print(epoch, i, f'loss - {avg_loss.item()}')
                lr = opt.param_groups[0]['lr']
                log = {
                    **log,
                    'epoch': epoch,
                    'iter': i,
                    'loss': avg_loss.item(),
                    'ce_loss': ce_loss,
                    'reg_loss': reg_loss,
                    'lr': lr
                }

            if ((epoch + 1) % 50) == 0 and i == 0:
                save_model(args.dalle_output_file_name + ".pt", dalle_params, vae_params, dalle)

            if (epoch % 10) == 0 and i == 0:
                log_outputs(video_indices, frame_indices, dalle, vae, log, images, epoch)

                wandb.save(f'./dalle.pt')

                log = {
                    **log,
                }

            if i % 10 == 9:
                sample_per_sec = args.batch_size * 10 / (time.time() - t)
                log["sample_per_sec"] = sample_per_sec
                print(epoch, i, f'sample_per_sec - {sample_per_sec}')

            wandb.log(log)
        if args.extract_codes:
            torch.save(all_codes, codes_filename)
            exit()
        if args.lr_decay:
            # Scheduler is automatically progressed after the step when
            # using DeepSpeed.
            scheduler.step(loss)
        save_model(args.dalle_output_file_name + ".pt", dalle_params, vae_params, dalle)
    save_model(args.dalle_output_file_name + ".pt", dalle_params, vae_params, dalle)
    wandb.save(args.dalle_output_file_name + ".pt")
    model_artifact = wandb.Artifact('trained-dalle', type='model', metadata=dict(model_config))
    model_artifact.add_file(args.dalle_output_file_name + ".pt")
    run.log_artifact(model_artifact)
    wandb.finish()


if __name__ == '__main__':
    main()
