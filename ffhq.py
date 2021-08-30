import os

import numpy as np
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
import json


class FFHQ(VisionDataset):

    def __init__(self, root='/cs/labs/peleg/avivga/data/images/ffhq/', img_size=256, transform=None,
                 target_transform=None, small=False):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self._base_dir = root
        self.img_size = img_size
        if os.path.isfile('ffhq_age.npy') and os.path.isfile('ffhq_gender.npy'):
            age = np.load('ffhq_age.npy')
            gender = np.load('ffhq_gender.npy')
        else:
            age, gender = self.create_attributes_dict()

        valid_idx = (age != -1)
        self.valid_indices = valid_idx.nonzero()[0]
        self.ages = age[valid_idx]
        self.gender = gender[valid_idx]
        self.cls = age[valid_idx].astype(np.int16) // 10
        self.num_classes = self.cls.max() + 1
        if small:
            new_indices = []
            r = np.random.RandomState(1234)
            for i in range(self.num_classes):
                video_indices = (np.array(self.cls) == i).nonzero()[0]
                sampled_frames = r.permutation(video_indices)[:2000]
                new_indices.append(sampled_frames)
            self.indices = np.concatenate(new_indices)
        else:
            self.indices = np.arange(len(self.cls))
        self.return_indices = True

    def __getitem__(self, item):
        valid_idx = self.indices[item]
        idx = self.valid_indices[valid_idx]
        img_path = os.path.join(self._base_dir, 'imgs', 'img{:08d}.png'.format(idx))
        img = default_loader(img_path)
        img = self.transform(img)

        if self.return_indices:
            return img, self.cls[valid_idx], item
        return img

    def __len__(self):
        return len(self.indices)


    def create_attributes_dict(self):
        LIMIT = 70000  # should be 70000 tops
        age = np.full(shape=(LIMIT,), fill_value=-1, dtype=np.float32)
        gender = np.full(shape=(LIMIT,), fill_value=-1, dtype=np.int16)
        img_ids = np.arange(LIMIT)
        for i in tqdm(img_ids):
            features_path = os.path.join('/cs/labs/daphna/avihu.dekel/ffhq-features-dataset/json',
                                         '{:05d}.json'.format(i))
            with open(features_path, 'r') as features_fp:
                features = json.load(features_fp)
                if len(features) != 0:
                    age[i] = features[0]['faceAttributes']['age']
                    gender[i] = (features[0]['faceAttributes']['gender'] == 'male')

        np.save('ffhq_age.npy', age)
        np.save('ffhq_gender.npy', gender)
        return age, gender


