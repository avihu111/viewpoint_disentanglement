import os

import numpy as np
from torchvision.datasets import ImageFolder, DatasetFolder


class FFHQ(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, FRAMES_PER_VIDEO = 100, return_indices=True, NUM_VIDEOS=100):
        super(FFHQ, self).__init__(os.path.join(root), transform, target_transform)
        self.num_classes = len(np.unique(self.targets))
        new_indices = []
        video_counts = np.unique(self.targets, return_counts=True)[1]
        video_ids_by_size = np.argsort(video_counts)[::-1][:NUM_VIDEOS]
        r = np.random.RandomState(1234)
        for i in video_ids_by_size:
            video_indices = (np.array(self.targets) == i).nonzero()[0]
            sampled_frames = r.permutation(video_indices)[:FRAMES_PER_VIDEO]
            new_indices.append(sampled_frames)
        self.indices = np.concatenate(new_indices)
        self.return_indices = return_indices

    def __getitem__(self, item):
        supset_index = self.indices[item]
        x, y = super(FFHQ, self).__getitem__(supset_index)
        if self.return_indices:
            return x, y, item
        return x

    def __len__(self):
        return len(self.indices)
