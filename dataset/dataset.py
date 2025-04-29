import torch
from torch.utils.data import Dataset
import os, pickle
import numpy as np

import pandas as pd

datasets = ["tsu", "etri"]

class HierarchicalDataset(Dataset):
    def __init__(self, config, split="train"):
        if config.data.dataset not in datasets:
            raise ValueError("Unknown dataset: {}".format(config.data.dataset))
        self.dataset = config.data.dataset
        self.inputs = []
        self.anno_path = os.path.join(config.data.anno_path, config.data.dataset)
        self.enc_steps = config.data.enc_steps
        self.split = split
        self.rgb_fts = config.data.rgb_features
        self.fts_model = config.data.fts_model
        self.data_path = config.data.data_path
        self.fine_only_ = config.data.coarse_action_source is None

        with open(os.path.join(self.anno_path, split + "_split.pkl"), "rb") as f:
            data = pickle.load(f)


        with open(os.path.join(self.anno_path, "fine-grained-classes.txt"), "r") as f:
            fine2id = {}
            for i, line in enumerate(f):
                fine2id[line.strip()] = i
        self.num_fine = len(fine2id)

        self.num_coarse = 0
        if not self.fine_only:
            with open(os.path.join(self.anno_path, "coarse-" + config.data.coarse_action_source + ".txt"), "r") as f:
                coarse2id = {}
                for i, line in enumerate(f):
                    coarse2id[line.strip()] = i
            self.num_coarse = len(coarse2id)

            df = pd.read_csv(os.path.join(self.anno_path, "coarse2fine-" + config.data.coarse_action_source + ".csv"), sep=",")
            df['coarse-grained'] = df['coarse-grained'].astype(str)
            fine2coarse = {}

            for i, row in df.iterrows():
                fine2coarse[fine2id[row["fine-grained"]]] = coarse2id[row["coarse-grained"]]
        for video,anno in data.items():
            for a in anno:
                anno_id, start, end, _, fine = a
                if self.dataset == "tsu":
                    ft_name = f"A{fine[0]}_S{start}_E{end}_V{video.split('.')[0]}"
                elif self.dataset == "etri":
                    ft_name = video
                length = end - start
                if length <= self.enc_steps:
                    continue
                if length > self.enc_steps:
                    start = np.random.randint(0, length - self.enc_steps)
                else:
                    start = 0

                coarse = []
                if not self.fine_only:
                    if config.data.dataset == "tsu":
                        for f in fine:
                            coarse.append(fine2coarse[f])
                    else:
                        coarse.append(fine2coarse[fine])
                self.inputs.append((video.split(".")[0], start, ft_name, coarse, fine, anno_id))

        if config.model.pretrained:
            with open(config.model.pretrained, "rb") as f:
                self.features = pickle.load(f)
        else:
            self.features = None

    @property
    def num_coarse_classes(self):
        return self.num_coarse

    @property
    def num_fine_classes(self):
        return self.num_fine

    @property
    def fine_only(self):
        return self.fine_only_

    
    def __getitem__(self, index):
        video_name, start, ft_name, coarse, fine, anno_id = self.inputs[index]
        rgb_t = torch.tensor([0])
        
        if self.features:
            rgb_t = torch.from_numpy(self.features[ft_name])
        else:
            file = os.path.join(self.data_path, "rgb", self.rgb_fts, video_name + "_rgb.npy")
            if not os.path.exists(file):
                raise ValueError("File not found: {}".format(file))
            features = np.load(file)
            rgb_t = torch.from_numpy(features[start:start+self.enc_steps])
            del features
    
        coarse_t = torch.tensor([0])
        if self.dataset == "tsu":
            if not self.fine_only:
                coarse_t = np.zeros(self.num_coarse)
                coarse_t[coarse] = 1
                coarse_t = torch.from_numpy(coarse_t)

            fine_t = np.zeros(self.num_fine)
            fine_t[fine] = 1
            fine_t = torch.from_numpy(fine_t)
        else:
            if not self.fine_only:
                coarse_t = coarse[0]
            fine_t = fine
        
        return rgb_t, coarse_t, fine_t
    
    def __len__(self):
        return len(self.inputs)