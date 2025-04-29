import torch.nn as nn
import torch.nn.functional as F
import torch
from .video_transformer import VideoTransformer


class AggregationTransformer(nn.Module):
    def __init__(self, config, train_ds):
        super().__init__()
        self.fine_only = train_ds.fine_only

        if config.model.pretrained:
            self.video_transformer = nn.Identity()
            mlp_input_dim = config.model.pretrained_dim
        else:
            self.video_transformer = VideoTransformer(config)
            mlp_input_dim = config.model.video_transformer.embedding_dim
        

        num_fine_classes = train_ds.num_fine_classes

        self.common_mlp = nn.Sequential(nn.Linear(mlp_input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1), 
                        nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                        nn.Linear(256, num_fine_classes)
                        )
        if not self.fine_only:
            num_coarse_classes = train_ds.num_coarse_classes
            self.coarse_mlp = nn.Sequential(nn.Linear(mlp_input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1), 
                        nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                        nn.Linear(256, num_coarse_classes)
                        )
    

    def forward(self, x):
        fts = self.video_transformer(x)

        fine = self.common_mlp(fts)

        if not self.fine_only:
            coarse = self.coarse_mlp(fts)
        else:
            coarse = torch.tensor([0])

        return coarse, fine