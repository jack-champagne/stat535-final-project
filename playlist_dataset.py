from numpy import dtype
import torch
from torch.utils.data import Dataset
import pandas as pd

top_song_count = 2000

class TopSongsTrain(Dataset):
    def __init__(self, file_name):
        pl_top_trk_df = pd.read_json()

        x=pl_top_trk_df[:,0:top_song_count].values
        y=pl_top_trk_df[:top_song_count].values

        self.x_train=torch.tensor(x, dtype=torch.bool)
        self.y_train=torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.y_train)

        def __getitem(self, idx):
            return self.x_train[idx], self.y_train[idx]