from numpy import dtype
import torch
from torch.utils.data import Dataset
import pandas as pd

top_song_count = 2000

class TopSongsTrain(Dataset):
    def __init__(self, file_name):
        pl_top_trk_df = pd.read_csv(file_name, names=['pid', 'followers'] + [i for i in range(0, top_song_count)])

        x=pl_top_trk_df[[i for i in range(0, top_song_count)]].values
        y=pl_top_trk_df[['pid']].values

        self.x_train=torch.tensor(x)
        self.y_train=torch.tensor(y)

        def __len__(self):
            return len(self.y_train)

        def __getitem__(self, idx):
            return self.x_train[idx], self.y_train[idx]

d = TopSongsTrain('data.csv')