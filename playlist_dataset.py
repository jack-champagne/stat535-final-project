from numpy import dtype
import torch
from torch.utils.data import Dataset
import pandas as pd

top_song_count = 2000
file_name = 'data.csv'
pl_top_trk_df = pd.read_csv(file_name, names=['pid', 'followers'] + [i for i in range(0, top_song_count)], header=0)
test = pl_top_trk_df.sample(frac=0.1, axis=0)
train = pl_top_trk_df.drop(index=test.index)

class TopSongsTrain(Dataset):
    def __init__(self):

        x=train[[i for i in range(0, top_song_count)]].values
        y=train[['followers']].values

        self.x_train=torch.tensor(x, dtype=torch.float32)
        self.y_train=torch.log(torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

class TopSongsTest(Dataset):
    def __init__(self):
        pl_top_trk_df = pd.read_csv(file_name, names=['pid'] + [i for i in range(0, top_song_count)])

        x=test[[i for i in range(0, top_song_count)]].values
        y=test[['followers']].values

        self.x_test=torch.tensor(x, dtype=torch.float32)
        self.y_test=torch.log(torch.tensor(y, dtype=torch.float32))

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        ## Remove a few hots to cross validate...
        return self.x_test[idx], self.y_test[idx]