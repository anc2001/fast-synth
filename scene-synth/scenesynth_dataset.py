from torch.utils.data import Dataset
import torch


class SceneSynthDataset(Dataset):
    def __init__(self, scenes):
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        input_img = torch.tensor(scene["input_img"], dtype=torch.float32)
        t_cat = torch.tensor(scene["t_cat"], dtype=torch.long)
        catcount = torch.tensor(scene["catcount"], dtype=torch.float32)
        return input_img, t_cat, catcount
