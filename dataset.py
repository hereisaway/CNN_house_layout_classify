import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir, self.label_dir)
        self.img_path=os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img=Image.open(img_item_path)
        label=int(self.label_dir)
        img = img.convert('L')
        if self.transform:
            img=self.transform(img)

        return img,label

# img,label=ants_dataset[0]
# img.show()

if __name__=="__main__":
    root_dir = "data"
    d1_dir = "0"
    d1_dataset = MyDataset(root_dir, d1_dir)
    d2_dir = "1"
    d2_dataset = MyDataset(root_dir, d2_dir)
    d3_dir = "2"
    d3_dataset = MyDataset(root_dir, d3_dir)
    dataset = d1_dataset + d2_dataset + d3_dataset
    img,label=dataset[12000]
    img.show()