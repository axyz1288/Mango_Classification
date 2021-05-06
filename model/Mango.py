from torch.utils.data import Dataset
from PIL import Image
import math
import pandas as pd
import torch

class Mango(Dataset):
    def __init__(self, csv_file, data_dir, label2idx, transform=None, trainable=True):
        self.data_dir = data_dir
        self.label2idx = label2idx
        self.transform = transform
        self.trainable = trainable
        self.dict = pd.read_csv(csv_file)
        self.data_key = self.dict.keys()[0]
        self.label_key = self.dict.keys()[1]
        
    def __len__(self):
        return len(self.dict[self.data_key])
    
    def __getitem__(self, idx):
        path = self.data_dir + '/' + self.dict[self.data_key][idx]
        img = Image.open(path)        
        if(self.transform != None):
            img = self.transform(img)
            
        if(self.trainable == True):
            label = self.label2idx[self.dict[self.label_key][idx]]
            return {'data': img, 'label':label}
        else:
            return {'data': img}