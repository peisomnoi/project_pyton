import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import RandomImagePixelationDataset
from utils import stack

torch.random.manual_seed(0)
image_dir = r"/Users/darakuklina/Downloads/photo_data"
width_range = (4, 32)
height_range = (4, 32)
size_range = (4, 16)
im_shape = 64
batch_size = 32
dataset = RandomImagePixelationDataset(image_dir, width_range, height_range, size_range, im_shape)
train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
val_size = len(dataset) - train_size  # 20% of the dataset for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=stack)
data_batch = next(iter(train_loader))
data = torch.cat(data_batch, dim=0).tolist()
data = np.array(data)
mean = np.mean(data)
std = np.std(data)


plt.hist(data, bins='auto')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Data Distribution')
plt.show()


print("Mean:", mean)
print("Standard Deviation:", std)
