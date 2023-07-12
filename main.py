from torch import sqrt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from architectures import SimpleCNN
from datasets import RandomImagePixelationDataset
from utils import stack
from sklearn.model_selection import train_test_split


def train_step(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for pixelated_images, known_arrays, target_arrays, _ in tqdm(train_loader, desc='Training', leave=False):
        pixelated_images = pixelated_images.to(device)
        known_arrays = known_arrays.to(device)
        target_arrays = target_arrays.to(device)

        input_data = torch.cat((pixelated_images, known_arrays), dim=1)
        outputs = model(input_data)

        loss = torch.sqrt(F.mse_loss(outputs * ~known_arrays, target_arrays * ~known_arrays))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def eval_step(model, val_loader, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for pixelated_images, known_arrays, target_arrays, _ in val_loader:
            pixelated_images = pixelated_images.to(device)
            known_arrays = known_arrays.to(device)
            target_arrays = target_arrays.to(device)

            input_data = torch.cat((pixelated_images, known_arrays), dim=1)
            outputs = model(input_data)

            loss = torch.sqrt(F.mse_loss(outputs * ~known_arrays, target_arrays * ~known_arrays))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


if __name__ == "__main__":
    torch.random.manual_seed(0)
    image_dir = r"/Users/darakuklina/Downloads/photo_data"
    width_range = (4, 32)
    height_range = (4, 32)
    size_range = (4, 16)
    im_shape = 64
    dataset = RandomImagePixelationDataset(image_dir, width_range, height_range, size_range, im_shape)

    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001

    train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
    val_size = len(dataset) - train_size  # 20% of the dataset for validation
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    model = SimpleCNN(n_in_channels=2, n_hidden_layers=5, n_kernels=64, kernel_size=3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=stack)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=stack)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, optimizer, device)
        print(f'Epoch {epoch + 1}/{num_epochs} | Training Loss: {train_loss:.4f}')

        val_loss = eval_step(model, val_loader, device)
        print(f'Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'depixelation_model7.pth')
"""

    model = SimpleCNN(n_in_channels=2, n_hidden_layers=5, n_kernels=64, kernel_size=3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=stack)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=stack)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
"""

"""
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for pixelated_images, known_arrays, target_arrays, _ in tqdm(train_loader,
                                                                     desc=f'Epoch {epoch + 1}/{num_epochs}',
                                                                     leave=False):
            pixelated_images = pixelated_images.to(device)
            known_arrays = known_arrays.to(device)
            target_arrays = target_arrays.to(device)

            input_data = torch.cat((pixelated_images, known_arrays), dim=1)
            outputs = model(input_data)
            #outputs = model(pixelated_images)

            loss = torch.sqrt(F.mse_loss(outputs * ~known_arrays, target_arrays * ~known_arrays))
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} | Training Loss: {epoch_loss:.4f}')
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pixelated_images, known_arrays, target_arrays, _ in val_loader:
            pixelated_images = pixelated_images.to(device)
            known_arrays = known_arrays.to(device)
            target_arrays = target_arrays.to(device)

            input_data = torch.cat((pixelated_images, known_arrays), dim=1)
            outputs = model(input_data)
            #outputs = model(pixelated_images)
            loss = torch.sqrt(F.mse_loss(outputs * ~known_arrays, target_arrays * ~known_arrays))
            print(loss)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
"""
