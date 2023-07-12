import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import Grayscale
from architectures import SimpleCNN
from datasets import RandomImagePixelationDataset
from utils import stack
"""
with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)
    pixelated_images_dataset = torch.from_numpy(np.array(test_set['pixelated_images']))
    known_arrays_dataset = torch.from_numpy(np.array(test_set['known_arrays']))

    pixelated_images_loader = DataLoader(dataset=TensorDataset(pixelated_images_dataset), batch_size=1, shuffle=False)
    known_arrays_loader = DataLoader(dataset=TensorDataset(known_arrays_dataset), batch_size=1, shuffle=False)
"""
# Load the trained model
model_path = 'depixelation_model5.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(n_in_channels=1, n_hidden_layers=5, n_kernels=32, kernel_size=3)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
torch.random.manual_seed(0)
image_dir = r"/Users/darakuklina/Downloads/photo_data"
width_range = (4, 32)
height_range = (4, 32)
size_range = (4, 16)
im_shape = 64
dataset = RandomImagePixelationDataset(image_dir, width_range, height_range, size_range, im_shape)
train_size = int(0.8 * len(dataset))  # 80% of the dataset for training
val_size = len(dataset) - train_size  # 20% of the dataset for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
model.to(device)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=stack)

for pixelated_images, known_arrays, target_arrays, _ in val_loader:
    pixelated_images = pixelated_images.to(device)
    target_arrays = torch.stack(target_arrays).to(device).float()
    import matplotlib.pyplot as plt

    target_img = target_arrays[0]  # Assuming batch size is 1

    # Convert the output tensor to a numpy array and squeeze the dimensions if necessary
    target_img = target_img.detach().cpu().numpy().squeeze()

    # Rescale the values to the range [0, 1]
    target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())

    # Display the image
    plt.imshow(target_img, cmap='gray')
    plt.axis('off')
    plt.show()


    pixel_img = pixelated_images[0]  # Assuming batch size is 1

    # Convert the output tensor to a numpy array and squeeze the dimensions if necessary
    pixel_img = pixel_img.detach().cpu().numpy().squeeze()

    # Rescale the values to the range [0, 1]
    pixel_img = (pixel_img - pixel_img.min()) / (pixel_img.max() - pixel_img.min())

    # Display the image
    plt.imshow(pixel_img, cmap='gray')
    plt.axis('off')
    plt.show()


    outputs = model(pixelated_images)

    # Assuming the output tensor is named "output"
    output = outputs[0]  # Assuming batch size is 1

    # Convert the output tensor to a numpy array and squeeze the dimensions if necessary
    output = output.detach().cpu().numpy().squeeze()

    # Rescale the values to the range [0, 1]
    output = (output - output.min()) / (output.max() - output.min())

    # Display the image
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.show()
    print("l")


# Iterate over the pixelated images and known arrays
# predictions = []
# for image, known_array in zip(pixelated_images, known_arrays):
#     # Convert image and known_array to tensors
#     image = image.squeeze(0)  # Add batch dimension
#     known_array = known_array.squeeze(0)  # Add batch dimension
#
#     # Make a prediction using the model
#     output = model(image)
#
#     # Flatten the predicted pixel values using the known array as a mask
#     prediction = output.squeeze().detach().numpy()  # Convert tensor to NumPy array
#     prediction[known_array.squeeze().bool()] = 0  # Set the pixel values in the known area to 0
#
#     # Collect the predictions
#     predictions.append(prediction.astype(np.uint8))
#
# # Print the predictions
# for prediction in predictions:
#     print(prediction)