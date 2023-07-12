import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToPILImage
from architectures import SimpleCNN
from utils import stack
from serialization import serialize
import matplotlib.pyplot as plt
"""
# Load the test set
with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)
    pixelated_images_dataset = torch.from_numpy(np.array(test_set['pixelated_images']))
    known_arrays_dataset = torch.from_numpy(np.array(test_set['known_arrays']))

    test_loader = DataLoader(dataset=TensorDataset(pixelated_images_dataset, known_arrays_dataset),
                             batch_size=1,
                             shuffle=False)

# Load the trained model
model_path = 'depixelation_model5.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(n_in_channels=1, n_hidden_layers=5, n_kernels=32, kernel_size=3)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

predictions = []
for pixelated_images, known_arrays in test_loader:
    pixelated_images = pixelated_images.to(device)

    # Get the output from the model
    outputs = model(pixelated_images)

    # Flatten the output tensor using torch.masked_select or output[~known_arrays]
    flattened_output = torch.masked_select(outputs[0], ~known_arrays[0])

    # Convert the flattened output to np.uint8 and append it to the list of predictions
    prediction = flattened_output.detach().cpu().numpy().astype(np.uint8)
    predictions.append(prediction)

# Serialize the predictions and save to a file
serialize(predictions, "predictions.bin")
"""
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToPILImage
from architectures import SimpleCNN
from utils import stack
from serialization import serialize
import matplotlib.pyplot as plt
"""
# Load the test set
with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)
    pixelated_images_dataset = torch.from_numpy(np.array(test_set['pixelated_images']))
    known_arrays_dataset = torch.from_numpy(np.array(test_set['known_arrays']))

    test_loader = DataLoader(dataset=TensorDataset(pixelated_images_dataset, known_arrays_dataset),
                             batch_size=1,
                             shuffle=False)

# Load the trained model
model_path = 'depixelation_model7.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#n_in_channels=1, n_hidden_layers=5, n_kernels=32, kernel_size=3 model5
#n_in_channels=1, n_hidden_layers=5, n_kernels=64, kernel_size=3 model6
#n_in_channels=2, n_hidden_layers=5, n_kernels=64, kernel_size=3 model7
model = SimpleCNN(n_in_channels=2, n_hidden_layers=5, n_kernels=64, kernel_size=3)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

predictions = []
for pixelated_images, known_arrays in test_loader:
    pixelated_images = pixelated_images.to(device)

    # Get the output from the model
    outputs = model(pixelated_images)

    # Flatten the output tensor using torch.masked_select or output[~known_arrays]
    flattened_output = torch.masked_select(outputs[0], ~known_arrays[0])

    # Convert the flattened output to np.uint8 and append it to the list of predictions
    prediction = flattened_output.detach().cpu().numpy().astype(np.uint8)
    predictions.append(prediction)

    # Visualize the target image
    target_img = known_arrays[0].detach().cpu().numpy().squeeze().astype(np.float32)
    target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
    plt.imshow(target_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Visualize the pixelated image
    pixel_img = pixelated_images[0].detach().cpu().numpy().squeeze()
    pixel_img = (pixel_img - pixel_img.min()) / (pixel_img.max() - pixel_img.min())
    plt.imshow(pixel_img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Visualize the output image
    output = outputs[0].detach().cpu().numpy().squeeze().astype(np.float32)
    output = (output - output.min()) / (output.max() - output.min())
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.show()
    break


# Serialize the predictions and save to a file
for prediction in predictions:

    # Reshape the prediction to match the size of the pixelated image without the known areas
    prediction_img = prediction.reshape(pixelated_images.shape[2:]).astype(np.float32)

    # Normalize the prediction image
    prediction_img = (prediction_img - prediction_img.min()) / (prediction_img.max() - prediction_img.min())

    # Display the prediction image
    plt.imshow(prediction_img, cmap='gray')
    plt.axis('off')
    plt.show()
serialize(predictions, "predictions.bin")
"""
with open('test_set.pkl', 'rb') as file:
    test_set = pickle.load(file)
    pixelated_images_dataset = torch.from_numpy(np.array(test_set['pixelated_images']))
    known_arrays_dataset = torch.from_numpy(np.array(test_set['known_arrays']))

    test_loader = DataLoader(dataset=TensorDataset(pixelated_images_dataset, known_arrays_dataset),
                             batch_size=32,
                             shuffle=False)


model_path = 'depixelation_model7.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCNN(n_in_channels=2, n_hidden_layers=5, n_kernels=64, kernel_size=3)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)

predicted_values = []
batch_index = 0

with torch.no_grad():
    for pixelated_images, known_arrays in test_loader:
        pixelated_images = pixelated_images.to(device)
        known_arrays = known_arrays.to(device)

        input_data = torch.cat((pixelated_images, known_arrays), dim=1)

        outputs = model(input_data)
        for index in range(len(outputs)):
            flattened_prediction = torch.masked_select(outputs[index], ~known_arrays[index])
            flattened_prediction = flattened_prediction.detach().cpu().numpy().astype(np.uint8)
            predicted_values.append(flattened_prediction)

            flattened_for_plot = flatten_prediction(outputs[index], known_arrays[index])

            # import matplotlib.pyplot as plt
            #
            # flattened_for_plot = flattened_for_plot.detach().cpu().numpy().squeeze()
            # plt.imshow(flattened_for_plot, cmap='gray')
            # plt.axis('off')
            # plt.show()
        batch_index += 1
        print(f'Batch: {batch_index}')


serialize(predicted_values, "predictions.bin")