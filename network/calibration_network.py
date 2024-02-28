import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from network.pointnet_model import PointNetCls
from pointcloud_utils import PointCloud
from image_utils import Image, CamModel
from config import params


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        translations_predicted, rotations_predicted = predictions[:, :3], predictions[:, 3:]
        translations_target, rotations_target = targets[:, :3], targets[:, 3:]
        translations_loss = torch.mean(torch.norm(translations_predicted - translations_target, dim=1))
        rotations_loss = torch.mean(torch.norm(rotations_predicted - rotations_target, dim=1))
        loss = translations_loss + rotations_loss/180
        return loss/2

class CustomMSELoss2(nn.Module):
    def __init__(self):
        super(CustomMSELoss2, self).__init__()

    def forward(self, predictions, targets):
        loss = 0
        for i in range(len(predictions)):
            # torch compute norm in axis 1
            loss += torch.mean(torch.norm(predictions[i] - targets[i], dim=1))
        return loss / len(predictions)


def predict(data, image, rotation, translation):
    lidar_points, camera_points = data[:, :3], data[:, 3:]
    target = torch.from_numpy(camera_points).float()
    pointcloud = PointCloud(lidar_points)
    # rotation = np.zeros(3)
    # translation = np.array([0.1, -0.05, -0.16415])
    pointcloud.define_transform_matrix(rotation, translation)
    pointcloud.get_spherical_coord(lidar2camera=1)
    image.sphere_coord = pointcloud.spherical_coord
    if params.simulated:
        image.sphere2equirect()
        image.norm_coord[1] = image.norm_coord[1] / 2
        predictions = torch.from_numpy(image.norm_coord.T).float()
        # # plot image with points. camera_points in green and predictions in red
        # plt.imshow(image.image)
        # plt.scatter(camera_points[:, 0], camera_points[:, 1], s=10, c='g')
        # plt.scatter(image.norm_coord.T[:, 0], image.norm_coord.T[:, 1], s=10, c='r')
        # plt.show()        
    else:
        image.sphere2fisheye()
        predictions = torch.from_numpy(image.spherical_proj.T).float()

    return predictions, target


def training(input_data, output_data):
    points_per_input = 8
    input_dim = 5
    output_dim = 6
    batch_size = 32

    # # Create a vector of random labels between 0 and and len(input_data) with a length of n
    # total_points = points_per_input * batch_size * 4096
    # indexes = np.random.randint(0, len(input_data), total_points)
    # train_data = input_data[indexes]
    # # Train data as input_data 256 times
    # train_data = np.tile(input_data, (128, 1))

    # # Redimension the train_data to be a tensor of size (total_points/points_per_input, points_per_input, 5)
    # train_data = np.reshape(train_data, (int(total_points/points_per_input), points_per_input, 5))
    # train_data = np.reshape(train_data, (int(len(train_data)/points_per_input), points_per_input, 5))
    train_data = np.reshape(input_data, (len(output_data), points_per_input, input_dim))
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(output_data).float())

    # Define the batch size and the number of epochs
    num_epochs = 10

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = PointNetCls(input_dim, output_dim)

    criterion = CustomMSELoss()
    # criterion = nn.MSELoss()

    learning_rate = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    iter = 0
    
    camera_model = CamModel(params.calibration_file)
    image = Image(image=np.zeros([1080, 2160]), cam_model=camera_model, spherical_image=True)

    for epoch in range(num_epochs):
        for data, label in zip(train_loader, test_loader):
            # Load data as a torch tensor with gradient accumulation abilities
            input = data[0].transpose(2, 1).requires_grad_()
            labels = label[0].requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            output_data = model(input)
            translations, rotations = output_data[0], output_data[1]
            outputs = torch.cat((translations, rotations), dim=1)

            # Calculate Loss
            data = data[0].numpy()
            predictions, targets = [], []
            for i in range(batch_size):
                # p, t = predict(data[i], image, rotations[i].detach().numpy(), translations[i].detach().numpy())
                predictions.extend([outputs[i]])
                targets.extend([labels[i]])
            # convert predictions and targets to torch tensors
            predictions = torch.stack(predictions).float().requires_grad_()
            targets = torch.stack(targets).float().requires_grad_()
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                print('Iteration: {}. Loss: {}.'.format(iter, loss.item()))
