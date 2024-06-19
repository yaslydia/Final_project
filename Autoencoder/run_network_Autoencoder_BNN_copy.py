import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import raybnn_python
from PIL import Image
import os
from torchvision import datasets, transforms,utils
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 7)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        xe = self.encoder(x)
        xd = self.decoder(xe)
        
        encoded_flat = xe.view(xe.size(0), -1)
        classification_output = self.classifier(encoded_flat)
        return xd, classification_output


# define features
def train_and_extract_features(epoch_num=200, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preprocess dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_dataset = datasets.MNIST(root="/home/cxyycl/scratch/RayBNN_Python-main/Rust_Code/data", transform=transform, train=True)
    test_dataset = datasets.MNIST(root="/home/cxyycl/scratch/RayBNN_Python-main/Rust_Code/data", transform=transform, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Define autoencoder
    model = Autoencoder().to(device)
    # Loss and optimizer function
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train model
    train_losses = []
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)
            # Forward pass
            output, class_output = model(img)
            # Calculate losses
            reconstruction_loss = reconstruction_criterion(output, img)
            classification_loss = classification_criterion(class_output, labels)
            total_loss = reconstruction_loss + classification_loss
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {epoch_loss:.4f}')
    # Evaluate the model on the test set
    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img, labels = data
            img = img.to(device)
            _, class_output = model(img)
            pred = class_output.argmax(dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(pred.cpu().numpy())
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    # Extract features
    train_features = []
    train_labels = []
    with torch.no_grad():
        for data in train_loader:
            img, label = data
            img = img.to(device)
            _, _, encoded_features = model(img)
            train_features.append(encoded_features.cpu().numpy())
            train_labels.append(label.numpy())
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    # Extract features for test set
    test_features = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            img = img.to(device)
            _, _, encoded_features = model(img)
            test_features.append(encoded_features.cpu().numpy())
            test_labels.append(label.numpy())
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return train_features, train_labels, test_features, test_labels


def train_raybnn(x_train, y_train, x_test, y_test):
    accuracy_values = []
    precision_values=[]
    recall_values = []
    f1_values=[]
    if isinstance(x_train, torch.Tensor):
        Rey_train = x_train.cpu().numpy()

    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = np.mean(x_train)

    x_train = (x_train.astype(np.float32) - mean_value) / (max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value) / (max_value - min_value)

    dir_path = "/tmp/"

    max_input_size = 784
    input_size = 784

    max_output_size = 10
    output_size = 10

    max_neuron_size = 2000

    batch_size = 1000
    traj_size = 1

    proc_num = 2
    active_size = 1000

    training_samples = 60
    crossval_samples = 60
    testing_samples = 10

    # Format MNIST dataset
    train_x = np.zeros((input_size, batch_size, traj_size, training_samples)).astype(np.float32)
    train_y = np.zeros((output_size, batch_size, traj_size, training_samples)).astype(np.float32)

    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i / batch_size)

        train_x[:, j, 0, k] = x_train[i, :]

        idx = y_train[i]
        train_y[idx, j, 0, k] = 1.0

    crossval_x = np.copy(train_x)
    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)

    # Create Neural Network
    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,

        output_size,
        max_output_size,

        active_size,
        max_neuron_size,

        batch_size,
        traj_size,

        proc_num,
        dir_path
    )

    sphere_rad = arch_search["neural_network"]["netdata"]["sphere_rad"]

    arch_search = raybnn_python.add_neuron_to_existing3(
        10,
        10000,
        sphere_rad / 1.3,
        sphere_rad / 1.3,
        sphere_rad / 1.3,

        arch_search,
    )

    arch_search = raybnn_python.select_forward_sphere(arch_search)

    raybnn_python.print_model_info(arch_search)

    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "SHUFFLE_CONNECTIONS"
    lr_strategy2 = "MAX_ALPHA"

    loss_function = "sigmoid_cross_entropy_5"

    max_epoch = 100000
    stop_epoch = 100000
    stop_train_loss = 0.005

    max_alpha = 0.01

    exit_counter_threshold = 100000
    shuffle_counter_threshold = 200

    total_epochs = 100

    for epoch in range(total_epochs):
        max_epoch += 1
        # Train Neural Network
        arch_search = raybnn_python.train_network(
            train_x,
            train_y,

            crossval_x,
            crossval_y,

            stop_strategy,
            lr_strategy,
            lr_strategy2,

            loss_function,

            max_epoch + 1,
            stop_epoch + 1,
            stop_train_loss,

            max_alpha,

            exit_counter_threshold,
            shuffle_counter_threshold,

            arch_search
        )

        test_x = np.zeros((input_size, batch_size, traj_size, testing_samples)).astype(np.float32)

        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            test_x[:, j, 0, k] = x_test[i, :]

        # Test Neural Network
        output_y = raybnn_python.test_network(
            test_x,

            arch_search
        )

        print(output_y.shape)

        pred = []
        for i in range(x_test.shape[0]):
            j = (i % batch_size)
            k = int(i / batch_size)

            sample = output_y[:, j, 0, k]
            print(sample)

            pred.append(np.argmax(sample))

        pred = [np.argmax(output_y[:, i % batch_size, 0, int(i/batch_size)]) for i in range(x_test.shape[0])]
        acc = accuracy_score(y_test, pred)

        ret = precision_recall_fscore_support(y_test, pred, average='macro')

        print(acc)
        print(ret)

        accuracy_values.append(acc)
        precision_values.append(ret[0])
        recall_values.append(ret[1])
        f1_values.append(ret[2])


    print(output_y.shape)
    return output_y.reshape(-1)




if __name__ == '__main__':

    train_features, train_labels, test_features, test_labels = train_and_extract_features()
    #print(outputs_CNN.shape)
    output_y = train_raybnn(train_features, train_labels, test_features, test_labels)