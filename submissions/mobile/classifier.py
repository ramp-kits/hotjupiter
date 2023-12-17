import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):
    def __init__(
        self, num_classes=5, learning_rate=0.001, epochs=10, batch_size=8
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # Freeze all layers in the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, self.num_classes)

    # Function to convert grayscale imgs to RGB imgs
    # for mobilenet pretrained model
    def grayscale_to_rgb(self, tensor):
        """
        Converts a 1-channel grayscale image to a 3-channel RGB image
        by repeating the single channel.
        Assumes the input tensor is a grayscale
        image with shape (1, height, width).
        :param tensor: A 1-channel grayscale image tensor.
        :return: A 3-channel RGB image tensor.
        """
        return tensor.repeat(3, 1, 1)

    def fit(self, X, y):
        # Add channel dimension
        X = X.reshape(-1, 1, 90, 180)
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.int64)
        print(X_tensor.shape)
        print(y_tensor.shape)

        # Transform imgs (dim and normalize)
        transform = transforms.Compose(
            [
                transforms.Lambda(self.grayscale_to_rgb),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        X_transformed = torch.stack([transform(x) for x in X_tensor])

        train_dataset = TensorDataset(X_transformed, y_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    # def predict(self, X):
    #    self.model.eval()
    #    X_tensor = torch.Tensor(X).unsqueeze(1)
    #    predictions = []
    #    with torch.no_grad():
    #        for inputs in DataLoader(X_tensor, batch_size=self.batch_size):
    #            inputs = inputs.to(self.device)
    #            outputs = self.model(inputs)
    #            _, predicted = torch.max(outputs.data, 1)
    #            predictions.append(predicted.cpu().numpy())
    #    return np.array(predictions)

    def predict_proba(self, X):
        # Add channel dimension
        X = X.reshape(-1, 1, 90, 180)
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Transform imgs (dim and normalize)
        transform = transforms.Compose(
            [
                transforms.Lambda(self.grayscale_to_rgb),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        X_transformed = torch.stack([transform(x) for x in X_tensor])

        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for inputs in DataLoader(
                X_transformed, batch_size=self.batch_size
            ):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities.extend(outputs.cpu().numpy())

        return np.array(probabilities)
