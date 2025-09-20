import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from data import cqt_data, split_training_valid, labels, SAMPLE_RATE_REF

training_data, validation_data = split_training_valid(cqt_data, training_size=0.8)
training_data.to_pickle("./pickled_data/training_data.pkl")
validation_data.to_pickle("./pickled_data/validation_data.pkl")
validation_data.to_csv("./csv_files/validation_data.csv", sep="\t", index=False)

SAMPLING_RATE = SAMPLE_RATE_REF
IS_USING_FULL_CQT = False

print(
    training_data.iloc[0][
        "CQT_DATA_FULL" if IS_USING_FULL_CQT else "CQT_DATA_MEAN_TRIMMED"
    ].shape
)


class GuitarDataset(Dataset):
    def __init__(self, data, labels, is_using_full_cqt=False):
        self.data = data
        self.labels = labels
        self.is_using_full_cqt = is_using_full_cqt

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get CQT data
        cqt_data = self.data.iloc[idx][
            "CQT_DATA_FULL" if self.is_using_full_cqt else "CQT_DATA_MEAN_TRIMMED"
        ]

        # Add channel dimension for CNN (1 channel for grayscale-like CQT data)
        cqt_tensor = torch.FloatTensor(cqt_data).unsqueeze(0)

        # Get and encode label
        label_str = self.data.iloc[idx]["LABEL"]
        label_encoded = self.label_encoder.transform([label_str])
        label_tensor = torch.LongTensor(label_encoded)[0]

        return cqt_tensor, label_tensor


class GuitarCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(GuitarCNN, self).__init__()

        # Calculate input dimensions
        channels, height, width = input_shape

        # Conv2D → MaxPool → Conv2D → Flatten → Dense(softmax) architecture
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=32, kernel_size=3, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after convolutions for the linear layer
        conv_output_size = self._get_conv_output_size(input_shape)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, num_classes)

    def _get_conv_output_size(self, input_shape):
        # Helper function to calculate the output size after convolutions
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Conv2D → MaxPool
        x = self.pool1(F.relu(self.conv1(x)))

        # Conv2D → MaxPool
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten
        x = self.flatten(x)

        # Dense layer with softmax (applied via log_softmax for numerical stability)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)

    return avg_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = val_loss / len(val_loader)

    return avg_loss, accuracy


def train(name="guitar_cnn_model", epochs=25, batch_size=32, learning_rate=0.001):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = GuitarDataset(training_data, labels, IS_USING_FULL_CQT)
    val_dataset = GuitarDataset(validation_data, labels, IS_USING_FULL_CQT)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get input shape from first sample
    sample_input, _ = train_dataset[0]
    input_shape = sample_input.shape
    num_classes = len(labels)

    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Initialize model
    model = GuitarCNN(input_shape, num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.NLLLoss()  # Use NLLLoss with log_softmax
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{name}.pth")
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model


if __name__ == "__main__":
    model = train("guitar_cnn_model")
