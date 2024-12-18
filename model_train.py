import torch
import torch.optim as optim
import model
from model import haversine_loss
from model import weighted_real_world_loss
from model import real_world_loss
from tqdm import tqdm
from torch.utils.data import DataLoader
from geo_dataset import GeoLocationDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# # Load targets from .npy files
# train_targets = np.load("output/train/targets.npy")  # Shape: (num_train_samples,)

# # Convert to integers if not already (ensure they're class indices)
# train_targets = train_targets.astype(int)

# # Get unique classes and their weights
# num_classes = len(np.unique(train_targets))
# class_weights = compute_class_weight(
#     class_weight="balanced",  # Balanced weights inversely proportional to frequencies
#     classes=np.arange(num_classes),
#     y=train_targets,
# )

# # Convert to PyTorch tensor
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
# class_weights_tensor = class_weights_tensor.to("cuda")

num_clusters = 64
model = model.GeoLocResNet(num_clusters)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
# criterion = haversine_loss  # Or nn.MSELoss()
# criterion = torch.nn.MSELoss()
# criterion = weighted_real_world_loss
# criterion = real_world_loss
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
device = torch.device("cuda")

train_dataset = GeoLocationDataset("output2/train", "wideres")
val_dataset = GeoLocationDataset("output2/val", "wideres")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
):
    """
    Training loop for fine-tuning the Vision Transformer model.

    Args:
        model (torch.nn.Module): The ViT model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (function): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').

    Returns:
        model (torch.nn.Module): The best model based on validation performance.
    """
    best_model_wts = None
    best_val_loss = float("inf")
    best_val_acc = 0
    results = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0  # Count of correctly classified samples
        total_train = 0  # Total number of samples
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass

            # print(f"targets: {targets}")
            # print(f"preds: {outputs}")

            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item() * inputs.size(0)  # Accumulate loss
            # Calculate training accuracy
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            correct_train += (predicted_classes == targets).sum().item()
            total_train += targets.size(0)

        train_loss /= len(train_loader.dataset)  # Average loss
        train_accuracy = correct_train / total_train  # Compute accuracy
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0  # Count of correctly classified samples
        total_val = 0  # Total number of samples
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                correct_val += (predicted_classes == targets).sum().item()
                total_val += targets.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val  # Compute accuracy
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_accuracy)

        # Print stats
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = model.state_dict()
            # Save the best model
            torch.save(model.state_dict(), "best_wideres4_geolocation.pth")

        # Step the scheduler
        scheduler.step()

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, results


torch.cuda.empty_cache()
num_epochs = 100
best_model, training_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device,
)

# Save the best model
torch.save(best_model.state_dict(), "best_wideres4_geolocation.pth")


def plot_training_results(results):
    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results["train_acc"], label="Train Accuracy")
    plt.plot(results["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_results(training_results)
