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

model = model.GeoLocViT()
optimizer = optim.AdamW(model.parameters(), lr=1e-2,weight_decay=0.1)
# criterion = haversine_loss  # Or nn.MSELoss()
criterion = torch.nn.MSELoss()
# criterion = weighted_real_world_loss
# criterion = real_world_loss
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
device = torch.device('cpu')

train_dataset = GeoLocationDataset("output2/train")
val_dataset = GeoLocationDataset("output2/val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
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
    best_val_loss = float('inf')
    results = {'train_loss': [], 'val_loss': []}

    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}") 
        print("-" * 20)

        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            print(f"targets: {targets}")
            print(f"preds: {outputs}")
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item() * inputs.size(0)  # Accumulate loss

        train_loss /= len(train_loader.dataset)  # Average loss
        results['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        results['val_loss'].append(val_loss)

        # Print stats
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

        # Step the scheduler
        # scheduler.step()

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, results

num_epochs = 75
best_model, training_results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device
)

# Save the best model
torch.save(best_model.state_dict(), 'best_vit_geolocation.pth')

def plot_training_results(results):
    plt.figure(figsize=(10, 5))
    plt.plot(results['train_loss'], label='Train Loss')
    plt.plot(results['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_training_results(training_results)
