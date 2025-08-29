import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def create_mnist_dataloaders(
    target_digit: int,
    batch_size: int = 64,
    data_path: str = '../data/mnist',
    num_workers: int = 2,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Creates training and validation data loaders for a specific digit from the MNIST dataset.

    Args:
        target_digit: The MNIST digit (0-9) to filter for.
        batch_size: Batch size for data loaders.
        data_path: Path to download/store MNIST data.
        num_workers: Number of workers for data loading.
        seed: Random seed for reproducibility.

    Returns:
        train_loader, val_loader
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # MNIST images are 28x28. We'll resize to 32x32 for easier CNN downsampling.
    # The model expects values in [-1, 1], so we normalize accordingly.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizes to [-1, 1]
    ])

    # Download and load the full MNIST training dataset
    full_train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

    # Filter indices for the target digit
    idx = [i for i, target in enumerate(full_train_dataset.targets) if target == target_digit]
    print(f"Found {len(idx)} samples for digit '{target_digit}'.")

    # Split the filtered indices into training and validation sets (80/20 split)
    np.random.shuffle(idx)
    split_point = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split_point], idx[split_point:]

    print(f"Splitting into {len(train_idx)} training samples and {len(val_idx)} validation samples.")

    # Create Subset datasets
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def visualize_mnist_batch(data_loader: DataLoader, title: str = "Sample MNIST Images"):
    """
    Visualize a batch of MNIST images from the data loader.
    """
    images, _ = next(iter(data_loader))
    images = images[:8]  # Show up to 8 images

    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, img_tensor in enumerate(images):
        img = img_tensor.squeeze().cpu().numpy()
        # Denormalize from [-1, 1] to [0, 1] for visualization
        img = (img + 1) / 2
        
        ax = axes[i] if len(images) > 1 else axes
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Test the data loader
    TARGET_DIGIT = 2
    train_loader, val_loader = create_mnist_dataloaders(target_digit=TARGET_DIGIT)
    print("\nVisualizing a batch of training data...")
    visualize_mnist_batch(train_loader, title=f"Training Samples of Digit '{TARGET_DIGIT}'")
