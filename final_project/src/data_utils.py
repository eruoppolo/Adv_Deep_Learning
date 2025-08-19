"""
Data utilities for JAFFE dataset
Step 2: Data loading with train/validation split and preprocessing
"""

import os
import re
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset


# Emotion mapping based on JAFFE filename convention
EMOTION_MAP = {
    'AN': 0,  # Angry
    'DI': 1,  # Disgust
    'FE': 2,  # Fear
    'HA': 3,  # Happy
    'SA': 4,  # Sad
    'SU': 5,  # Surprise
    'NE': 6   # Neutral
}

EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class JAFFEDataset(Dataset):
    """
    JAFFE Dataset loader with enhanced preprocessing.
    """
    
    def __init__(
        self, 
        folder_path: str,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        return_filename: bool = False
    ):
        """
        Args:
            folder_path: Path to JAFFE dataset folder
            transform: Torchvision transforms to apply
            augment: Whether to apply data augmentation
            return_filename: Whether to return filename with data
        """
        self.folder = folder_path
        self.transform = transform
        self.augment = augment
        self.return_filename = return_filename
        
        # Filter out hidden files and only keep .tiff files
        self.files = sorted([
            f for f in os.listdir(folder_path) 
            if f.endswith('.tiff') and not f.startswith('._')
        ])
        
        if len(self.files) == 0:
            raise ValueError(f"No .tiff files found in {folder_path}")
        
        # Extract labels from filenames
        self.labels = []
        self.subjects = []
        
        for filename in self.files:
            # Extract emotion code (e.g., "HA" from "KA.HA1.29.tiff")
            match = re.search(r'\.([A-Z]{2})\d', filename)
            if match:
                emotion_code = match.group(1)
                self.labels.append(EMOTION_MAP.get(emotion_code, -1))
            else:
                self.labels.append(-1)
            
            # Extract subject ID (first 2 characters)
            subject_id = filename[:2] if len(filename) >= 2 else 'UN'
            self.subjects.append(subject_id)
        
        print(f"Loaded {len(self.files)} images from {folder_path}")
        print(f"Emotion distribution: {self._get_emotion_distribution()}")
        
    def _get_emotion_distribution(self) -> dict:
        """Get distribution of emotions in dataset."""
        distribution = {}
        for label in self.labels:
            if label >= 0:
                emotion_name = EMOTION_NAMES[label]
                distribution[emotion_name] = distribution.get(emotion_name, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item from dataset.
        
        Returns:
            If return_filename=False: (image, label)
            If return_filename=True: (image, label, filename)
        """
        img_name = self.files[idx]
        label = self.labels[idx]
        
        # Load image
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_filename:
            return image, label, img_name
        return image, label


def get_data_transforms(
    image_size: int = 128,
    augment: bool = False,
    normalize: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for training and validation.
    
    Args:
        image_size: Target image size
        augment: Whether to include augmentation in training transforms
        normalize: Whether to normalize images to [-1, 1]
        
    Returns:
        train_transform, val_transform
    """
    # Base transforms
    base_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    
    if normalize:
        base_transforms.append(
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        )
    
    # Training transforms with optional augmentation
    if augment:
        train_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Slight rotation to simulate head pose variation
            transforms.RandomRotation(degrees=5),
            # Slight translation
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor()
        ]
        if normalize:
            train_transforms.append(
                transforms.Normalize(mean=[0.5], std=[0.5])
            )
        train_transform = transforms.Compose(train_transforms)
    else:
        train_transform = transforms.Compose(base_transforms)
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(base_transforms)
    
    return train_transform, val_transform

def old_create_data_loaders(
    data_path: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    image_size: int = 128,
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """
    Create training and validation data loaders.
    
    Args:
        data_path: Path to JAFFE dataset
        batch_size: Batch size for data loaders
        val_split: Validation split ratio (0.2 = 20% validation)
        image_size: Target image size
        augment: Whether to use data augmentation for training
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(
        image_size=image_size,
        augment=augment
    )
    
    # Create full dataset with validation transforms (for splitting)
    full_dataset = JAFFEDataset(
        folder_path=data_path,
        transform=val_transform,
        augment=False
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Random split
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create separate datasets for train and validation with appropriate transforms
    train_dataset = JAFFEDataset(
        folder_path=data_path,
        transform=train_transform,
        augment=augment
    )
    
    val_dataset = JAFFEDataset(
        folder_path=data_path,
        transform=val_transform,
        augment=False
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Total samples: {total_size}")
    
    return train_loader, val_loader, train_subset, val_subset

def create_data_loaders(
    data_path: str,
    batch_size: int = 16,
    val_split_subject_count: int = 2,  # Use 2 subjects for validation (20%)
    image_size: int = 128,
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """
    Create training and validation data loaders with a subject-independent split.
    
    Args:
        data_path: Path to JAFFE dataset
        batch_size: Batch size for data loaders
        val_split_subject_count: Number of subjects to hold out for validation.
        image_size: Target image size
        augment: Whether to use data augmentation for training
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(
        image_size=image_size,
        augment=augment
    )
    
    # --- Start of Major Changes ---
    
    # 1. Create a single dataset instance to get file lists and subject groups
    # We apply the validation transform initially; we will change it for the training subset later.
    full_dataset = JAFFEDataset(
        folder_path=data_path,
        transform=val_transform 
    )
    
    # 2. Get subjects for a grouped split
    subjects = np.array(full_dataset.subjects)
    unique_subjects = np.unique(subjects)
    
    # Shuffle subjects to ensure a random split each time (if seed changes)
    np.random.shuffle(unique_subjects)
    
    # Split subjects into training and validation groups
    val_subjects = unique_subjects[:val_split_subject_count]
    train_subjects = unique_subjects[val_split_subject_count:]
    
    print(f"\nSplitting by subject:")
    print(f"  Training subjects: {list(train_subjects)}")
    print(f"  Validation subjects: {list(val_subjects)}")

    # 3. Create indices for train and validation sets based on subjects
    train_indices = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_indices = [i for i, s in enumerate(subjects) if s in val_subjects]

    # 4. Create Subset for validation (which will use the initial val_transform)
    val_subset = Subset(full_dataset, val_indices)
    
    # 5. Create a new dataset instance FOR TRAINING ONLY with augmentation transforms
    # This is a clean way to apply different transforms to train/val splits.
    train_dataset_augmented = JAFFEDataset(
        folder_path=data_path,
        transform=train_transform,
        augment=augment
    )
    train_subset = Subset(train_dataset_augmented, train_indices)
    
    # --- End of Major Changes ---

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    print(f"  Total samples: {len(full_dataset)}")
    
    return train_loader, val_loader, train_subset, val_subset

def visualize_batch(
    data_loader: DataLoader,
    num_samples: int = 8,
    title: str = "Sample Images",
    denormalize: bool = True
) -> None:
    """
    Visualize a batch of images from the data loader.
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to display
        title: Title for the plot
        denormalize: Whether to denormalize images from [-1, 1] to [0, 1]
    """
    # Get one batch
    images, labels = next(iter(data_loader))
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(2, 4, i + 1)
        
        # Get image
        img = images[i].squeeze().cpu().numpy()
        
        # Denormalize if needed
        if denormalize:
            img = (img + 1) / 2  # From [-1, 1] to [0, 1]
            img = np.clip(img, 0, 1)
        
        # Display
        ax.imshow(img, cmap='gray')
        
        # Set title with emotion label
        if labels[i] >= 0:
            emotion = EMOTION_NAMES[labels[i]]
            ax.set_title(f"{emotion}")
        else:
            ax.set_title("Unknown")
        
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def compute_dataset_statistics(data_loader: DataLoader) -> Tuple[float, float]:
    """
    Compute mean and std of the dataset.
    
    Args:
        data_loader: DataLoader to compute statistics from
        
    Returns:
        mean, std
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for data, _ in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.item(), std.item()

def test_data_loading():
    """Test data loading functionality."""
    print("Testing Data Loading Module...")
    print("=" * 50)
    
    # Adjust these paths according to your setup
    data_path = "../data/resized_jaffe"  # or "../../resized_jaffe"
    
    # Check if path exists
    if not os.path.exists(data_path):
        print(f"⚠️  Data path '{data_path}' not found!")
        print("Please update the path to your JAFFE dataset.")
        return
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        data_path=data_path,
        batch_size=16,
        val_split=0.2,
        augment=True
    )
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Images shape: {images.shape}")
        print(f"    Labels shape: {labels.shape}")
        print(f"    Image range: [{images.min():.2f}, {images.max():.2f}]")
        if batch_idx >= 2:  # Just test first 3 batches
            break
    
    # Compute statistics
    print("\nComputing dataset statistics...")
    mean, std = compute_dataset_statistics(val_loader)
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    
    # Visualize samples
    print("\nVisualizing samples...")
    visualize_batch(train_loader, num_samples=8, title="Training Samples")
    
    print("\n✅ Data loading test passed!")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Run test
    test_data_loading()
