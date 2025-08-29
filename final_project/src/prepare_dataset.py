import numpy as np
import torch
import os
from mnist_data_utils import create_mnist_dataloaders

def save_digit_data_as_numpy(target_digit: int, output_dir: str = '../data'):
    """Loads a specific MNIST digit, flattens the images, and saves them to a .npy file."""
    print(f"Preparing dataset for digit '{target_digit}'...")

    # We can use the validation loader as it doesn't shuffle, giving a consistent set.
    # Set batch_size to a large number to get all data in one go if memory allows, or iterate.
    _, val_loader = create_mnist_dataloaders(target_digit=target_digit, batch_size=1024)

    all_images = []
    for images, _ in val_loader:
        # Flatten each image from [1, 32, 32] to a vector of size 1024
        # Keep the original normalization from [-1, 1] as both models see this
        flattened_images = images.view(images.size(0), -1).numpy()
        all_images.append(flattened_images)

    # Concatenate all batches into a single numpy array
    full_dataset = np.concatenate(all_images, axis=0)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'mnist_digit_{target_digit}_flat.npy')

    # Save the array
    np.save(save_path, full_dataset)
    print(f"✅ Saved {full_dataset.shape[0]} samples to {save_path}")
    print(f"   Dataset shape: {full_dataset.shape}") # Should be (num_samples, 1024)

if __name__ == "__main__":
    TARGET_DIGIT_FOR_KPCA = 2
    save_digit_data_as_numpy(TARGET_DIGIT_FOR_KPCA)
