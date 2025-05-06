"""
Script to create a 70/10/20 split of the slice data for training,
validation, and testing of Adaptive Thresholding and U-Net Methods.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

def create_data_split(input_dir='preprocessed_data', output_dir='split_data', random_state=27):
    """
    Create a 70/10/20 stratified split of the data for training, validation, and testing.

    Args:
        input_dir: Directory containing the preprocessed data
        output_dir: Directory to save the split data
        random_state: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed data
    with open(os.path.join(input_dir, 'preprocessed_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    # Get images, masks, and case IDs
    images = data['images']
    masks = data['masks']
    case_ids = data['case_ids']

    # Create stratification based on tumor size
    # Ensures distribution of tumor sizes is similar across splits
    tumor_areas = np.array([np.sum(mask == 2) for mask in masks])

    # Create bins for stratification
    tumor_area_bins = np.zeros_like(tumor_areas, dtype=int)

    # Cases with no tumor
    tumor_area_bins[tumor_areas == 0] = 0

    # Cases with tumors - divide into tertiles
    tumor_cases = tumor_areas > 0
    if np.sum(tumor_cases) > 0:
        # Get tertile thresholds
        tertiles = np.quantile(tumor_areas[tumor_cases], [0.33, 0.67])

        # Assign bin 1 for small tumors
        tumor_area_bins[(tumor_areas > 0) & (tumor_areas <= tertiles[0])] = 1

        # Assign bin 2 for medium tumors
        tumor_area_bins[(tumor_areas > tertiles[0]) & (tumor_areas <= tertiles[1])] = 2

        # Assign bin 3 for large tumors
        tumor_area_bins[tumor_areas > tertiles[1]] = 3

    # First split: 80% train+val, 20% test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(images)),
        test_size=0.2,
        random_state=random_state,
        stratify=tumor_area_bins
    )

    # Second split: 70% train, 10% validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.125,  # 0.125 * 80% = 10%
        random_state=random_state,
        stratify=tumor_area_bins[train_val_indices]
    )

    # Create the split datasets
    train_data = {
        'images': images[train_indices],
        'masks': masks[train_indices],
        'case_ids': [case_ids[i] for i in train_indices]
    }

    val_data = {
        'images': images[val_indices],
        'masks': masks[val_indices],
        'case_ids': [case_ids[i] for i in val_indices]
    }

    test_data = {
        'images': images[test_indices],
        'masks': masks[test_indices],
        'case_ids': [case_ids[i] for i in test_indices]
    }

    # Save the splits
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)

    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)

    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)

    # Save indices for reproducibility
    split_indices = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }

    with open(os.path.join(output_dir, 'split_indices.pkl'), 'wb') as f:
        pickle.dump(split_indices, f)

    # Print summary
    print(f"Data Split:")
    print(f"  Training set: {len(train_data['images'])} images")
    print(f"  Validation set: {len(val_data['images'])} images")
    print(f"  Test set: {len(test_data['images'])} images")

    # Visualize the distribution of tumor areas in each split
    plt.figure(figsize=(15, 12))

    # Training set
    plt.subplot(3, 1, 1)
    train_tumor_areas = [np.sum(mask == 2) for mask in train_data['masks']]
    plt.hist(train_tumor_areas, bins=20, alpha=0.7, color='blue')
    plt.title('Training Set Tumor Area Distribution')
    plt.xlabel('Tumor Area (pixels)')
    plt.ylabel('Frequency')

    # Validation set
    plt.subplot(3, 1, 2)
    val_tumor_areas = [np.sum(mask == 2) for mask in val_data['masks']]
    plt.hist(val_tumor_areas, bins=20, alpha=0.7, color='green')
    plt.title('Validation Set Tumor Area Distribution')
    plt.xlabel('Tumor Area (pixels)')
    plt.ylabel('Frequency')

    # Test set
    plt.subplot(3, 1, 3)
    test_tumor_areas = [np.sum(mask == 2) for mask in test_data['masks']]
    plt.hist(test_tumor_areas, bins=20, alpha=0.7, color='red')
    plt.title('Test Set Tumor Area Distribution')
    plt.xlabel('Tumor Area (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Visualize sample images from each split
    num_samples = 2
    plt.figure(figsize=(15, 10))

    # Training samples
    for i in range(num_samples):
        idx = np.random.choice(len(train_data['images']))
        plt.subplot(3, num_samples, i + 1)
        plt.imshow(train_data['images'][idx], cmap='gray')
        plt.title(f"Train Sample {i+1}")
        plt.axis('off')

    # Validation samples
    for i in range(num_samples):
        idx = np.random.choice(len(val_data['images']))
        plt.subplot(3, num_samples, num_samples + i + 1)
        plt.imshow(val_data['images'][idx], cmap='gray')
        plt.title(f"Validation Sample {i+1}")
        plt.axis('off')

    # Test samples
    for i in range(num_samples):
        idx = np.random.choice(len(test_data['images']))
        plt.subplot(3, num_samples, 2*num_samples + i + 1)
        plt.imshow(test_data['images'][idx], cmap='gray')
        plt.title(f"Test Sample {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return train_data, val_data, test_data, split_indices
