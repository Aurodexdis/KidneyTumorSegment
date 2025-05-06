"""
Script to preprocess the 2D slices that were extracted with
`slice_extraction.py`.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from skimage import transform

def window_image(img, window_center, window_width):
    """
    Apply windowing to the CT image.

    Args:
        img: Input CT image
        window_center: Window center in HU
        window_width: Window width in HU

    Returns:
        Windowed image with values scaled to [0, 1]
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    windowed = np.clip(img, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)

    return windowed

def normalize_image(img):
    """
    Normalize image to have zero mean and unit variance.

    Args:
        img: Input image

    Returns:
        Normalized image
    """
    if np.std(img) > 0:
        return (img - np.mean(img)) / np.std(img)
    else:
        return img - np.mean(img)

def preprocess_data(input_dir='extracted_slices', output_dir='preprocessed_data', target_size=(512, 512)):
    """
    Preprocess the extracted 2D slices.

    Args:
        input_dir: Directory containing the extracted slices
        output_dir: Directory to save the preprocessed data
        target_size: Target size for the processed images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load extracted data
    with open(os.path.join(input_dir, 'extracted_data.pkl'), 'rb') as f:
        extracted_data = pickle.load(f)

    # Initialize dictionaries for preprocessed data
    preprocessed_data = {
        'images': [],
        'masks': [],
        'case_ids': extracted_data['case_ids']
    }

    # Process each image
    for i, (image, mask) in enumerate(tqdm(zip(extracted_data['images'], extracted_data['masks']),
                                          desc="Preprocessing images",
                                          total=len(extracted_data['images']))):
        # Apply windowing for kidney/tumor contrast
        # Kidneys and tumors typically show well in the soft tissue window
        windowed_image = window_image(image, window_center=50, window_width=400)

        # Normalize the windowed image
        normalized_image = normalize_image(windowed_image)

        # Resize if needed
        if image.shape != target_size:
            normalized_image = transform.resize(normalized_image, target_size, preserve_range=True)
            mask = transform.resize(mask, target_size, order=0, preserve_range=True).astype(np.uint8)

        # Store preprocessed data
        preprocessed_data['images'].append(normalized_image)
        preprocessed_data['masks'].append(mask)

    # Convert lists to arrays for easier handling
    preprocessed_data['images'] = np.array(preprocessed_data['images'])
    preprocessed_data['masks'] = np.array(preprocessed_data['masks'])

    # Save preprocessed data
    with open(os.path.join(output_dir, 'preprocessed_data.pkl'), 'wb') as f:
        pickle.dump(preprocessed_data, f)

    print(f"Preprocessed {len(preprocessed_data['images'])} images and saved to {output_dir}")

    # Visualize preprocessing steps
    num_examples = min(3, len(extracted_data['images']))
    plt.figure(figsize=(15, 4 * num_examples))

    for i in range(num_examples):
        # Original image
        plt.subplot(num_examples, 3, i*3 + 1)
        plt.imshow(extracted_data['images'][i], cmap='gray')
        plt.title(f"Original (Case {extracted_data['case_ids'][i]})")
        plt.colorbar(label='HU Value')
        plt.axis('off')

        # Windowed image
        windowed = window_image(extracted_data['images'][i], window_center=50, window_width=400)
        plt.subplot(num_examples, 3, i*3 + 2)
        plt.imshow(windowed, cmap='gray')
        plt.title("After Windowing")
        plt.colorbar(label='Normalized Value')
        plt.axis('off')

        # Normalized image
        plt.subplot(num_examples, 3, i*3 + 3)
        plt.imshow(preprocessed_data['images'][i], cmap='gray')
        plt.title("After Normalization")
        plt.colorbar(label='Normalized Value')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Generate summary statistics
    print("Summary statistics for preprocessed images:")

    # Calculate mean and std of preprocessed images
    mean_val = np.mean(preprocessed_data['images'])
    std_val = np.std(preprocessed_data['images'])
    min_val = np.min(preprocessed_data['images'])
    max_val = np.max(preprocessed_data['images'])

    print(f"  Mean: {mean_val:.4f}")
    print(f"  Standard Deviation: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")

    return preprocessed_data
