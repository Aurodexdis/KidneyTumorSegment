"""
Script to extract 2D slices from 3D CT volumes at the maximum tumor
cross-sectional area.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import pickle
import eda

def find_max_tumor_slice(segmentation):
    """
    Find the slice with the maximum tumor cross-sectional area.

    Args:
        segmentation: 3D numpy array where tumor voxels are labeled as 2

    Returns:
        int: Index of the slice with the maximum tumor area
    """
    # Calculate the sum of the tumor voxels in each slice
    tumor_areas = [np.sum(segmentation[i] == 2) for i in range(segmentation.shape[0])]

    # Find the slice with the maximum tumor area
    if max(tumor_areas) > 0:
        return np.argmax(tumor_areas)
    else:
        # If no tumor found, return the middle slice
        return segmentation.shape[0] // 2

def extract_slices(data_dir='kits19/data', output_dir='extracted_slices', max_case_idx=209):
    """
    Extract 2D slices from 3D volumes at the maximum tumor cross-sectional area.

    Args:
        data_dir: Directory containing the KiTS19 dataset
        output_dir: Directory to save the extracted slices
        max_case_idx: Maximum case index to process (default: 209)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of case directories, only up to max_case_idx
    case_dirs = sorted([d for d in os.listdir(data_dir)
                      if d.startswith('case_') and
                      int(d.split('_')[1]) <= max_case_idx])

    print(f"Processing {len(case_dirs)} cases (case_00000 to case_{max_case_idx:05d})")

    # Dictionaries for storing metadata
    slice_metadata = {
        'case_ids': [],
        'slice_indices': [],
        'has_tumor': [],
        'tumor_areas': [],
        'kidney_areas': []
    }

    extracted_data = {
        'images': [],
        'masks': [],
        'case_ids': []
    }

    for case_dir in tqdm(case_dirs, desc="Extracting slices"):
        case_num = int(case_dir.split('_')[1])

        try:
            # Load the case
            volume_obj, segmentation_obj = eda.load_case(case_num)

            # Convert to numpy arrays
            volume = volume_obj.get_fdata()
            segmentation = segmentation_obj.get_fdata()

            # Find the slice with the maximum tumor area
            max_tumor_slice = find_max_tumor_slice(segmentation)

            # Extract slice data
            image_slice = volume[max_tumor_slice].astype(np.float32)
            mask_slice = segmentation[max_tumor_slice].astype(np.uint8)

            # Store the extracted data
            extracted_data['images'].append(image_slice)
            extracted_data['masks'].append(mask_slice)
            extracted_data['case_ids'].append(case_num)

            # Update metadata
            slice_metadata['case_ids'].append(case_num)
            slice_metadata['slice_indices'].append(max_tumor_slice)
            slice_metadata['has_tumor'].append(np.any(mask_slice == 2))
            slice_metadata['tumor_areas'].append(np.sum(mask_slice == 2))
            slice_metadata['kidney_areas'].append(np.sum(mask_slice == 1))

        except Exception as e:
            print(f"Error processing case {case_num}: {e}")

    # Save the extracted data and metadata
    with open(os.path.join(output_dir, 'extracted_data.pkl'), 'wb') as f:
        pickle.dump(extracted_data, f)

    with open(os.path.join(output_dir, 'slice_metadata.pkl'), 'wb') as f:
        pickle.dump(slice_metadata, f)

    print(f"Extracted {len(extracted_data['images'])} slices and saved to {output_dir}")

    # Visualize a few example slices
    if len(extracted_data['images']) > 0:
        num_examples = min(5, len(extracted_data['images']))
        plt.figure(figsize=(15, 4 * num_examples))

        for i in range(num_examples):
            # Original slice
            plt.subplot(num_examples, 2, i*2 + 1)
            plt.imshow(extracted_data['images'][i], cmap='gray')
            plt.title(f"Case {extracted_data['case_ids'][i]}, Slice {slice_metadata['slice_indices'][i]}")
            plt.colorbar(label='HU Value')
            plt.axis('off')

            # Mask overlay
            plt.subplot(num_examples, 2, i*2 + 2)

            # Create a colored mask
            mask = extracted_data['masks'][i]
            rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb_mask[mask == 1] = [255, 0, 0]  # Red for kidney
            rgb_mask[mask == 2] = [0, 0, 255]  # Blue for tumor

            plt.imshow(extracted_data['images'][i], cmap='gray')
            plt.imshow(rgb_mask, alpha=0.5)
            plt.title("Segmentation Mask")
            plt.axis('off')

            # Legend
            legend_elements = [
                Patch(facecolor='red', alpha=0.5, label='Kidney'),
                Patch(facecolor='blue', alpha=0.5, label='Tumor')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

        # Create summary visualizations
        plt.figure(figsize=(15, 5))

        # Distribution of tumor areas
        plt.subplot(1, 2, 1)
        plt.hist(slice_metadata['tumor_areas'], bins=30, color='blue', alpha=0.7)
        plt.title('Distribution of Tumor Areas')
        plt.xlabel('Tumor Area (pixels)')
        plt.ylabel('Frequency')

        # Scatter plot of kidney vs tumor areas
        plt.subplot(1, 2, 2)
        plt.scatter(slice_metadata['kidney_areas'], slice_metadata['tumor_areas'],
                    alpha=0.5, color='purple')
        plt.title('Kidney Area VS. Tumor Area')
        plt.xlabel('Kidney Area (pixels)')
        plt.ylabel('Tumor Area (pixels)')

        plt.tight_layout()
        plt.show()

    return extracted_data, slice_metadata
