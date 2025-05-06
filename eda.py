"""
Script to perform Exploratory Data Analysis on the KiTS19 dataset.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

# Import starter code
from starter_code.utils import load_case
from starter_code.visualize import visualize

def eda(data_dir='kits19/data', sample_size=5):
    """Explore the KiTS19 dataset structure and visualize sample images."""

    # Load metadata
    with open(os.path.join(data_dir, 'kits.json'), 'r') as f:
        kits_data = json.load(f)

    # Print some basic statistics
    print(f"Total number of cases: {len(kits_data)}")

    # Get list of case directories
    case_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('case_')])
    print(f"Number of case directories: {len(case_dirs)}")

    # Count malignant and benign tumors
    malignant_count = 0
    benign_count = 0

    for case in kits_data:
        if case.get('malignant', False):
            malignant_count += 1
        else:
            benign_count += 1

    print(f"Number of malignant tumors: {malignant_count}")
    print(f"Number of benign tumors: {benign_count}")

    # Explore case dimensions and distributions
    case_dims = []
    kidney_voxel_counts = []
    tumor_voxel_counts = []

    # Examine a subset of cases
    sample_cases = case_dirs[:sample_size]

    for case_id in tqdm(sample_cases, desc="Analyzing sample cases"):
        case_num = int(case_id.split('_')[1])
        volume_obj, segmentation_obj = load_case(case_num)

        # Convert to numpy arrays first
        volume = volume_obj.get_fdata()
        segmentation = segmentation_obj.get_fdata()

        # Record dimensions
        case_dims.append(volume.shape)

        # Count voxels for kidney and tumor
        kidney_voxels = np.sum(segmentation == 1)
        tumor_voxels = np.sum(segmentation == 2)

        kidney_voxel_counts.append(kidney_voxels)
        tumor_voxel_counts.append(tumor_voxels)

    print("\nSample case dimensions:")
    for i, dims in enumerate(case_dims):
        print(f"{sample_cases[i]}: {dims}")

    # Create histogram of voxel counts - fixed plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(sample_cases)), kidney_voxel_counts, color='red')
    plt.title('Sample Case Kidney Voxel Counts')
    plt.xlabel('Case Index')
    plt.ylabel('Voxel Count')
    plt.xticks(range(len(sample_cases)), range(len(sample_cases)))

    plt.subplot(1, 2, 2)
    plt.bar(range(len(sample_cases)), tumor_voxel_counts, color='blue')
    plt.title('Sample Case Tumor Voxel Counts')
    plt.xlabel('Case Index')
    plt.ylabel('Voxel Count')
    plt.xticks(range(len(sample_cases)), range(len(sample_cases)))

    plt.tight_layout()
    plt.show()

    # Visualize a sample case
    sample_idx = 0  # Change this to view other samples
    case_num = int(sample_cases[sample_idx].split('_')[1])

    volume_obj, segmentation_obj = load_case(case_num)

    # Convert to numpy arrays
    volume = volume_obj.get_fdata()
    segmentation = segmentation_obj.get_fdata()

    # Create a directory for visualization
    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)

    # Visualize using the provided function
    visualize(case_num, viz_dir)

    print(f"\nSample Case CT Scans saved in {viz_dir} directory.")

    # Also visualize a few slices directly in the notebook
    num_slices_to_show = 3
    mid_slice = volume.shape[0] // 2
    slices_to_show = [
        mid_slice - volume.shape[0] // 4,
        mid_slice,
        mid_slice + volume.shape[0] // 4
    ]

    plt.figure(figsize=(15, 5 * num_slices_to_show))

    for i, slice_idx in enumerate(slices_to_show):
        # Original slice
        plt.subplot(num_slices_to_show, 2, i*2 + 1)
        plt.imshow(volume[slice_idx], cmap='gray')
        plt.title(f"Original CT Slice ({sample_cases[sample_idx]}, Slice {slice_idx})")
        plt.colorbar(label='HU Value')
        plt.axis('off')

        # Slice with segmentation overlay
        plt.subplot(num_slices_to_show, 2, i*2 + 2)
        plt.imshow(volume[slice_idx], cmap='gray')

        # Create masks for kidney and tumor
        kidney_mask = segmentation[slice_idx] == 1
        tumor_mask = segmentation[slice_idx] == 2

        # Apply colored masks with some transparency
        plt.imshow(np.ma.masked_where(~kidney_mask, np.ones_like(volume[slice_idx])),
                  cmap='autumn', alpha=0.5)
        plt.imshow(np.ma.masked_where(~tumor_mask, np.ones_like(volume[slice_idx])),
                  cmap='winter', alpha=0.5)

        plt.title(f"Segmentation Overlay (Slice {slice_idx})")
        plt.axis('off')

        # Legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Kidney'),
            Patch(facecolor='blue', alpha=0.5, label='Tumor')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    return kits_data, case_dirs
