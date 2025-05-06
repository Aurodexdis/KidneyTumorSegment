"""
Script to extract and analyze morphological features from kidney and
tumor segmentation masks.
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
from skimage import morphology, measure

def extract_morphological_features(mask, case_id=None):
    """
    Extract morphological features from a segmentation mask.

    Args:
        mask: Segmentation mask
        case_id: Case ID for reference

    Returns:
        Dictionary of morphological features
    """
    # Create kidney and tumor masks
    kidney_mask = (mask == 1)
    tumor_mask = (mask == 2)

    # Initialize features dictionary
    features = {}

    if case_id is not None:
        features['case_id'] = case_id

    # Basic area measurements
    features['kidney_area'] = np.sum(kidney_mask)
    features['tumor_area'] = np.sum(tumor_mask)

    # If no kidney or tumor, return basic features
    if features['kidney_area'] == 0 or features['tumor_area'] == 0:
        features['tumor_kidney_ratio'] = 0
        features['tumor_circularity'] = 0
        features['tumor_compactness'] = 0
        features['kidney_circularity'] = 0
        features['kidney_compactness'] = 0
        features['tumor_kidney_interface'] = 0
        features['tumor_kidney_interface_ratio'] = 0
        features['tumor_location'] = 'N/A'
        return features

    # Area ratio
    features['tumor_kidney_ratio'] = features['tumor_area'] / features['kidney_area']

    # Label connected components
    labeled_kidney = measure.label(kidney_mask)
    labeled_tumor = measure.label(tumor_mask)

    # Region properties for kidney
    kidney_props = measure.regionprops(labeled_kidney)
    kidney_props = sorted(kidney_props, key=lambda x: x.area, reverse=True)

    # Region properties for tumor
    tumor_props = measure.regionprops(labeled_tumor)
    tumor_props = sorted(tumor_props, key=lambda x: x.area, reverse=True)

    # If no regions found, return basic features
    if not kidney_props or not tumor_props:
        features['tumor_circularity'] = 0
        features['tumor_compactness'] = 0
        features['kidney_circularity'] = 0
        features['kidney_compactness'] = 0
        features['tumor_kidney_interface'] = 0
        features['tumor_kidney_interface_ratio'] = 0
        features['tumor_location'] = 'N/A'
        return features

    # Get the largest kidney and tumor regions
    largest_kidney = kidney_props[0]
    largest_tumor = tumor_props[0]

    # Circularity (4 * pi * area / perimeter^2)
    if largest_kidney.perimeter > 0:
        features['kidney_circularity'] = (4 * np.pi * largest_kidney.area) / (largest_kidney.perimeter ** 2)
    else:
        features['kidney_circularity'] = 0

    if largest_tumor.perimeter > 0:
        features['tumor_circularity'] = (4 * np.pi * largest_tumor.area) / (largest_tumor.perimeter ** 2)
    else:
        features['tumor_circularity'] = 0

    # Compactness (ratio of area to the area of the minimum enclosing circle)
    kidney_bbox_area = (largest_kidney.bbox[2] - largest_kidney.bbox[0]) * (largest_kidney.bbox[3] - largest_kidney.bbox[1])
    if kidney_bbox_area > 0:
        features['kidney_compactness'] = largest_kidney.area / kidney_bbox_area
    else:
        features['kidney_compactness'] = 0

    tumor_bbox_area = (largest_tumor.bbox[2] - largest_tumor.bbox[0]) * (largest_tumor.bbox[3] - largest_tumor.bbox[1])
    if tumor_bbox_area > 0:
        features['tumor_compactness'] = largest_tumor.area / tumor_bbox_area
    else:
        features['tumor_compactness'] = 0

    # Tumor-kidney interface (pixels where tumor and kidney are adjacent)
    # Dilate the tumor mask and count overlapping pixels with kidney
    dilated_tumor = morphology.binary_dilation(tumor_mask)
    interface = np.logical_and(dilated_tumor, kidney_mask)
    features['tumor_kidney_interface'] = np.sum(interface)

    # Interface ratio (interface length / tumor perimeter)
    if largest_tumor.perimeter > 0:
        features['tumor_kidney_interface_ratio'] = features['tumor_kidney_interface'] / largest_tumor.perimeter
    else:
        features['tumor_kidney_interface_ratio'] = 0

    # Tumor location relative to kidney centroid
    kidney_centroid = largest_kidney.centroid
    tumor_centroid = largest_tumor.centroid

    # Determine tumor location (assuming image coordinates with origin at top-left)
    if tumor_centroid[0] < kidney_centroid[0]:
        if tumor_centroid[1] < kidney_centroid[1]:
            features['tumor_location'] = 'Upper Left'
        else:
            features['tumor_location'] = 'Lower Left'
    else:
        if tumor_centroid[1] < kidney_centroid[1]:
            features['tumor_location'] = 'Upper Right'
        else:
            features['tumor_location'] = 'Lower Right'

    return features

def analyze_morphological_features(data_dir='split_data', output_dir='feature_analysis'):
    """
    Extract and analyze morphological features from segmentation masks.

    Args:
        data_dir: Directory containing the data splits
        output_dir: Directory to save the analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    with open(os.path.join(data_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)

    with open(os.path.join(data_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    with open(os.path.join(data_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    # Combine all data for feature extraction
    all_images = np.concatenate([train_data['images'], val_data['images'], test_data['images']])
    all_masks = np.concatenate([train_data['masks'], val_data['masks'], test_data['masks']])
    all_case_ids = train_data['case_ids'] + val_data['case_ids'] + test_data['case_ids']

    # Extract features for all cases
    all_features = []

    for i, (mask, case_id) in enumerate(tqdm(zip(all_masks, all_case_ids),
                                           desc="Extracting morphological features",
                                           total=len(all_masks))):
        features = extract_morphological_features(mask, case_id)
        all_features.append(features)

    # Convert to DataFrame for easier analysis
    import pandas as pd
    features_df = pd.DataFrame(all_features)

    # Save the features
    features_df.to_csv(os.path.join(output_dir, 'morphological_features.csv'), index=False)

    # Basic statistical analysis
    print("Morphological Features Statistics:")
    print(features_df.describe())

    # Visualize feature distributions
    numerical_features = [
        'kidney_area', 'tumor_area', 'tumor_kidney_ratio',
        'kidney_circularity', 'tumor_circularity',
        'kidney_compactness', 'tumor_compactness',
        'tumor_kidney_interface', 'tumor_kidney_interface_ratio'
    ]

    # Histograms of numerical features
    n_cols = 3
    n_rows = (len(numerical_features) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, n_rows * 4))

    for i, feature in enumerate(numerical_features):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(features_df[feature].dropna(), bins=20, alpha=0.7)
        plt.title(feature)
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.show()

    # Scatter plots to explore relationships
    plt.figure(figsize=(15, 12))

    # Tumor area vs. kidney area
    plt.subplot(2, 2, 1)
    plt.scatter(features_df['kidney_area'], features_df['tumor_area'], alpha=0.7)
    plt.title('Tumor Area vs. Kidney Area')
    plt.xlabel('Kidney Area (pixels)')
    plt.ylabel('Tumor Area (pixels)')

    # Tumor circularity vs. tumor area
    plt.subplot(2, 2, 2)
    plt.scatter(features_df['tumor_area'], features_df['tumor_circularity'], alpha=0.7)
    plt.title('Tumor Circularity vs. Tumor Area')
    plt.xlabel('Tumor Area (pixels)')
    plt.ylabel('Tumor Circularity')

    # Tumor-kidney interface vs. tumor area
    plt.subplot(2, 2, 3)
    plt.scatter(features_df['tumor_area'], features_df['tumor_kidney_interface'], alpha=0.7)
    plt.title('Tumor-Kidney Interface vs. Tumor Area')
    plt.xlabel('Tumor Area (pixels)')
    plt.ylabel('Tumor-Kidney Interface (pixels)')

    # Tumor location distribution
    plt.subplot(2, 2, 4)
    location_counts = features_df['tumor_location'].value_counts()
    plt.bar(location_counts.index, location_counts.values)
    plt.title('Tumor Location Distribution')
    plt.xlabel('Location')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_relationships.png'))
    plt.show()

    # Correlation matrix for numerical features
    plt.figure(figsize=(12, 10))
    correlation = features_df[numerical_features].corr()
    plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Feature Correlation Matrix')
    plt.xticks(range(len(numerical_features)), numerical_features, rotation=90)
    plt.yticks(range(len(numerical_features)), numerical_features)

    # Add correlation values
    for i in range(len(numerical_features)):
        for j in range(len(numerical_features)):
            plt.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if abs(correlation.iloc[i, j]) > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.show()

    return features_df
