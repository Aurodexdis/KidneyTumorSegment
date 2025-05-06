"""
Implementation of Adaptive Thresholding Method for kidney and tumor
image segmentation on the training and validation data from the KiTS19
dataset.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import pickle
from skimage import morphology, measure, filters

def threshold_segment(input_dir='split_data', output_dir='threshold_results'):
    """
    Thresholding method for kidney and tumor segmentation.

    Args:
        input_dir: Directory containing the split data
        output_dir: Directory to save the thresholding results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load training and validation datasets
    with open(os.path.join(input_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)

    with open(os.path.join(input_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    print("Running on Training Data:")
    train_results = process_dataset(train_data, os.path.join(output_dir, 'train'))

    print("\nRunning on Validation Data:")
    val_results = process_dataset(val_data, os.path.join(output_dir, 'val'))

    # Compare performance between train and validation sets
    compare_datasets(train_results, val_results, output_dir)

    return train_results, val_results

def process_dataset(data, output_dir, epsilon = 1e-10):
    """
    Apply thresholding to the dataset.

    Args:
        data: Dictionary containing images, masks, and case_ids
        output_dir: Directory to save results
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero

    Returns:
        Dictionary with results and metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    images = data['images']
    true_masks = data['masks']
    case_ids = data['case_ids']

    # Initialize storage for results
    predicted_masks = []
    metrics = {
        'accuracy': [],
        'cross_entropy': [],
        'precision_kidney': [],
        'recall_kidney': [],
        'precision_tumor': [],
        'recall_tumor': []
    }

    # Process each image
    for i, (image, true_mask) in enumerate(tqdm(zip(images, true_masks),
                                             desc="Processing images",
                                             total=len(images))):
        # Apply the thresholding algorithm
        predicted_mask = segment_kidney_and_tumor(image)

        # Calculate metrics
        # Pixel-wise accuracy
        accuracy = np.mean(predicted_mask == true_mask)
        metrics['accuracy'].append(accuracy)

        # Cross entropy
        true_mask_flat = true_mask.flatten()
        pred_mask_flat = predicted_mask.flatten()

        # One-hot encode
        true_one_hot = np.eye(3)[true_mask_flat]

        # For the predicted mask, we need the probabilities
        pred_probs = np.zeros((len(pred_mask_flat), 3))
        for j in range(len(pred_mask_flat)):
            pred_probs[j, pred_mask_flat[j]] = 0.8
            other_classes = [c for c in range(3) if c != pred_mask_flat[j]]
            pred_probs[j, other_classes] = 0.2 / len(other_classes)

        # Calculate cross entropy
        cross_entropy = -np.mean(np.sum(true_one_hot * np.log(pred_probs + epsilon), axis=1))
        metrics['cross_entropy'].append(cross_entropy)

        # Calculate precision and recall for kidney
        kidney_true = true_mask == 1
        kidney_pred = predicted_mask == 1
        if np.sum(kidney_true) > 0:
            precision_kidney = np.sum(kidney_pred & kidney_true) / (np.sum(kidney_pred) + epsilon)
            recall_kidney = np.sum(kidney_pred & kidney_true) / np.sum(kidney_true)
        else:
            precision_kidney = 1.0 if np.sum(kidney_pred) == 0 else 0.0
            recall_kidney = 1.0

        metrics['precision_kidney'].append(precision_kidney)
        metrics['recall_kidney'].append(recall_kidney)

        # Calculate precision and recall for tumor
        tumor_true = true_mask == 2
        tumor_pred = predicted_mask == 2
        if np.sum(tumor_true) > 0:
            precision_tumor = np.sum(tumor_pred & tumor_true) / (np.sum(tumor_pred) + epsilon)
            recall_tumor = np.sum(tumor_pred & tumor_true) / np.sum(tumor_true)
        else:
            precision_tumor = 1.0 if np.sum(tumor_pred) == 0 else 0.0
            recall_tumor = 1.0

        metrics['precision_tumor'].append(precision_tumor)
        metrics['recall_tumor'].append(recall_tumor)

        # Store the predicted mask
        predicted_masks.append(predicted_mask)

    # Convert list to array
    predicted_masks = np.array(predicted_masks)

    # Save results
    results = {
        'predicted_masks': predicted_masks,
        'metrics': metrics,
        'case_ids': case_ids
    }

    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Print summary statistics
    print(f"\nResults Summary:")
    print(f"Average Accuracy: {np.mean(metrics['accuracy']):.4f}")
    print(f"Average Cross Entropy: {np.mean(metrics['cross_entropy']):.4f}")
    print(f"Average Kidney Precision: {np.mean(metrics['precision_kidney']):.4f}")
    print(f"Average Kidney Recall: {np.mean(metrics['recall_kidney']):.4f}")
    print(f"Average Tumor Precision: {np.mean(metrics['precision_tumor']):.4f}")
    print(f"Average Tumor Recall: {np.mean(metrics['recall_tumor']):.4f}")

    # Visualize sample results
    visualize_results(images, true_masks, predicted_masks, case_ids, output_dir)

    return results

def segment_kidney_and_tumor(image):
    """
    Segmentation of kidney and tumor based on anatomical constraints.

    Args:
        image: Input CT image

    Returns:
        Mask with kidney (1) and tumor (2) segmentation
    """
    # Initialize mask
    mask = np.zeros_like(image, dtype=np.uint8)
    height, width = image.shape

    # Step 1: Apply bilateral filter to preserve edges while reducing noise
    filtered = filters.gaussian(image, sigma=1.0, preserve_range=True)

    # Step 2: Identify the abdominal region
    # Exclude as much of the background/air which has very low intensity
    body_mask = image > np.percentile(image, 5)
    body_mask = morphology.binary_erosion(body_mask, morphology.disk(5))

    # Step 3: Create kidney ROI based on anatomical knowledge
    # Kidneys are typically located in the posterior lateral area of the
    # abdominal cavity
    roi_mask = np.zeros_like(image, dtype=bool)

    # Left and right sides of the abdominal cavity (anatomical position)
    center_x = width // 2

    # Create left kidney ROI (patient's right kidney)
    x_start_left = int(center_x - width * 0.4)
    x_end_left = int(center_x - width * 0.05)
    y_start = int(height * 0.2)
    y_end = int(height * 0.8)

    # Create right kidney ROI (patient's left kidney)
    x_start_right = int(center_x + width * 0.05)
    x_end_right = int(center_x + width * 0.4)

    # Apply ROI constraints to original mask
    roi_mask[y_start:y_end, x_start_left:x_end_left] = True
    roi_mask[y_start:y_end, x_start_right:x_end_right] = True

    # Combine with body mask to avoid capturing background
    roi_mask = roi_mask & body_mask

    # Step 4: Apply adaptive thresholding for kidney segmentation
    # Kidney tissue has a characteristic intensity range in CT
    # Typically around 30-50 HU in non-contrast CT
    min_kidney_threshold = np.percentile(filtered[roi_mask], 40)
    max_kidney_threshold = np.percentile(filtered[roi_mask], 90)

    kidney_candidates = (filtered >= min_kidney_threshold) & \
                         (filtered <= max_kidney_threshold) & \
                         roi_mask

    # Step 5: Apply morphological operations to clean up the kidney mask
    kidney_candidates = morphology.binary_opening(kidney_candidates, morphology.disk(2))
    kidney_candidates = morphology.binary_closing(kidney_candidates, morphology.disk(3))

    # Step 6: Label connected components and filter by size and shape
    kidney_labels = measure.label(kidney_candidates)
    kidney_props = measure.regionprops(kidney_labels)

    # Create a kidney mask
    kidney_mask = np.zeros_like(kidney_candidates, dtype=bool)

    for prop in kidney_props:
        # Filter by area (kidneys are reasonably large structures)
        min_kidney_area = 500  # Based on image resolution

        # Filter by shape (kidneys tend to be bean-shaped/round on CT Scans)
        min_solidity = 0.5  # Solidity = area/convex_hull_area
        max_eccentricity = 0.85  # Eccentricity = 0 for circle, 1 for line

        if (prop.area >= min_kidney_area and
            prop.solidity >= min_solidity and
            prop.eccentricity <= max_eccentricity):
            kidney_mask[kidney_labels == prop.label] = True

    # Step 7: Find tumors within or adjacent to kidneys
    # Create a dilated kidney mask to find adjacent areas
    dilated_kidney = morphology.binary_dilation(kidney_mask, morphology.disk(5))

    # Tumor candidates - higher intensity than kidney tissue
    if np.sum(kidney_mask) > 0:
        upper_kidney_value = np.percentile(filtered[kidney_mask], 85)
        lower_tumor_threshold = upper_kidney_value

        # Tumors typically appear brighter than kidney tissue
        tumor_candidates = (filtered > lower_tumor_threshold) & dilated_kidney

        # Clean up tumor mask with morphological operations
        tumor_candidates = morphology.binary_opening(tumor_candidates, morphology.disk(1))

        # Find connected components for tumors
        tumor_labels = measure.label(tumor_candidates)
        tumor_props = measure.regionprops(tumor_labels)

        # Filter tumor candidates
        tumor_mask = np.zeros_like(tumor_candidates, dtype=bool)

        for prop in tumor_props:
            # Tumors should be of a reasonable size
            min_tumor_area = 50  # Adjusted based on data

            # Tumors typically have smooth boundaries
            min_tumor_solidity = 0.6

            if prop.area >= min_tumor_area and prop.solidity >= min_tumor_solidity:
                tumor_mask[tumor_labels == prop.label] = True

        # Ensure tumors are connected to or within kidney region
        tumor_mask = tumor_mask & dilated_kidney
    else:
        # No kidney detected
        tumor_mask = np.zeros_like(kidney_mask, dtype=bool)

    # Create final mask
    mask[kidney_mask] = 1
    mask[tumor_mask] = 2

    return mask

def visualize_results(images, true_masks, predicted_masks, case_ids, output_dir, num_samples=5):
    """
    Visualize segmentation results.

    Args:
        images: List of input images
        true_masks: List of ground truth masks
        predicted_masks: List of predicted masks
        case_ids: List of case IDs
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    plt.figure(figsize=(15, 5 * len(indices)))

    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(len(indices), 3, i*3 + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Original (Case {case_ids[idx]})")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(len(indices), 3, i*3 + 2)

        # Create a colored visualization mask
        rgb_mask = np.zeros((*true_masks[idx].shape, 3), dtype=np.uint8)
        rgb_mask[true_masks[idx] == 1] = [255, 0, 0]  # Red for kidney
        rgb_mask[true_masks[idx] == 2] = [0, 0, 255]  # Blue for tumor

        plt.imshow(images[idx], cmap='gray')
        plt.imshow(rgb_mask, alpha=0.5)
        plt.title("Ground Truth")
        plt.axis('off')

        # Predicted mask
        plt.subplot(len(indices), 3, i*3 + 3)

        # Create a colored visualization mask
        rgb_pred = np.zeros((*predicted_masks[idx].shape, 3), dtype=np.uint8)
        rgb_pred[predicted_masks[idx] == 1] = [255, 0, 0]  # Red for kidney
        rgb_pred[predicted_masks[idx] == 2] = [0, 0, 255]  # Blue for tumor

        plt.imshow(images[idx], cmap='gray')
        plt.imshow(rgb_pred, alpha=0.5)
        plt.title("Thresholding Prediction")
        plt.axis('off')

        # Add legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Kidney'),
            Patch(facecolor='blue', alpha=0.5, label='Tumor')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sample_results.png'))
    plt.show()

    # Create overlay visualization showing the match between prediction and ground truth
    plt.figure(figsize=(15, 5 * len(indices)))

    for i, idx in enumerate(indices):
        plt.subplot(len(indices), 1, i + 1)

        # Create an RGB image where:
        # Red = kidney only in ground truth
        # Green = kidney only in prediction
        # Yellow = kidney in both
        # Blue = tumor only in ground truth
        # Cyan = tumor only in prediction
        # Purple = tumor in both

        img_rgb = np.zeros((*true_masks[idx].shape, 3), dtype=np.uint8)

        # True kidney but not predicted (red)
        img_rgb[(true_masks[idx] == 1) & (predicted_masks[idx] != 1)] = [255, 0, 0]

        # Predicted kidney but not true (green)
        img_rgb[(predicted_masks[idx] == 1) & (true_masks[idx] != 1)] = [0, 255, 0]

        # Both true and predicted kidney (yellow)
        img_rgb[(true_masks[idx] == 1) & (predicted_masks[idx] == 1)] = [255, 255, 0]

        # True tumor but not predicted (blue)
        img_rgb[(true_masks[idx] == 2) & (predicted_masks[idx] != 2)] = [0, 0, 255]

        # Predicted tumor but not true (cyan)
        img_rgb[(predicted_masks[idx] == 2) & (true_masks[idx] != 2)] = [0, 255, 255]

        # Both true and predicted tumor (purple)
        img_rgb[(true_masks[idx] == 2) & (predicted_masks[idx] == 2)] = [255, 0, 255]

        plt.imshow(images[idx], cmap='gray')
        plt.imshow(img_rgb, alpha=0.5)
        plt.title(f"Case {case_ids[idx]} - Overlay Analysis")

        # Create custom legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Kidney missed (FN)'),
            Patch(facecolor='green', alpha=0.7, label='Kidney false (FP)'),
            Patch(facecolor='yellow', alpha=0.7, label='Kidney correct (TP)'),
            Patch(facecolor='blue', alpha=0.7, label='Tumor missed (FN)'),
            Patch(facecolor='cyan', alpha=0.7, label='Tumor false (FP)'),
            Patch(facecolor='magenta', alpha=0.7, label='Tumor correct (TP)')
        ]

        plt.legend(handles=legend_elements, loc='lower right', fontsize='small')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'overlay_analysis.png'))
    plt.show()

def compare_datasets(train_results, val_results, output_dir):
    """
    Compare performance metrics between training and validation datasets.

    Args:
        train_results: Dictionary with training results
        val_results: Dictionary with validation results
        output_dir: Directory to save comparison results
    """
    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    train_metrics = train_results['metrics']
    val_metrics = val_results['metrics']

    # Metric names for plotting
    metric_names = [
        'Accuracy',
        'Cross Entropy',
        'Kidney Precision',
        'Kidney Recall',
        'Tumor Precision',
        'Tumor Recall'
    ]

    # Extract mean values
    train_values = [
        np.mean(train_metrics['accuracy']),
        np.mean(train_metrics['cross_entropy']),
        np.mean(train_metrics['precision_kidney']),
        np.mean(train_metrics['recall_kidney']),
        np.mean(train_metrics['precision_tumor']),
        np.mean(train_metrics['recall_tumor'])
    ]

    val_values = [
        np.mean(val_metrics['accuracy']),
        np.mean(val_metrics['cross_entropy']),
        np.mean(val_metrics['precision_kidney']),
        np.mean(val_metrics['recall_kidney']),
        np.mean(val_metrics['precision_tumor']),
        np.mean(val_metrics['recall_tumor'])
    ]

    # Create bar chart comparing metrics
    plt.figure(figsize=(14, 8))

    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width/2, train_values, width, label='Training')
    plt.bar(x + width/2, val_values, width, label='Validation')

    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Thresholding Performance Comparison - Training vs. Validation')
    plt.xticks(x, metric_names, rotation=45)
    plt.legend()

    # Add values above bars
    for i, v in enumerate(train_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')

    for i, v in enumerate(val_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_val_comparison.png'))
    plt.show()

    # Create a table with results
    summary = {
        'Dataset': ['Training', 'Validation'],
        'Accuracy': [train_values[0], val_values[0]],
        'Cross Entropy': [train_values[1], val_values[1]],
        'Kidney Precision': [train_values[2], val_values[2]],
        'Kidney Recall': [train_values[3], val_values[3]],
        'Tumor Precision': [train_values[4], val_values[4]],
        'Tumor Recall': [train_values[5], val_values[5]]
    }

    summary_df = pd.DataFrame(summary)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))

    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
