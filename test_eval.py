"""
Script to evaluate Adaptive Thresholding and U-Net Methods for kidney
and tumor image segmentation on the test dataset.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
import torch
import threshold
import unet

def test_eval(input_dir='split_data', unet_dir='unet_results', threshold_dir='threshold_results', output_dir='test_evaluation'):
    """
    Evaluate U-Net and thresholding methods on the test set.

    Args:
        input_dir: Directory containing the split data
        unet_dir: Directory containing the trained U-Net model
        threshold_dir: Directory containing the thresholding results
        output_dir: Directory to save the evaluation results

    Returns:
        Dictionary containing all evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    with open(os.path.join(input_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    # Load trained U-Net model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet().to(device)
    model.load_state_dict(torch.load(os.path.join(unet_dir, 'best_model.pth')))
    model.eval()

    # Create test data loader for U-Net
    test_loader = prep_test(test_data)

    # Evaluate both methods
    unet_results = unet_test_eval(model, test_loader, test_data, device, output_dir)

    threshold_results = thresholding_test_eval(test_data, output_dir)

    # Compare methods
    print("\nComparing U-Net vs. Thresholding on Test Data:")
    comparison_results = test_method_compare(unet_results, threshold_results, test_data, output_dir)

    # Combine all results
    evaluation_results = {
        'unet_results': unet_results,
        'threshold_results': threshold_results,
        'comparison_results': comparison_results
    }

    # Save evaluation results
    with open(os.path.join(output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(evaluation_results, f)

    return evaluation_results


def prep_test(test_data, batch_size=8):
    """
    Prepare PyTorch DataLoader for the test set.

    Args:
        test_data: Dictionary containing test images and masks
        batch_size: Batch size for evaluation

    Returns:
        test_loader: PyTorch DataLoader for test set
    """
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, images, masks):
            self.images = images
            self.masks = masks

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            mask = self.masks[idx]

            # Convert to torch tensors
            image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension [1, H, W]

            # Convert mask to one-hot encoding
            mask_one_hot = np.zeros((3, *mask.shape), dtype=np.float32)
            for i in range(3):
                mask_one_hot[i][mask == i] = 1.0

            mask = torch.from_numpy(mask_one_hot).float()

            return image, mask

    # Create dataset
    test_dataset = TestDataset(test_data['images'], test_data['masks'])

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


def unet_test_eval(model, test_loader, test_data, device, output_dir, epsilon = 1e-10):
    """
    Evaluate U-Net model on test set.

    Args:
        model: Trained U-Net model
        test_loader: DataLoader for test set
        test_data: Original test data dictionary
        device: Device to evaluate on (cuda or cpu)
        output_dir: Directory to save results
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero

    Returns:
        results: Dictionary with results and metrics
    """
    # Create directory for U-Net results
    unet_dir = os.path.join(output_dir, 'unet')
    os.makedirs(unet_dir, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Initialize storage for results
    predicted_masks = []

    # Metrics dictionary
    metrics = {
        'accuracy': [],
        'cross_entropy': [],
        'precision_kidney': [],
        'recall_kidney': [],
        'precision_tumor': [],
        'recall_tumor': [],
        'dice_kidney': [],  # Adding Dice coefficient for kidney
        'dice_tumor': []    # Adding Dice coefficient for tumor
    }

    # Process each batch
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating U-Net model on test set"):
            # Move data to device
            images = images.to(device)  # [B, 1, H, W]

            # Forward pass
            outputs = model(images)  # [B, 3, H, W]

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)  # [B, 3, H, W]

            # Get predicted classes
            _, preds = torch.max(probs, dim=1)  # [B, H, W]

            # Move predictions back to CPU and convert to numpy
            preds_np = preds.cpu().numpy()

            # Store predicted masks
            for pred in preds_np:
                predicted_masks.append(pred)

    # Convert list to array
    predicted_masks = np.array(predicted_masks)

    # Get original true masks
    true_masks = test_data['masks']

    # Calculate metrics for each image
    for i in range(len(true_masks)):
        pred_mask = predicted_masks[i]
        true_mask = true_masks[i]

        # Pixel-wise accuracy
        accuracy = np.mean(pred_mask == true_mask)
        metrics['accuracy'].append(accuracy)

        # Cross entropy
        # One-hot encode the true mask
        true_mask_one_hot = np.zeros((3, *true_mask.shape), dtype=np.float32)
        for j in range(3):
            true_mask_one_hot[j][true_mask == j] = 1.0

        # Create probabilities for predicted mask (using a softmax-like approach)
        pred_probs = np.zeros((3, *pred_mask.shape), dtype=np.float32)
        for j in range(3):
            pred_probs[j][pred_mask == j] = 0.8
            # Add small probability to other classes
            for k in range(3):
                if k != j:
                    pred_probs[k][pred_mask == j] = 0.1

        # Calculate cross entropy
        true_flat = true_mask_one_hot.reshape(3, -1).T
        pred_flat = pred_probs.reshape(3, -1).T
        cross_entropy = -np.mean(np.sum(true_flat * np.log(pred_flat + epsilon), axis=1))
        metrics['cross_entropy'].append(cross_entropy)

        # Precision and recall for kidney
        kidney_true = true_mask == 1
        kidney_pred = pred_mask == 1

        if np.sum(kidney_true) > 0:
            precision_kidney = np.sum(kidney_pred & kidney_true) / (np.sum(kidney_pred) + epsilon)
            recall_kidney = np.sum(kidney_pred & kidney_true) / np.sum(kidney_true)
            # Dice coefficient for kidney
            dice_kidney = 2 * np.sum(kidney_pred & kidney_true) / (np.sum(kidney_pred) + np.sum(kidney_true) + epsilon)
        else:
            precision_kidney = 1.0 if np.sum(kidney_pred) == 0 else 0.0
            recall_kidney = 1.0
            dice_kidney = 1.0 if np.sum(kidney_pred) == 0 else 0.0

        metrics['precision_kidney'].append(precision_kidney)
        metrics['recall_kidney'].append(recall_kidney)
        metrics['dice_kidney'].append(dice_kidney)

        # Precision and recall for tumor
        tumor_true = true_mask == 2
        tumor_pred = pred_mask == 2

        if np.sum(tumor_true) > 0:
            precision_tumor = np.sum(tumor_pred & tumor_true) / (np.sum(tumor_pred) + epsilon)
            recall_tumor = np.sum(tumor_pred & tumor_true) / np.sum(tumor_true)
            # Dice coefficient for tumor
            dice_tumor = 2 * np.sum(tumor_pred & tumor_true) / (np.sum(tumor_pred) + np.sum(tumor_true) + epsilon)
        else:
            precision_tumor = 1.0 if np.sum(tumor_pred) == 0 else 0.0
            recall_tumor = 1.0
            dice_tumor = 1.0 if np.sum(tumor_pred) == 0 else 0.0

        metrics['precision_tumor'].append(precision_tumor)
        metrics['recall_tumor'].append(recall_tumor)
        metrics['dice_tumor'].append(dice_tumor)

    # Save results
    results = {
        'predicted_masks': predicted_masks,
        'metrics': metrics,
        'case_ids': test_data['case_ids']
    }

    # Save to file
    with open(os.path.join(unet_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Print summary statistics
    print(f"\nU-Net Results Summary:")
    print(f"Average Accuracy: {np.mean(metrics['accuracy']):.4f}")
    print(f"Average Cross Entropy: {np.mean(metrics['cross_entropy']):.4f}")
    print(f"Average Kidney Precision: {np.mean(metrics['precision_kidney']):.4f}")
    print(f"Average Kidney Recall: {np.mean(metrics['recall_kidney']):.4f}")
    print(f"Average Kidney Dice: {np.mean(metrics['dice_kidney']):.4f}")
    print(f"Average Tumor Precision: {np.mean(metrics['precision_tumor']):.4f}")
    print(f"Average Tumor Recall: {np.mean(metrics['recall_tumor']):.4f}")
    print(f"Average Tumor Dice: {np.mean(metrics['dice_tumor']):.4f}")

    # Visualize sample results
    test_result_viz(test_data['images'], true_masks, predicted_masks, test_data['case_ids'],
                           unet_dir, method_name="U-Net")

    return results


def thresholding_test_eval(test_data, output_dir, epsilon = 1e-10):
    """
    Evaluate thresholding method on test set.

    Args:
        test_data: Test data dictionary
        output_dir: Directory to save results
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero

    Returns:
        results: Dictionary with results and metrics
    """
    # Create directory for thresholding results
    threshold_dir = os.path.join(output_dir, 'threshold')
    os.makedirs(threshold_dir, exist_ok=True)

    # Extract data
    images = test_data['images']
    true_masks = test_data['masks']
    case_ids = test_data['case_ids']

    # Initialize storage for results
    predicted_masks = []

    # Metrics dictionary
    metrics = {
        'accuracy': [],
        'cross_entropy': [],
        'precision_kidney': [],
        'recall_kidney': [],
        'precision_tumor': [],
        'recall_tumor': [],
        'dice_kidney': [],  # Adding Dice coefficient for kidney
        'dice_tumor': []    # Adding Dice coefficient for tumor
    }

    # Process each image
    for i, image in enumerate(tqdm(images, desc="Evaluating Thresholding on test set")):
        # Apply the thresholding algorithm (reuse from Section 6)
        predicted_mask = threshold.segment_kidney_and_tumor(image)

        # Store the predicted mask
        predicted_masks.append(predicted_mask)

        # Calculate metrics
        true_mask = true_masks[i]

        # Pixel-wise accuracy
        accuracy = np.mean(predicted_mask == true_mask)
        metrics['accuracy'].append(accuracy)

        # Cross entropy
        true_mask_flat = true_mask.flatten()
        pred_mask_flat = predicted_mask.flatten()

        # One-hot encode
        true_one_hot = np.eye(3)[true_mask_flat]

        # For predicted mask, we need probabilities
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
            # Dice coefficient for kidney
            dice_kidney = 2 * np.sum(kidney_pred & kidney_true) / (np.sum(kidney_pred) + np.sum(kidney_true) + epsilon)
        else:
            precision_kidney = 1.0 if np.sum(kidney_pred) == 0 else 0.0
            recall_kidney = 1.0
            dice_kidney = 1.0 if np.sum(kidney_pred) == 0 else 0.0

        metrics['precision_kidney'].append(precision_kidney)
        metrics['recall_kidney'].append(recall_kidney)
        metrics['dice_kidney'].append(dice_kidney)

        # Calculate precision and recall for tumor
        tumor_true = true_mask == 2
        tumor_pred = predicted_mask == 2
        if np.sum(tumor_true) > 0:
            precision_tumor = np.sum(tumor_pred & tumor_true) / (np.sum(tumor_pred) + epsilon)
            recall_tumor = np.sum(tumor_pred & tumor_true) / np.sum(tumor_true)
            # Dice coefficient for tumor
            dice_tumor = 2 * np.sum(tumor_pred & tumor_true) / (np.sum(tumor_pred) + np.sum(tumor_true) + epsilon)
        else:
            precision_tumor = 1.0 if np.sum(tumor_pred) == 0 else 0.0
            recall_tumor = 1.0
            dice_tumor = 1.0 if np.sum(tumor_pred) == 0 else 0.0

        metrics['precision_tumor'].append(precision_tumor)
        metrics['recall_tumor'].append(recall_tumor)
        metrics['dice_tumor'].append(dice_tumor)

    # Convert list to array
    predicted_masks = np.array(predicted_masks)

    # Save results
    results = {
        'predicted_masks': predicted_masks,
        'metrics': metrics,
        'case_ids': case_ids
    }

    with open(os.path.join(threshold_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Print summary statistics
    print(f"\nThresholding Results Summary:")
    print(f"Average Accuracy: {np.mean(metrics['accuracy']):.4f}")
    print(f"Average Cross Entropy: {np.mean(metrics['cross_entropy']):.4f}")
    print(f"Average Kidney Precision: {np.mean(metrics['precision_kidney']):.4f}")
    print(f"Average Kidney Recall: {np.mean(metrics['recall_kidney']):.4f}")
    print(f"Average Kidney Dice: {np.mean(metrics['dice_kidney']):.4f}")
    print(f"Average Tumor Precision: {np.mean(metrics['precision_tumor']):.4f}")
    print(f"Average Tumor Recall: {np.mean(metrics['recall_tumor']):.4f}")
    print(f"Average Tumor Dice: {np.mean(metrics['dice_tumor']):.4f}")

    # Visualize sample results
    test_result_viz(test_data['images'], true_masks, predicted_masks, test_data['case_ids'],
                           threshold_dir, method_name="Thresholding")

    return results


def test_result_viz(images, true_masks, predicted_masks, case_ids, output_dir, method_name, num_samples=5):
    """
    Visualize segmentation results on test set.

    Args:
        images: List of input images
        true_masks: List of ground truth masks
        predicted_masks: List of predicted masks
        case_ids: List of case IDs
        output_dir: Directory to save visualizations
        method_name: Name of the method (U-Net or Thresholding)
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
        plt.title(f"{method_name} Prediction")
        plt.axis('off')

        # Add legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Kidney'),
            Patch(facecolor='blue', alpha=0.5, label='Tumor')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{method_name.lower()}_sample_results.png'))
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
        plt.title(f"Case {case_ids[idx]} - {method_name} Overlay Analysis")

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
    plt.savefig(os.path.join(viz_dir, f'{method_name.lower()}_overlay_analysis.png'))
    plt.show()


def test_method_compare(unet_results, threshold_results, test_data, output_dir, epsilon = 1e-10):
    """
    Compare U-Net and thresholding methods on test set.

    Args:
        unet_results: Dictionary with U-Net results
        threshold_results: Dictionary with thresholding results
        test_data: Test data dictionary
        output_dir: Directory to save comparison results
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero

    Returns:
        comparison_results: Dictionary with comparison metrics and
        visualizations
    """
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Extract metrics
    unet_metrics = unet_results['metrics']
    threshold_metrics = threshold_results['metrics']

    # Metric names for plotting
    metric_names = [
        'Accuracy',
        'Cross Entropy',
        'Kidney Precision',
        'Kidney Recall',
        'Kidney Dice',
        'Tumor Precision',
        'Tumor Recall',
        'Tumor Dice'
    ]

    # Extract mean values
    unet_values = [
        np.mean(unet_metrics['accuracy']),
        np.mean(unet_metrics['cross_entropy']),
        np.mean(unet_metrics['precision_kidney']),
        np.mean(unet_metrics['recall_kidney']),
        np.mean(unet_metrics['dice_kidney']),
        np.mean(unet_metrics['precision_tumor']),
        np.mean(unet_metrics['recall_tumor']),
        np.mean(unet_metrics['dice_tumor'])
    ]

    threshold_values = [
        np.mean(threshold_metrics['accuracy']),
        np.mean(threshold_metrics['cross_entropy']),
        np.mean(threshold_metrics['precision_kidney']),
        np.mean(threshold_metrics['recall_kidney']),
        np.mean(threshold_metrics['dice_kidney']),
        np.mean(threshold_metrics['precision_tumor']),
        np.mean(threshold_metrics['recall_tumor']),
        np.mean(threshold_metrics['dice_tumor'])
    ]

    # Create bar chart comparing metrics
    plt.figure(figsize=(16, 10))

    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width/2, threshold_values, width, label='Thresholding')
    plt.bar(x + width/2, unet_values, width, label='U-Net')

    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison on Test Set - Adaptive Thresholding vs. U-Net')
    plt.xticks(x, metric_names, rotation=45)
    plt.legend()

    # Add values above bars
    for i, v in enumerate(threshold_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')

    for i, v in enumerate(unet_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'performance_comparison.png'))
    plt.show()

    # Create a table with results
    summary = {
        'Method': ['Thresholding', 'U-Net'],
        'Accuracy': [threshold_values[0], unet_values[0]],
        'Cross Entropy': [threshold_values[1], unet_values[1]],
        'Kidney Precision': [threshold_values[2], unet_values[2]],
        'Kidney Recall': [threshold_values[3], unet_values[3]],
        'Kidney Dice': [threshold_values[4], unet_values[4]],
        'Tumor Precision': [threshold_values[5], unet_values[5]],
        'Tumor Recall': [threshold_values[6], unet_values[6]],
        'Tumor Dice': [threshold_values[7], unet_values[7]]
    }

    summary_df = pd.DataFrame(summary)
    print("\nPerformance Comparison Summary on Test Set:")
    print(summary_df.to_string(index=False))

    # Calculate absolute and relative improvement
    improvement = {
        'Metric': metric_names,
        'Absolute Improvement': [unet - threshold for unet, threshold in zip(unet_values, threshold_values)],
        'Relative Improvement (%)': [((unet - threshold) / (threshold + epsilon)) * 100 for unet, threshold in zip(unet_values, threshold_values)]
    }

    improvement_df = pd.DataFrame(improvement)
    print("\nImprovement with U-Net over Thresholding on Test Set:")
    print(improvement_df.to_string(index=False))

    # Save summaries to CSV
    summary_df.to_csv(os.path.join(comparison_dir, 'performance_comparison.csv'), index=False)
    improvement_df.to_csv(os.path.join(comparison_dir, 'improvement_summary.csv'), index=False)

    # # Create box plots for each metric to show distribution of performance
    # metrics_to_plot = ['accuracy', 'dice_kidney', 'dice_tumor']

    # for metric in metrics_to_plot:
    #     plt.figure(figsize=(8, 6))
    #     data = [threshold_metrics[metric], unet_metrics[metric]]
    #     plt.boxplot(data, labels=['Thresholding', 'U-Net'])
    #     plt.title(f'Distribution of {metric.replace("_", " ").title()} on Test Set')
    #     plt.ylabel(f'{metric.replace("_", " ").title()}')
    #     plt.grid(alpha=0.3)
    #     plt.savefig(os.path.join(comparison_dir, f'{metric}_distribution.png'))
    #     plt.show()

    # # Create scatter plot of tumor size vs performance
    # plt.figure(figsize=(12, 10))

    # # Calculate tumor areas
    # tumor_areas = [np.sum(mask == 2) for mask in test_data['masks']]

    # # Plot tumor dice coefficient vs tumor area
    # plt.scatter(tumor_areas, threshold_metrics['dice_tumor'],
    #             alpha=0.7, label='Thresholding', s=50, c='blue')
    # plt.scatter(tumor_areas, unet_metrics['dice_tumor'],
    #             alpha=0.7, label='U-Net', s=50, c='red')

    # # Add trend lines
    # z_threshold = np.polyfit(tumor_areas, threshold_metrics['dice_tumor'], 1)
    # p_threshold = np.poly1d(z_threshold)

    # z_unet = np.polyfit(tumor_areas, unet_metrics['dice_tumor'], 1)
    # p_unet = np.poly1d(z_unet)

    # # Define a line space for tumor areas
    # tumor_area_line = np.linspace(min(tumor_areas), max(tumor_areas), 100)

    # plt.plot(tumor_area_line, p_threshold(tumor_area_line), 'b--', alpha=0.7)
    # plt.plot(tumor_area_line, p_unet(tumor_area_line), 'r--', alpha=0.7)

    # plt.title('Tumor Dice Coefficient vs. Tumor Size')
    # plt.xlabel('Tumor Size (pixels)')
    # plt.ylabel('Dice Coefficient')
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.savefig(os.path.join(comparison_dir, 'tumor_size_vs_performance.png'))
    # plt.show()

    # Visualize comparison of both methods
    method_compare_viz(unet_results, threshold_results, test_data, comparison_dir)

    # Create confusion matrices for both methods
    create_conf_matrix(unet_results, threshold_results, test_data, comparison_dir)

    # Store comparison results
    comparison_results = {
        'summary': summary,
        'improvement': improvement,
        'unet_values': unet_values,
        'threshold_values': threshold_values,
        'metric_names': metric_names
    }

    with open(os.path.join(comparison_dir, 'comparison_results.pkl'), 'wb') as f:
        pickle.dump(comparison_results, f)

    return comparison_results


def method_compare_viz(unet_results, threshold_results, test_data, output_dir, num_samples=3):
    """
    Visualize comparison between U-Net and thresholding methods.

    Args:
        unet_results: Dictionary with U-Net results
        threshold_results: Dictionary with thresholding results
        test_data: Test data dictionary
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Get case IDs and find common ones
    unet_case_ids = unet_results['case_ids']
    threshold_case_ids = threshold_results['case_ids']

    # Find indices of common cases
    common_indices = []
    for i, case_id in enumerate(unet_case_ids):
        if case_id in threshold_case_ids:
            threshold_idx = threshold_case_ids.index(case_id)
            common_indices.append((i, threshold_idx, case_id))

    # Select samples
    if len(common_indices) > num_samples:
        # Try to select a diverse set of examples based on tumor size
        tumor_sizes = []
        for i, _, _ in common_indices:
            tumor_size = np.sum(test_data['masks'][i] == 2)
            tumor_sizes.append(tumor_size)

        # Sort indices by tumor size
        sorted_indices = [x for _, x in sorted(zip(tumor_sizes, range(len(common_indices))))]

        # Select examples with small, medium, and large tumors
        step = len(sorted_indices) // num_samples
        selected_idx = sorted_indices[::step][:num_samples]
        selected_indices = [common_indices[idx] for idx in selected_idx]
    else:
        selected_indices = common_indices

    # Create comparison visualization
    plt.figure(figsize=(15, 6 * len(selected_indices)))

    for i, (unet_idx, threshold_idx, case_id) in enumerate(selected_indices):
        # Get data
        image = test_data['images'][unet_idx]
        true_mask = test_data['masks'][unet_idx]
        unet_mask = unet_results['predicted_masks'][unet_idx]
        threshold_mask = threshold_results['predicted_masks'][threshold_idx]

        # Original image
        plt.subplot(len(selected_indices), 4, i*4 + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Original (Case {case_id})")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(len(selected_indices), 4, i*4 + 2)

        # Create a colored visualization mask
        rgb_mask = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
        rgb_mask[true_mask == 1] = [255, 0, 0]  # Red for kidney
        rgb_mask[true_mask == 2] = [0, 0, 255]  # Blue for tumor

        plt.imshow(image, cmap='gray')
        plt.imshow(rgb_mask, alpha=0.5)
        plt.title("Ground Truth")
        plt.axis('off')

        # Thresholding prediction
        plt.subplot(len(selected_indices), 4, i*4 + 3)

        # Create a colored visualization mask
        rgb_threshold = np.zeros((*threshold_mask.shape, 3), dtype=np.uint8)
        rgb_threshold[threshold_mask == 1] = [255, 0, 0]  # Red for kidney
        rgb_threshold[threshold_mask == 2] = [0, 0, 255]  # Blue for tumor

        plt.imshow(image, cmap='gray')
        plt.imshow(rgb_threshold, alpha=0.5)
        plt.title("Adaptive Thresholding Prediction")
        plt.axis('off')

        # U-Net prediction
        plt.subplot(len(selected_indices), 4, i*4 + 4)

        # Create a colored visualization mask
        rgb_unet = np.zeros((*unet_mask.shape, 3), dtype=np.uint8)
        rgb_unet[unet_mask == 1] = [255, 0, 0]  # Red for kidney
        rgb_unet[unet_mask == 2] = [0, 0, 255]  # Blue for tumor

        plt.imshow(image, cmap='gray')
        plt.imshow(rgb_unet, alpha=0.5)
        plt.title("U-Net Prediction")
        plt.axis('off')

        # Add legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label='Kidney'),
            Patch(facecolor='blue', alpha=0.5, label='Tumor')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'method_comparison.png'))
    plt.show()

    # Create side-by-side overlay visualization
    plt.figure(figsize=(15, 10 * len(selected_indices)))

    for i, (unet_idx, threshold_idx, case_id) in enumerate(selected_indices):
        # Get data
        image = test_data['images'][unet_idx]
        true_mask = test_data['masks'][unet_idx]
        unet_mask = unet_results['predicted_masks'][unet_idx]
        threshold_mask = threshold_results['predicted_masks'][threshold_idx]

        # Thresholding overlay
        plt.subplot(len(selected_indices), 2, i*2 + 1)

        img_rgb_threshold = np.zeros((*true_mask.shape, 3), dtype=np.uint8)

        # True kidney but not predicted (red)
        img_rgb_threshold[(true_mask == 1) & (threshold_mask != 1)] = [255, 0, 0]

        # Predicted kidney but not true (green)
        img_rgb_threshold[(threshold_mask == 1) & (true_mask != 1)] = [0, 255, 0]

        # Both true and predicted kidney (yellow)
        img_rgb_threshold[(true_mask == 1) & (threshold_mask == 1)] = [255, 255, 0]

        # True tumor but not predicted (blue)
        img_rgb_threshold[(true_mask == 2) & (threshold_mask != 2)] = [0, 0, 255]

        # Predicted tumor but not true (cyan)
        img_rgb_threshold[(threshold_mask == 2) & (true_mask != 2)] = [0, 255, 255]

        # Both true and predicted tumor (purple)
        img_rgb_threshold[(true_mask == 2) & (threshold_mask == 2)] = [255, 0, 255]

        plt.imshow(image, cmap='gray')
        plt.imshow(img_rgb_threshold, alpha=0.5)
        plt.title(f"Case {case_id} - Adaptive Thresholding Overlay Analysis")

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

        # U-Net overlay
        plt.subplot(len(selected_indices), 2, i*2 + 2)

        img_rgb_unet = np.zeros((*true_mask.shape, 3), dtype=np.uint8)

        # True kidney but not predicted (red)
        img_rgb_unet[(true_mask == 1) & (unet_mask != 1)] = [255, 0, 0]

        # Predicted kidney but not true (green)
        img_rgb_unet[(unet_mask == 1) & (true_mask != 1)] = [0, 255, 0]

        # Both true and predicted kidney (yellow)
        img_rgb_unet[(true_mask == 1) & (unet_mask == 1)] = [255, 255, 0]

        # True tumor but not predicted (blue)
        img_rgb_unet[(true_mask == 2) & (unet_mask != 2)] = [0, 0, 255]

        # Predicted tumor but not true (cyan)
        img_rgb_unet[(unet_mask == 2) & (true_mask != 2)] = [0, 255, 255]

        # Both true and predicted tumor (purple)
        img_rgb_unet[(true_mask == 2) & (unet_mask == 2)] = [255, 0, 255]

        plt.imshow(image, cmap='gray')
        plt.imshow(img_rgb_unet, alpha=0.5)
        plt.title(f"Case {case_id} - U-Net Overlay Analysis")

        plt.legend(handles=legend_elements, loc='lower right', fontsize='small')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'side_by_side_overlay.png'))
    plt.show()


def create_conf_matrix(unet_results, threshold_results, test_data, output_dir):
    """
    Create and visualize confusion matrices for both methods.

    Args:
        unet_results: Dictionary with U-Net results
        threshold_results: Dictionary with thresholding results
        test_data: Test data dictionary
        output_dir: Directory to save visualizations
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Flatten all masks for overall confusion matrix
    y_true = []
    y_unet = []
    y_threshold = []

    for i in range(len(test_data['masks'])):
        y_true.extend(test_data['masks'][i].flatten())
        y_unet.extend(unet_results['predicted_masks'][i].flatten())

    for i in range(len(threshold_results['predicted_masks'])):
        y_threshold.extend(threshold_results['predicted_masks'][i].flatten())

    # Compute confusion matrices
    cm_unet = confusion_matrix(y_true, y_unet, labels=[0, 1, 2])
    cm_threshold = confusion_matrix(y_true, y_threshold, labels=[0, 1, 2])

    # Normalize matrices
    cm_unet_norm = cm_unet.astype('float') / cm_unet.sum(axis=1)[:, np.newaxis]
    cm_threshold_norm = cm_threshold.astype('float') / cm_threshold.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrices
    class_names = ['Background', 'Kidney', 'Tumor']

    # U-Net confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_unet_norm, annot=True, fmt='.2f', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.title('U-Net Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'unet_confusion_matrix.png'))
    plt.show()

    # Thresholding confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_threshold_norm, annot=True, fmt='.2f', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.title('Adaptive Thresholding Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'threshold_confusion_matrix.png'))
    plt.show()

    # Print classification metrics
    print("\nClassification Report - U-Net:")
    for i, class_name in enumerate(class_names):
        precision = cm_unet[i, i] / cm_unet[:, i].sum()
        recall = cm_unet[i, i] / cm_unet[i, :].sum()
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

    print("\nClassification Report - Adaptive Thresholding:")
    for i, class_name in enumerate(class_names):
        precision = cm_threshold[i, i] / cm_threshold[:, i].sum()
        recall = cm_threshold[i, i] / cm_threshold[i, :].sum()
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
