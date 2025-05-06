"""
Implementation of U-Net Model for kidney and tumor image segmentation on
the training and validation data from the KiTS19 dataset.
"""

# Copyright (c) 2025 Aurod Ounsinegad.
#
# This is free, open software released under the MIT License.  See
# `LICENSE` or https://choosealicense.com/licenses/mit/ for details.

# Import necessary packages
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import pickle
import torch

def implement_unet(input_dir='split_data', output_dir='unet_results', epochs=250, batch_size=8):
    """
    Implement U-Net model for kidney and tumor segmentation with PyTorch.

    Args:
        input_dir: Directory containing the split data
        output_dir: Directory to save U-Net results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(27)
    np.random.seed(27)
    random.seed(27)

    # Load training and validation datasets
    with open(os.path.join(input_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)

    with open(os.path.join(input_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare both datasets for PyTorch
    train_loader, val_loader = prep_data(train_data, val_data,  batch_size)

    # Create and train the U-Net model
    print("\nTraining U-Net model:")
    model, train_metrics, val_metrics = train_unet(train_loader, val_loader, device, output_dir, epochs)

    # Evaluate the model on the validation set to get segmentation masks
    print("\nEvaluating U-Net Model on Validation Set:")
    val_results = unet_eval(model, val_loader, val_data, device, output_dir)

    # Compare with the baseline thresholding method
    print("\nComparing U-Net against the Adaptive Thresholding Method:")
    # Load thresholding results on validation data
    with open(os.path.join('threshold_results/val', 'results.pkl'), 'rb') as f:
        threshold_results = pickle.load(f)

    compare_methods(val_results, threshold_results, output_dir)

    return model, val_results


def prep_data(train_data, val_data, batch_size=8):
    """
    Prepare PyTorch DataLoaders for training and validation sets.

    Args:
        train_data: Dictionary containing training images and masks
        val_data: Dictionary containing validation images and masks
        batch_size: Batch size for training

    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    class SegmentationDataset(torch.utils.data.Dataset):
        def __init__(self, images, masks, transform=None):
            self.images = images
            self.masks = masks
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            mask = self.masks[idx]

            # Convert to torch tensors and make sure they're contiguous
            image = torch.from_numpy(image.copy()).float().unsqueeze(0)  # Add channel dimension [1, H, W]

            # Convert mask to one-hot encoding
            mask_one_hot = np.zeros((3, *mask.shape), dtype=np.float32)
            for i in range(3):
                mask_one_hot[i][mask == i] = 1.0

            mask = torch.from_numpy(mask_one_hot.copy()).float()

            # Apply transforms if appropriate
            if self.transform:
                image, mask = self.transform(image, mask)

            return image, mask

    # Create datasets
    train_dataset = SegmentationDataset(train_data['images'], train_data['masks'],
                                      transform=train_augment())
    val_dataset = SegmentationDataset(val_data['images'], val_data['masks'])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def train_augment():
    """
    Returns a function for data augmentation during training.
    """
    def augment(image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[2]).contiguous()
            mask = torch.flip(mask, dims=[2]).contiguous()

        # Random vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[1]).contiguous()
            mask = torch.flip(mask, dims=[1]).contiguous()

        # Random rotation (90 degrees steps)
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k=k, dims=[1, 2]).contiguous()
            mask = torch.rot90(mask, k=k, dims=[1, 2]).contiguous()

        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            image = image * brightness_factor

        return image, mask

    return augment


class UNet(torch.nn.Module):
    """
    U-Net architecture for medical image segmentation.
    Adapted from the original paper with some modifications.
    """
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features

        # Encoder path
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        # Decoder path
        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Final output layer
        self.conv = torch.nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final output
        output = self.conv(dec1)

        return output

    @staticmethod
    def _block(in_channels, features, name):
        """Create a block with two convolutional layers."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(inplace=True),
        )


class DiceLoss(torch.nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten predictions and targets
        # Using reshape to handle non-contiguous tensors
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        # Calculate intersections and unions
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()

        # Calculate dice coefficient
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)

        return 1.0 - dice


class CombinedLoss(torch.nn.Module):
    """
    Combined BCE and Dice loss for segmentation tasks.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        # Apply sigmoid to predictions for Dice loss
        sigmoid_preds = torch.sigmoid(predictions)

        # Calculate Dice loss for each class
        dice_loss = 0
        for i in range(predictions.shape[1]):  # Iterate over classes
            # Make sure the tensors are contiguous before flattening
            dice_loss += self.dice_loss(
                sigmoid_preds[:, i].contiguous(),
                targets[:, i].contiguous()
            )
        dice_loss /= predictions.shape[1]  # Average over classes

        # Calculate BCE loss
        bce_loss = self.bce_loss(predictions, targets)

        # Combine losses
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return combined_loss


def train_unet(train_loader, val_loader, device, output_dir, num_epochs=250, epsilon = 1e-10):
    """
    Train the U-Net model.

    Args:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        device: Device to train on (cuda or cpu)
        output_dir: Directory to save model and results
        num_epochs: Number of training epochs
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero

    Returns:
        model: Trained U-Net model
        train_metrics: Dictionary of training metrics over epochs
        val_metrics: Dictionary of validation metrics over epochs
    """
    # Initialize model, loss function, and optimizer
    model = UNet().to(device)
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Initialize metrics dictionaries
    train_metrics = {
        'loss': [],
        'accuracy': [],
        'cross_entropy': [],
        'precision_kidney': [],
        'recall_kidney': [],
        'precision_tumor': [],
        'recall_tumor': []
    }

    val_metrics = {
        'loss': [],
        'accuracy': [],
        'cross_entropy': [],
        'precision_kidney': [],
        'recall_kidney': [],
        'precision_tumor': [],
        'recall_tumor': []
    }

    # Initialize for early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        train_cross_entropy = 0.0
        train_precision_kidney = 0.0
        train_recall_kidney = 0.0
        train_precision_tumor = 0.0
        train_recall_tumor = 0.0

        # Process mini-batches
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            images = images.to(device)  # [B, 1, H, W]
            masks = masks.to(device)    # [B, 3, H, W]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)  # [B, 3, H, W]

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)  # [B, 3, H, W]

                # Get predicted classes
                _, preds = torch.max(probs, dim=1)  # [B, H, W]

                # Convert one-hot masks to class indices
                _, mask_indices = torch.max(masks, dim=1)  # [B, H, W]

                # Calculate metrics for batch
                batch_accuracy = torch.mean((preds == mask_indices).float()).item()

                # Cross entropy (using torch's built-in function)
                batch_cross_entropy = torch.nn.functional.cross_entropy(outputs, mask_indices).item()

                # Precision and recall for kidney (class 1)
                kidney_true = mask_indices == 1
                kidney_pred = preds == 1
                kidney_tp = torch.sum(kidney_true & kidney_pred).item()
                kidney_fp = torch.sum(~kidney_true & kidney_pred).item()
                kidney_fn = torch.sum(kidney_true & ~kidney_pred).item()

                batch_precision_kidney = kidney_tp / (kidney_tp + kidney_fp + epsilon)
                batch_recall_kidney = kidney_tp / (kidney_tp + kidney_fn + epsilon)

                # Precision and recall for tumor (class 2)
                tumor_true = mask_indices == 2
                tumor_pred = preds == 2
                tumor_tp = torch.sum(tumor_true & tumor_pred).item()
                tumor_fp = torch.sum(~tumor_true & tumor_pred).item()
                tumor_fn = torch.sum(tumor_true & ~tumor_pred).item()

                batch_precision_tumor = tumor_tp / (tumor_tp + tumor_fp + epsilon)
                batch_recall_tumor = tumor_tp / (tumor_tp + tumor_fn + epsilon)

            # Update running metrics
            train_loss += loss.item()
            train_accuracy += batch_accuracy
            train_cross_entropy += batch_cross_entropy
            train_precision_kidney += batch_precision_kidney
            train_recall_kidney += batch_recall_kidney
            train_precision_tumor += batch_precision_tumor
            train_recall_tumor += batch_recall_tumor

        # Calculate average metrics for epoch
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_cross_entropy /= len(train_loader)
        train_precision_kidney /= len(train_loader)
        train_recall_kidney /= len(train_loader)
        train_precision_tumor /= len(train_loader)
        train_recall_tumor /= len(train_loader)

        # Store metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['accuracy'].append(train_accuracy)
        train_metrics['cross_entropy'].append(train_cross_entropy)
        train_metrics['precision_kidney'].append(train_precision_kidney)
        train_metrics['recall_kidney'].append(train_recall_kidney)
        train_metrics['precision_tumor'].append(train_precision_tumor)
        train_metrics['recall_tumor'].append(train_recall_tumor)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_cross_entropy = 0.0
        val_precision_kidney = 0.0
        val_recall_kidney = 0.0
        val_precision_tumor = 0.0
        val_recall_tumor = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                images = images.to(device)  # [B, 1, H, W]
                masks = masks.to(device)    # [B, 3, H, W]

                # Forward pass
                outputs = model(images)  # [B, 3, H, W]

                # Calculate loss
                loss = criterion(outputs, masks)

                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)  # [B, 3, H, W]

                # Get predicted classes
                _, preds = torch.max(probs, dim=1)  # [B, H, W]

                # Convert one-hot masks to class indices
                _, mask_indices = torch.max(masks, dim=1)  # [B, H, W]

                # Calculate metrics for batch
                batch_accuracy = torch.mean((preds == mask_indices).float()).item()

                # Cross entropy (using torch's built-in function)
                batch_cross_entropy = torch.nn.functional.cross_entropy(outputs, mask_indices).item()

                # Precision and recall for kidney (class 1)
                kidney_true = mask_indices == 1
                kidney_pred = preds == 1
                kidney_tp = torch.sum(kidney_true & kidney_pred).item()
                kidney_fp = torch.sum(~kidney_true & kidney_pred).item()
                kidney_fn = torch.sum(kidney_true & ~kidney_pred).item()

                batch_precision_kidney = kidney_tp / (kidney_tp + kidney_fp + epsilon)
                batch_recall_kidney = kidney_tp / (kidney_tp + kidney_fn + epsilon)

                # Precision and recall for tumor (class 2)
                tumor_true = mask_indices == 2
                tumor_pred = preds == 2
                tumor_tp = torch.sum(tumor_true & tumor_pred).item()
                tumor_fp = torch.sum(~tumor_true & tumor_pred).item()
                tumor_fn = torch.sum(tumor_true & ~tumor_pred).item()

                batch_precision_tumor = tumor_tp / (tumor_tp + tumor_fp + epsilon)
                batch_recall_tumor = tumor_tp / (tumor_tp + tumor_fn + epsilon)

                # Update running metrics
                val_loss += loss.item()
                val_accuracy += batch_accuracy
                val_cross_entropy += batch_cross_entropy
                val_precision_kidney += batch_precision_kidney
                val_recall_kidney += batch_recall_kidney
                val_precision_tumor += batch_precision_tumor
                val_recall_tumor += batch_recall_tumor

        # Calculate average metrics for epoch
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_cross_entropy /= len(val_loader)
        val_precision_kidney /= len(val_loader)
        val_recall_kidney /= len(val_loader)
        val_precision_tumor /= len(val_loader)
        val_recall_tumor /= len(val_loader)

        # Store metrics
        val_metrics['loss'].append(val_loss)
        val_metrics['accuracy'].append(val_accuracy)
        val_metrics['cross_entropy'].append(val_cross_entropy)
        val_metrics['precision_kidney'].append(val_precision_kidney)
        val_metrics['recall_kidney'].append(val_recall_kidney)
        val_metrics['precision_tumor'].append(val_precision_tumor)
        val_metrics['recall_tumor'].append(val_recall_tumor)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1} - New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        # Print epoch metrics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Train Prec Tumor: {train_precision_tumor:.4f}, Val Prec Tumor: {val_precision_tumor:.4f}, "
              f"Train Rec Tumor: {train_recall_tumor:.4f}, Val Rec Tumor: {val_recall_tumor:.4f}")

        # Check if early stopping criterion is met
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as validation loss hasn't improved for {patience} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(best_model_path))

    # Save training curves
    plot_train_curves(train_metrics, val_metrics, output_dir)

    return model, train_metrics, val_metrics


def plot_train_curves(train_metrics, val_metrics, output_dir):
    """
    Plot training and validation curves.

    Args:
        train_metrics: Dictionary of training metrics over epochs
        val_metrics: Dictionary of validation metrics over epochs
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('U-Net Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['accuracy'], label='Training Accuracy')
    plt.plot(val_metrics['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('U-Net Training and Validation Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'))
    plt.close()

    # Plot precision curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['precision_kidney'], label='Training Precision')
    plt.plot(val_metrics['precision_kidney'], label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('U-Net Kidney Precision')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['precision_tumor'], label='Training Precision')
    plt.plot(val_metrics['precision_tumor'], label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('U-Net Tumor Precision')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_curves.png'))
    plt.close()

    # Plot recall curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['recall_kidney'], label='Training Recall')
    plt.plot(val_metrics['recall_kidney'], label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('U-Net Kidney Recall')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['recall_tumor'], label='Training Recall')
    plt.plot(val_metrics['recall_tumor'], label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('U-Net Tumor Recall')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'recall_curves.png'))
    plt.close()


def unet_eval(model, val_loader, val_data, device, output_dir, epsilon = 1e-10):
    """
    Evaluate U-Net model on validation set.

    Args:
        model: Trained U-Net model
        val_loader: DataLoader for validation set
        val_data: Original validation data dictionary
        device: Device to evaluate on (cuda or cpu)
        output_dir: Directory to save results
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero

    Returns:
        results: Dictionary with results and metrics
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize storage for results
    predicted_masks = []
    true_masks = []

    # Metrics dictionary
    metrics = {
        'accuracy': [],
        'cross_entropy': [],
        'precision_kidney': [],
        'recall_kidney': [],
        'precision_tumor': [],
        'recall_tumor': []
    }

    # Process each batch
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating model"):
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
    true_masks = val_data['masks']

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
        else:
            precision_kidney = 1.0 if np.sum(kidney_pred) == 0 else 0.0
            recall_kidney = 1.0

        metrics['precision_kidney'].append(precision_kidney)
        metrics['recall_kidney'].append(recall_kidney)

        # Precision and recall for tumor
        tumor_true = true_mask == 2
        tumor_pred = pred_mask == 2

        if np.sum(tumor_true) > 0:
            precision_tumor = np.sum(tumor_pred & tumor_true) / (np.sum(tumor_pred) + epsilon)
            recall_tumor = np.sum(tumor_pred & tumor_true) / np.sum(tumor_true)
        else:
            precision_tumor = 1.0 if np.sum(tumor_pred) == 0 else 0.0
            recall_tumor = 1.0

        metrics['precision_tumor'].append(precision_tumor)
        metrics['recall_tumor'].append(recall_tumor)

    # Save results
    results = {
        'predicted_masks': predicted_masks,
        'metrics': metrics,
        'case_ids': val_data['case_ids']
    }

    # Save to file
    with open(os.path.join(output_dir, 'unet_results.pkl'), 'wb') as f:
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
    unet_result_viz(val_data['images'], true_masks, predicted_masks, val_data['case_ids'], output_dir)

    return results


def unet_result_viz(images, true_masks, predicted_masks, case_ids, output_dir, num_samples=5):
    """
    Visualize U-Net segmentation results.

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
        plt.title("U-Net Prediction")
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


def compare_methods(unet_results, threshold_results, output_dir, epsilon = 1e-10):
    """
    Compare U-Net and thresholding methods.

    Args:
        unet_results: Dictionary with U-Net results
        threshold_results: Dictionary with thresholding results
        output_dir: Directory to save comparison results
        epsilon: Float Value close to zero used to avoid log(0) and
        dividing by zero
    """
    # Create comparison directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    unet_metrics = unet_results['metrics']
    threshold_metrics = threshold_results['metrics']

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
    unet_values = [
        np.mean(unet_metrics['accuracy']),
        np.mean(unet_metrics['cross_entropy']),
        np.mean(unet_metrics['precision_kidney']),
        np.mean(unet_metrics['recall_kidney']),
        np.mean(unet_metrics['precision_tumor']),
        np.mean(unet_metrics['recall_tumor'])
    ]

    threshold_values = [
        np.mean(threshold_metrics['accuracy']),
        np.mean(threshold_metrics['cross_entropy']),
        np.mean(threshold_metrics['precision_kidney']),
        np.mean(threshold_metrics['recall_kidney']),
        np.mean(threshold_metrics['precision_tumor']),
        np.mean(threshold_metrics['recall_tumor'])
    ]

    # Create bar chart comparing metrics
    plt.figure(figsize=(14, 8))

    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width/2, threshold_values, width, label='Thresholding')
    plt.bar(x + width/2, unet_values, width, label='U-Net')

    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison - Adaptive Thresholding vs. U-Net')
    plt.xticks(x, metric_names, rotation=45)
    plt.legend()

    # Add values above bars
    for i, v in enumerate(threshold_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')

    for i, v in enumerate(unet_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'))
    plt.show()

    # Create a table with results
    summary = {
        'Method': ['Thresholding', 'U-Net'],
        'Accuracy': [threshold_values[0], unet_values[0]],
        'Cross Entropy': [threshold_values[1], unet_values[1]],
        'Kidney Precision': [threshold_values[2], unet_values[2]],
        'Kidney Recall': [threshold_values[3], unet_values[3]],
        'Tumor Precision': [threshold_values[4], unet_values[4]],
        'Tumor Recall': [threshold_values[5], unet_values[5]]
    }

    summary_df = pd.DataFrame(summary)
    print("\nPerformance Comparison Summary:")
    print(summary_df.to_string(index=False))

    # Calculate absolute improvement
    improvement = {
        'Metric': metric_names,
        'Absolute Improvement': [unet - threshold for unet, threshold in zip(unet_values, threshold_values)],
        'Relative Improvement (%)': [((unet - threshold) / (threshold + epsilon)) * 100 for unet, threshold in zip(unet_values, threshold_values)]
    }

    improvement_df = pd.DataFrame(improvement)
    print("\nImprovement with U-Net:")
    print(improvement_df.to_string(index=False))

    # Save summaries to CSV
    summary_df.to_csv(os.path.join(output_dir, 'performance_comparison.csv'), index=False)
    improvement_df.to_csv(os.path.join(output_dir, 'improvement_summary.csv'), index=False)

    # Visualize sample comparison
    method_compare_viz(unet_results, threshold_results, output_dir)


def method_compare_viz(unet_results, threshold_results, output_dir, num_samples=3):
    """
    Visualize comparison between U-Net and thresholding methods.

    Args:
        unet_results: Dictionary with U-Net results
        threshold_results: Dictionary with thresholding results
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'comparison_viz')
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

    # Select random samples
    if len(common_indices) > num_samples:
        selected_indices = random.sample(common_indices, num_samples)
    else:
        selected_indices = common_indices

    # Load validation data for images
    with open(os.path.join('split_data', 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    # Create comparison visualization
    plt.figure(figsize=(15, 6 * len(selected_indices)))

    for i, (unet_idx, threshold_idx, case_id) in enumerate(selected_indices):
        # Get data
        image = val_data['images'][unet_idx]
        true_mask = val_data['masks'][unet_idx]
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
        plt.title("Thresholding Prediction")
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
