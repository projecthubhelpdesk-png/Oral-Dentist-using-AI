"""
Spectral Dental AI Model Training
=================================
Training script for the spectral dental disease detection model.

Features:
- EfficientNet-B4 backbone with spectral feature fusion
- PCA + Ensemble classifier training
- Data augmentation for dental images
- Mixed precision training
- Learning rate scheduling
- Model checkpointing

Usage:
    python train_spectral_model.py --epochs 50 --batch-size 16

Dataset: ODSI-DB or custom spectral dental images
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets" / "ODSI-DB" / "processed"
MODEL_DIR = BASE_DIR / "models" / "spectral"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Dental conditions
DENTAL_CONDITIONS = {
    0: 'healthy',
    1: 'early_caries',
    2: 'enamel_caries', 
    3: 'dentin_caries',
    4: 'demineralization',
    5: 'calculus',
    6: 'gingivitis',
    7: 'periodontal',
}

CONDITION_TO_IDX = {v: k for k, v in DENTAL_CONDITIONS.items()}


class SpectralFeatureExtractor:
    """Extract spectral features from dental images."""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_spectral_bands(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract pseudo-spectral bands from RGB image."""
        img_norm = img.astype(np.float32) / 255.0
        
        if len(img_norm.shape) == 2:
            # Grayscale
            r = g = b = img_norm
        else:
            r, g, b = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
        
        bands = {
            'blue': b,
            'green': g,
            'red': r,
            'nir_approx': (r + g) / 2,
            'fluorescence': b - 0.5 * (r + g),
            'demineralization': (b - r) / (b + r + 1e-6),
            'inflammation': (r - g) / (r + g + 1e-6),
            'calculus': np.abs(g - 0.5 * (r + b)),
        }
        
        return bands
    
    def compute_features(self, img: np.ndarray) -> np.ndarray:
        """Compute statistical features from spectral bands."""
        bands = self.extract_spectral_bands(img)
        features = []
        
        for name, band in bands.items():
            features.extend([
                np.mean(band),
                np.std(band),
                np.min(band),
                np.max(band),
                np.percentile(band, 25),
                np.percentile(band, 75),
                np.median(band),
                np.abs(np.diff(band, axis=1)).mean(),  # Gradient X
                np.abs(np.diff(band, axis=0)).mean(),  # Gradient Y
            ])
        
        return np.array(features, dtype=np.float32)


class SpectralDentalDataset(Dataset):
    """Dataset for spectral dental images."""
    
    def __init__(
        self,
        data_dir: Path,
        transform=None,
        return_spectral_features: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_spectral_features = return_spectral_features
        self.spectral_extractor = SpectralFeatureExtractor()
        
        # Load all images
        self.samples = []
        self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_samples(self):
        """Load all image paths and labels."""
        for condition_name, condition_idx in CONDITION_TO_IDX.items():
            condition_dir = self.data_dir / condition_name
            if condition_dir.exists():
                for img_path in condition_dir.glob("*.png"):
                    self.samples.append((img_path, condition_idx))
                for img_path in condition_dir.glob("*.jpg"):
                    self.samples.append((img_path, condition_idx))
                for img_path in condition_dir.glob("*.jpeg"):
                    self.samples.append((img_path, condition_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Extract spectral features before transform
        if self.return_spectral_features:
            spectral_features = self.spectral_extractor.compute_features(img_np)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        if self.return_spectral_features:
            return img, torch.tensor(spectral_features), label
        else:
            return img, label


class SpectralCNN(nn.Module):
    """CNN for spectral dental image classification."""
    
    def __init__(self, num_classes: int = 8, spectral_features_dim: int = 72):
        super().__init__()
        
        # EfficientNet-B4 backbone
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT
        )
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Spectral feature processing
        self.spectral_fc = nn.Sequential(
            nn.Linear(spectral_features_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
        # Feature dimension for ensemble
        self.feature_dim = in_features + 256
    
    def forward(self, x: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        # CNN features
        cnn_features = self.backbone(x)
        
        # Spectral features
        spectral_out = self.spectral_fc(spectral_features)
        
        # Combine
        combined = torch.cat([cnn_features, spectral_out], dim=1)
        
        return self.classifier(combined)
    
    def extract_features(self, x: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """Extract combined features for ensemble classifier."""
        cnn_features = self.backbone(x)
        spectral_out = self.spectral_fc(spectral_features)
        return torch.cat([cnn_features, spectral_out], dim=1)


def get_transforms(train: bool = True, img_size: int = 224):
    """Get image transforms for training/validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images, spectral_features, labels = batch
        images = images.to(device)
        spectral_features = spectral_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images, spectral_features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, spectral_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images, spectral_features, labels = batch
            images = images.to(device)
            spectral_features = spectral_features.to(device)
            labels = labels.to(device)
            
            outputs = model(images, spectral_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(dataloader),
        correct / total,
        np.array(all_preds),
        np.array(all_labels)
    )


def extract_features_for_ensemble(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for ensemble classifier training."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, spectral_features, labels = batch
            images = images.to(device)
            spectral_features = spectral_features.to(device)
            
            features = model.extract_features(images, spectral_features)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.vstack(all_features), np.array(all_labels)


def train_ensemble_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: int = 50
) -> Tuple[PCA, StandardScaler, RandomForestClassifier, GradientBoostingClassifier]:
    """Train ensemble classifier on extracted features."""
    logger.info("Training ensemble classifier...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA dimensionality reduction
    n_components = min(n_components, features_scaled.shape[1], features_scaled.shape[0])
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    logger.info(f"PCA: {features.shape[1]} -> {n_components} dimensions")
    logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED
    )
    rf.fit(features_pca, labels)
    
    # Train Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.1,
        random_state=SEED
    )
    gb.fit(features_pca, labels)
    
    return pca, scaler, rf, gb


def create_synthetic_dataset():
    """Create synthetic dataset if no real data exists."""
    logger.info("Creating synthetic dataset for training...")
    
    from download_odsi_db import create_sample_spectral_data, PROCESSED_DIR
    
    # Check if data exists
    train_dir = PROCESSED_DIR / "train"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        create_sample_spectral_data()
    
    return PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Train Spectral Dental AI Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--data-dir', type=str, default=None, help='Dataset directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = DATASET_DIR
        if not data_dir.exists() or not any(data_dir.iterdir()):
            data_dir = create_synthetic_dataset()
    
    logger.info(f"Dataset directory: {data_dir}")
    
    # Create datasets
    train_dataset = SpectralDentalDataset(
        data_dir / "train",
        transform=get_transforms(train=True, img_size=args.img_size)
    )
    
    val_dataset = SpectralDentalDataset(
        data_dir / "val",
        transform=get_transforms(train=False, img_size=args.img_size)
    )
    
    if len(train_dataset) == 0:
        logger.error("No training data found!")
        sys.exit(1)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    num_classes = len(DENTAL_CONDITIONS)
    model = SpectralCNN(num_classes=num_classes).to(device)
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2%}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info("=" * 60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, preds, labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history,
            }
            torch.save(checkpoint, MODEL_DIR / "spectral_cnn_best.pth")
            logger.info(f"✓ Saved best model (acc: {best_acc:.2%})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history,
            }
            torch.save(checkpoint, MODEL_DIR / f"spectral_cnn_epoch_{epoch+1}.pth")
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_acc:.2%}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(MODEL_DIR / "spectral_cnn_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Classification report
    _, _, preds, labels = validate(model, val_loader, criterion, device)
    
    logger.info("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=list(DENTAL_CONDITIONS.values()),
        digits=3
    ))
    
    # Train ensemble classifier
    logger.info("\n" + "=" * 60)
    logger.info("Training ensemble classifier...")
    
    # Extract features
    train_features, train_labels = extract_features_for_ensemble(
        model, train_loader, device
    )
    
    # Train ensemble
    pca, scaler, rf, gb = train_ensemble_classifier(
        train_features, train_labels, n_components=50
    )
    
    # Save ensemble models
    joblib.dump(pca, MODEL_DIR / "spectral_pca.joblib")
    joblib.dump(scaler, MODEL_DIR / "spectral_scaler.joblib")
    joblib.dump(rf, MODEL_DIR / "spectral_rf.joblib")
    joblib.dump(gb, MODEL_DIR / "spectral_gb.joblib")
    
    logger.info("✓ Saved ensemble models")
    
    # Evaluate ensemble on validation set
    val_features, val_labels = extract_features_for_ensemble(
        model, val_loader, device
    )
    
    val_features_scaled = scaler.transform(val_features)
    val_features_pca = pca.transform(val_features_scaled)
    
    rf_preds = rf.predict(val_features_pca)
    gb_preds = gb.predict(val_features_pca)
    
    # Ensemble prediction (voting)
    ensemble_preds = np.round((rf_preds + gb_preds) / 2).astype(int)
    
    rf_acc = (rf_preds == val_labels).mean()
    gb_acc = (gb_preds == val_labels).mean()
    ensemble_acc = (ensemble_preds == val_labels).mean()
    
    logger.info(f"\nEnsemble Results:")
    logger.info(f"Random Forest Accuracy: {rf_acc:.2%}")
    logger.info(f"Gradient Boosting Accuracy: {gb_acc:.2%}")
    logger.info(f"Ensemble Accuracy: {ensemble_acc:.2%}")
    
    # Save training history
    with open(MODEL_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model info
    model_info = {
        'model_type': 'SpectralCNN',
        'backbone': 'EfficientNet-B4',
        'num_classes': num_classes,
        'classes': DENTAL_CONDITIONS,
        'img_size': args.img_size,
        'best_val_acc': best_acc,
        'ensemble_acc': ensemble_acc,
        'trained_at': datetime.now().isoformat(),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
    }
    
    with open(MODEL_DIR / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("All models saved to: " + str(MODEL_DIR))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
