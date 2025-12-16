"""
Oral Disease Detection Model - Training Pipeline
================================================
Medical-grade AI screening model for oral disease detection.

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from PIL import Image
from tqdm import tqdm

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.001,
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_state': 42,
    'model_save_path': 'models/oral_disease_rgb_model.h5',
    'labels_save_path': 'config/class_labels.json',
}

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('config', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ============================================
# 1. DATASET EXPLORATION
# ============================================
def download_and_explore_dataset():
    """Load dataset from local data directory."""
    print("\n" + "="*60)
    print("1. DATASET EXPLORATION")
    print("="*60)
    
    # Use local data directory (prepared by prepare_data.py)
    data_root = Path(__file__).parent / "data"
    
    if not data_root.exists():
        print("\n⚠️  Local data directory not found!")
        print("Please run: py -3.11 prepare_data.py")
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    print(f"\nUsing local data directory: {data_root}")
    
    # Analyze classes
    class_info = {}
    total_images = 0
    
    for class_dir in sorted(data_root.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            images += list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.PNG'))
            class_info[class_dir.name] = len(images)
            total_images += len(images)
    
    print(f"\n{'Class Name':<30} {'Image Count':<15} {'Percentage':<10}")
    print("-" * 55)
    for class_name, count in sorted(class_info.items(), key=lambda x: -x[1]):
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"{class_name:<30} {count:<15} {percentage:.1f}%")
    print("-" * 55)
    print(f"{'TOTAL':<30} {total_images:<15}")
    
    # Check for class imbalance
    counts = list(class_info.values())
    if counts:
        imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            print("⚠️  Significant class imbalance detected. Will use class weights during training.")
    
    # Display sample images
    display_sample_images(data_root, class_info)
    
    return str(data_root), list(class_info.keys())


def display_sample_images(data_root, class_info, samples_per_class=3):
    """Display sample images from each class."""
    print("\nGenerating sample images visualization...")
    
    n_classes = len(class_info)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(12, 3*n_classes))
    
    if n_classes == 1:
        axes = [axes]
    
    for idx, class_name in enumerate(sorted(class_info.keys())):
        class_path = Path(data_root) / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        images += list(class_path.glob('*.JPG')) + list(class_path.glob('*.JPEG')) + list(class_path.glob('*.PNG'))
        
        for j in range(samples_per_class):
            ax = axes[idx][j] if n_classes > 1 else axes[j]
            if j < len(images):
                img = Image.open(images[j])
                ax.imshow(img)
                if j == 0:
                    ax.set_ylabel(class_name, fontsize=10, rotation=0, ha='right')
            ax.axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sample images saved to: results/sample_images.png")


# ============================================
# 2. DATA PREPROCESSING
# ============================================
def create_data_generators(data_dir, class_names):
    """Create data generators with augmentation."""
    print("\n" + "="*60)
    print("2. DATA PREPROCESSING")
    print("="*60)
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.3  # 30% for val+test
    )
    
    # Validation/Test data - only rescaling
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3
    )
    
    print(f"\nImage size: {CONFIG['image_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print("\nData augmentation applied:")
    print("  - Horizontal flip")
    print("  - Rotation (±20°)")
    print("  - Zoom (±20%)")
    print("  - Brightness adjustment (0.8-1.2)")
    print("  - Width/Height shift (±20%)")
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator (half of the 30%)
    val_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Number of classes: {train_generator.num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator


# ============================================
# 3. MODEL ARCHITECTURE
# ============================================
def build_model(num_classes):
    """Build transfer learning model with EfficientNetB0."""
    print("\n" + "="*60)
    print("3. MODEL ARCHITECTURE")
    print("="*60)
    
    # Load pre-trained EfficientNetB0
    print("\nLoading EfficientNetB0 (pre-trained on ImageNet)...")
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*CONFIG['image_size'], 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    print(f"Base model layers: {len(base_model.layers)} (frozen)")
    
    # Build custom top layers
    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    
    # Data augmentation layers (applied during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='OralDiseaseDetector')
    
    # Model summary
    print("\nModel Architecture:")
    print("-" * 50)
    model.summary()
    
    return model, base_model


def compile_model(model):
    """Compile model with optimizer and metrics."""
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    print("\nModel compiled with:")
    print(f"  - Optimizer: Adam (lr={CONFIG['learning_rate']})")
    print("  - Loss: categorical_crossentropy")
    print("  - Metrics: accuracy, precision, recall")
    
    return model


# ============================================
# 4. MODEL TRAINING
# ============================================
def train_model(model, train_generator, val_generator):
    """Train the model with callbacks."""
    print("\n" + "="*60)
    print("4. MODEL TRAINING")
    print("="*60)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            CONFIG['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\nTraining for up to {CONFIG['epochs']} epochs...")
    print("Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau")
    
    # Train
    history = model.fit(
        train_generator,
        epochs=CONFIG['epochs'],
        validation_data=val_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def fine_tune_model(model, base_model, train_generator, val_generator, history):
    """Fine-tune the model by unfreezing some base layers."""
    print("\n" + "-"*40)
    print("Fine-tuning phase...")
    print("-"*40)
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate'] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print(f"Unfrozen last 20 layers of base model")
    print(f"New learning rate: {CONFIG['learning_rate'] / 10}")
    
    # Continue training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(CONFIG['model_save_path'], monitor='val_accuracy', save_best_only=True)
    ]
    
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history_fine


# ============================================
# 5. MODEL EVALUATION
# ============================================
def evaluate_model(model, val_generator, class_names):
    """Evaluate model and generate reports."""
    print("\n" + "="*60)
    print("5. MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    print("\nGenerating predictions...")
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification Report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Save report
    with open('results/classification_report.txt', 'w') as f:
        f.write("Oral Disease Detection - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Oral Disease Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to: results/confusion_matrix.png")
    
    return accuracy, report


def plot_training_history(history):
    """Plot training and validation metrics."""
    print("\nGenerating training history plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History - Oral Disease Detection Model', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.close()
    print("Training history saved to: results/training_history.png")


# ============================================
# 6. MODEL EXPORT
# ============================================
def save_model_and_labels(model, class_indices):
    """Save model and class labels."""
    print("\n" + "="*60)
    print("6. MODEL EXPORT")
    print("="*60)
    
    # Save model
    model.save(CONFIG['model_save_path'])
    print(f"\n✓ Model saved to: {CONFIG['model_save_path']}")
    
    # Save class labels
    # Invert class_indices to get index -> class_name mapping
    class_labels = {str(v): k for k, v in class_indices.items()}
    
    labels_data = {
        'class_labels': class_labels,
        'num_classes': len(class_labels),
        'image_size': CONFIG['image_size'],
        'model_version': '1.0.0',
        'trained_date': datetime.now().isoformat(),
        'disclaimer': 'This AI provides preliminary screening only and is not a medical diagnosis.'
    }
    
    with open(CONFIG['labels_save_path'], 'w') as f:
        json.dump(labels_data, f, indent=2)
    print(f"✓ Class labels saved to: {CONFIG['labels_save_path']}")
    
    return class_labels


# ============================================
# MAIN TRAINING PIPELINE
# ============================================
def main():
    """Run the complete training pipeline."""
    print("\n" + "="*60)
    print("ORAL DISEASE DETECTION MODEL - TRAINING PIPELINE")
    print("="*60)
    print("\n⚠️  DISCLAIMER: This AI provides preliminary screening only")
    print("    and is not a medical diagnosis.")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(CONFIG['random_state'])
    tf.random.set_seed(CONFIG['random_state'])
    
    # 1. Dataset Exploration
    data_dir, class_names = download_and_explore_dataset()
    
    # 2. Data Preprocessing
    train_gen, val_gen = create_data_generators(data_dir, class_names)
    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())
    
    # 3. Model Architecture
    model, base_model = build_model(num_classes)
    model = compile_model(model)
    
    # 4. Model Training
    history = train_model(model, train_gen, val_gen)
    
    # Fine-tuning
    history_fine = fine_tune_model(model, base_model, train_gen, val_gen, history)
    
    # 5. Model Evaluation
    accuracy, report = evaluate_model(model, val_gen, class_names)
    plot_training_history(history)
    
    # 6. Model Export
    class_labels = save_model_and_labels(model, train_gen.class_indices)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print(f"Labels saved to: {CONFIG['labels_save_path']}")
    print("\nNext steps:")
    print("1. Run inference_api.py to start the prediction server")
    print("2. Test with: curl -X POST http://localhost:8000/predict -F 'file=@image.jpg'")
    

if __name__ == '__main__':
    main()
