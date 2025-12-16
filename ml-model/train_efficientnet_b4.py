"""
EfficientNet-B4 Training Script
===============================
Train the best-performing CNN model for dental disease classification.

EfficientNet-B4 offers the best accuracy/efficiency tradeoff for medical imaging.

Usage:
    python train_efficientnet_b4.py --data-dir datasets/oral-diseases --epochs 50
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
CONFIG_DIR = BASE_DIR / 'config'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# EfficientNet-B4 optimal settings
IMG_SIZE = (380, 380)
BATCH_SIZE = 16  # Smaller batch for B4 due to memory


def create_efficientnet_b4_model(num_classes: int, 
                                  dropout_rate: float = 0.4,
                                  fine_tune_layers: int = 50) -> keras.Model:
    """
    Create EfficientNet-B4 model for dental disease classification.
    
    Args:
        num_classes: Number of disease classes
        dropout_rate: Dropout rate for regularization
        fine_tune_layers: Number of top layers to fine-tune
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNet-B4
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classification head
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    
    # Data augmentation layers (built into model)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 0.75)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with initial learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model, base_model


def create_data_generators(data_dir: str, validation_split: float = 0.2):
    """Create training and validation data generators with augmentation."""
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Validation - only rescale
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator


def get_callbacks(model_name: str):
    """Create training callbacks."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            str(MODELS_DIR / f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=str(RESULTS_DIR / 'logs' / f'{model_name}_{timestamp}'),
            histogram_freq=1
        ),
        
        # CSV logging
        CSVLogger(
            str(RESULTS_DIR / f'{model_name}_training_log.csv'),
            append=True
        )
    ]
    
    return callbacks


def fine_tune_model(model: keras.Model, base_model: keras.Model, 
                   fine_tune_layers: int = 50):
    """Unfreeze top layers for fine-tuning."""
    base_model.trainable = True
    
    # Freeze all layers except the last `fine_tune_layers`
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def plot_training_history(history, save_path: str = None):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # AUC
    axes[2].plot(history.history['auc'], label='Train')
    axes[2].plot(history.history['val_auc'], label='Validation')
    axes[2].set_title('Model AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training plot saved to: {save_path}")
    
    plt.show()


def save_class_labels(class_indices: dict, num_classes: int):
    """Save class labels to config."""
    # Invert the dictionary
    class_labels = {str(v): k for k, v in class_indices.items()}
    
    config = {
        'class_labels': class_labels,
        'num_classes': num_classes,
        'image_size': list(IMG_SIZE),
        'model_version': '2.0.0-efficientnet-b4',
        'trained_date': datetime.now().isoformat(),
        'architecture': 'EfficientNetB4',
        'disclaimer': 'This AI provides preliminary screening only.'
    }
    
    config_path = CONFIG_DIR / 'class_labels_b4.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Class labels saved to: {config_path}")
    return config


def train(data_dir: str, epochs: int = 50, fine_tune_epochs: int = 30):
    """
    Complete training pipeline for EfficientNet-B4.
    
    Args:
        data_dir: Path to dataset directory
        epochs: Initial training epochs
        fine_tune_epochs: Fine-tuning epochs
    """
    print("="*60)
    print("EfficientNet-B4 Dental Disease Classification Training")
    print("="*60)
    
    # Create data generators
    print("\n📁 Loading dataset...")
    train_gen, val_gen = create_data_generators(data_dir)
    
    num_classes = len(train_gen.class_indices)
    print(f"Classes: {train_gen.class_indices}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Save class labels
    save_class_labels(train_gen.class_indices, num_classes)
    
    # Create model
    print("\n🏗️ Building EfficientNet-B4 model...")
    model, base_model = create_efficientnet_b4_model(num_classes)
    model.summary()
    
    # Phase 1: Train classification head
    print("\n📈 Phase 1: Training classification head...")
    callbacks = get_callbacks('efficientnet_b4_phase1')
    
    history1 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune top layers
    print("\n📈 Phase 2: Fine-tuning top layers...")
    model = fine_tune_model(model, base_model, fine_tune_layers=50)
    callbacks = get_callbacks('efficientnet_b4_phase2')
    
    history2 = model.fit(
        train_gen,
        epochs=fine_tune_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = MODELS_DIR / 'efficientnet_b4_dental.h5'
    model.save(str(final_model_path))
    print(f"\n✅ Final model saved to: {final_model_path}")
    
    # Also save as the main model
    main_model_path = MODELS_DIR / 'oral_disease_rgb_model.h5'
    model.save(str(main_model_path))
    print(f"✅ Also saved as main model: {main_model_path}")
    
    # Plot training history
    plot_training_history(history2, str(RESULTS_DIR / 'efficientnet_b4_training.png'))
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    results = model.evaluate(val_gen, verbose=1)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    print(f"Validation AUC: {results[2]:.4f}")
    
    return model, history2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EfficientNet-B4 for dental disease classification')
    parser.add_argument('--data-dir', type=str, default='datasets/oral-diseases/train',
                       help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Initial training epochs')
    parser.add_argument('--fine-tune-epochs', type=int, default=20, help='Fine-tuning epochs')
    
    args = parser.parse_args()
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🎮 GPU available: {gpus}")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("⚠️ No GPU found, training on CPU (will be slow)")
    
    # Train
    train(args.data_dir, args.epochs, args.fine_tune_epochs)
