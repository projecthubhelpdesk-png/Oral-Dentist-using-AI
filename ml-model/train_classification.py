"""
Train EfficientNet Classification Model for Oral Disease Detection
===================================================================

Usage:
    py -3.11 train_classification.py --epochs 30
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
DATASET_DIR = Path(__file__).parent / 'datasets' / 'oral-diseases-classification'
MODEL_DIR = Path(__file__).parent / 'models'
CONFIG_DIR = Path(__file__).parent / 'config'

# Training config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def create_model(num_classes: int) -> keras.Model:
    """Create EfficientNetB0 model for classification."""
    # Load pretrained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def train(epochs: int = 30, fine_tune_epochs: int = 10):
    """Train the classification model."""
    print("🦷 Training Oral Disease Classification Model")
    print("=" * 50)
    
    # Check dataset
    train_dir = DATASET_DIR / 'train'
    val_dir = DATASET_DIR / 'val'
    
    if not train_dir.exists():
        print("❌ Dataset not found! Run prepare_classification_dataset.py first.")
        return
    
    # Get class names
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_names)
    
    print(f"\n📊 Dataset:")
    print(f"   Classes: {class_names}")
    print(f"   Num classes: {num_classes}")
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n   Train samples: {train_generator.samples}")
    print(f"   Val samples: {val_generator.samples}")
    
    # Create model
    print("\n🔧 Creating model...")
    model, base_model = create_model(num_classes)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / 'oral_disease_rgb_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Phase 1: Train top layers
    print("\n📈 Phase 1: Training top layers...")
    history1 = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    if fine_tune_epochs > 0:
        print(f"\n📈 Phase 2: Fine-tuning ({fine_tune_epochs} epochs)...")
        
        # Unfreeze top layers of base model
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
    
    # Save final model
    model_path = MODEL_DIR / 'oral_disease_rgb_model.h5'
    model.save(str(model_path))
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save class labels
    class_labels = {str(i): name for i, name in enumerate(class_names)}
    labels_config = {
        "class_labels": class_labels,
        "num_classes": num_classes,
        "image_size": list(IMAGE_SIZE),
        "model_version": "2.0.0",
        "trained_date": datetime.now().isoformat(),
        "disclaimer": "This AI provides preliminary screening only and is not a medical diagnosis."
    }
    
    labels_path = CONFIG_DIR / 'class_labels.json'
    with open(labels_path, 'w') as f:
        json.dump(labels_config, f, indent=2)
    print(f"✅ Class labels saved to: {labels_path}")
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"   Validation Loss: {loss:.4f}")
    print(f"   Validation Accuracy: {accuracy:.2%}")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Oral Disease Classification Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--fine-tune', type=int, default=10, help='Fine-tuning epochs')
    
    args = parser.parse_args()
    
    train(epochs=args.epochs, fine_tune_epochs=args.fine_tune)
