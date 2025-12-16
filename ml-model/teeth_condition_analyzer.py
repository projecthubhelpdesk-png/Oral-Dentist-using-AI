#!/usr/bin/env python3
"""
EfficientNet Teeth Condition Analyzer (RGB Images)
==================================================
Comprehensive teeth condition analysis using EfficientNet architecture.

Analyzes:
- Overall teeth health condition
- Teeth whiteness/discoloration
- Visible decay/cavities
- Gum health indicators
- Plaque/tartar buildup
- Teeth alignment assessment

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
import io
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2
from tensorflow.keras.preprocessing.image import img_to_array

# ============================================
# CONFIGURATION
# ============================================
MODEL_DIR = Path(__file__).parent / 'models'
CONFIG_DIR = Path(__file__).parent / 'config'
IMAGE_SIZE = (224, 224)

# Condition categories and their descriptions
CONDITION_CATEGORIES = {
    'overall_health': {
        'name': 'Overall Dental Health',
        'levels': ['Poor', 'Fair', 'Good', 'Excellent'],
        'description': 'General assessment of teeth and oral health'
    },
    'whiteness': {
        'name': 'Teeth Whiteness',
        'levels': ['Severely Discolored', 'Moderately Discolored', 'Slightly Discolored', 'White/Bright'],
        'description': 'Assessment of teeth color and staining'
    },
    'decay_risk': {
        'name': 'Decay/Cavity Risk',
        'levels': ['High Risk', 'Moderate Risk', 'Low Risk', 'Minimal Risk'],
        'description': 'Visible signs of decay or cavity formation'
    },
    'gum_health': {
        'name': 'Gum Health',
        'levels': ['Severe Issues', 'Moderate Issues', 'Minor Issues', 'Healthy'],
        'description': 'Assessment of gum condition and inflammation'
    },
    'plaque_level': {
        'name': 'Plaque/Tartar Level',
        'levels': ['Heavy Buildup', 'Moderate Buildup', 'Light Buildup', 'Clean'],
        'description': 'Visible plaque or tartar accumulation'
    },
    'alignment': {
        'name': 'Teeth Alignment',
        'levels': ['Severely Misaligned', 'Moderately Misaligned', 'Slightly Misaligned', 'Well Aligned'],
        'description': 'Assessment of teeth positioning and alignment'
    }
}

# Medical disclaimer
DISCLAIMER = "This AI provides preliminary screening only and is not a medical diagnosis. Always consult a qualified dental professional."


class TeethConditionModel:
    """
    Multi-output EfficientNet model for teeth condition analysis.
    Predicts multiple condition categories simultaneously.
    """
    
    def __init__(self, num_categories: int = 6, num_levels: int = 4):
        self.num_categories = num_categories
        self.num_levels = num_levels
        self.model = None
        self.image_size = IMAGE_SIZE
        
    def build_model(self) -> Model:
        """Build multi-output EfficientNet model."""
        # Base model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Input
        inputs = keras.Input(shape=(*self.image_size, 3))
        
        # Base model features
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Shared dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Multiple output heads for each condition category
        outputs = []
        output_names = list(CONDITION_CATEGORIES.keys())
        
        for category in output_names:
            branch = layers.Dense(64, activation='relu', name=f'{category}_dense')(x)
            output = layers.Dense(self.num_levels, activation='softmax', name=category)(branch)
            outputs.append(output)
        
        # Build model
        self.model = Model(inputs=inputs, outputs=outputs, name='teeth_condition_analyzer')
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with appropriate loss and metrics."""
        if self.model is None:
            self.build_model()
        
        # Loss for each output
        losses = {cat: 'categorical_crossentropy' for cat in CONDITION_CATEGORIES.keys()}
        
        # Metrics for each output
        metrics = {cat: ['accuracy'] for cat in CONDITION_CATEGORIES.keys()}
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses,
            metrics=metrics
        )
        
        return self.model
    
    def save_model(self, path: str):
        """Save model weights."""
        if self.model:
            self.model.save_weights(path)
            print(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        if self.model is None:
            self.build_model()
        self.model.load_weights(path)
        print(f"Model loaded from: {path}")


class TeethConditionAnalyzer:
    """
    Complete teeth condition analyzer using EfficientNet.
    Provides comprehensive analysis of dental health from RGB images.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the analyzer."""
        self.model_path = Path(model_path) if model_path else MODEL_DIR / 'teeth_condition_model.h5'
        self.image_size = IMAGE_SIZE
        self.model = None
        self.condition_model = TeethConditionModel()
        
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one."""
        self.condition_model.build_model()
        self.model = self.condition_model.model
        
        if self.model_path.exists():
            try:
                self.condition_model.load_model(str(self.model_path))
                print("Teeth condition model loaded successfully!")
            except Exception as e:
                print(f"Could not load model weights: {e}")
                print("Using untrained model (will provide baseline predictions)")
        else:
            print("No trained teeth condition model found. Using baseline predictions.")
            print(f"Train a model and save to: {self.model_path}")
    
    def preprocess_image(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess image for prediction."""
        # Handle different input types
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    
    def analyze(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze teeth condition from image.
        
        Args:
            image: Input dental image
            
        Returns:
            Comprehensive condition analysis results
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)
        
        # Process results
        results = self._process_predictions(predictions)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        # Build complete analysis
        analysis = {
            'analysis_id': f"condition_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'condition_breakdown': results,
            'summary': self._generate_summary(results, overall_score),
            'recommendations': recommendations,
            'risk_factors': self._identify_risk_factors(results),
            'positive_aspects': self._identify_positive_aspects(results),
            'follow_up': self._get_follow_up_recommendation(overall_score),
            'disclaimer': DISCLAIMER
        }
        
        return analysis
    
    def _process_predictions(self, predictions: List[np.ndarray]) -> Dict:
        """Process model predictions into readable results."""
        results = {}
        categories = list(CONDITION_CATEGORIES.keys())
        
        for i, category in enumerate(categories):
            pred = predictions[i][0]  # Remove batch dimension
            level_idx = int(np.argmax(pred))
            confidence = float(pred[level_idx])
            
            category_info = CONDITION_CATEGORIES[category]
            
            results[category] = {
                'name': category_info['name'],
                'level': category_info['levels'][level_idx],
                'level_index': level_idx,  # 0=worst, 3=best
                'confidence': round(confidence, 3),
                'description': category_info['description'],
                'all_probabilities': {
                    category_info['levels'][j]: round(float(pred[j]), 3)
                    for j in range(len(category_info['levels']))
                }
            }
        
        return results
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall teeth condition score (0-100)."""
        # Weight each category
        weights = {
            'overall_health': 0.25,
            'decay_risk': 0.20,
            'gum_health': 0.20,
            'plaque_level': 0.15,
            'whiteness': 0.10,
            'alignment': 0.10
        }
        
        total_score = 0
        for category, weight in weights.items():
            if category in results:
                # level_index: 0=worst(0%), 3=best(100%)
                level_score = (results[category]['level_index'] / 3) * 100
                confidence = results[category]['confidence']
                
                # Weight by confidence
                weighted_score = level_score * confidence + 50 * (1 - confidence)
                total_score += weighted_score * weight
        
        return round(total_score, 1)
    
    def _generate_summary(self, results: Dict, overall_score: float) -> str:
        """Generate human-readable summary."""
        if overall_score >= 80:
            summary = "Your teeth appear to be in excellent condition. "
        elif overall_score >= 60:
            summary = "Your teeth appear to be in good condition with some areas for improvement. "
        elif overall_score >= 40:
            summary = "Your teeth show signs that need attention. "
        else:
            summary = "Your teeth show significant concerns that require professional attention. "
        
        # Add specific observations
        concerns = []
        positives = []
        
        for category, data in results.items():
            if data['level_index'] <= 1:  # Poor or Fair
                concerns.append(data['name'].lower())
            elif data['level_index'] >= 3:  # Excellent
                positives.append(data['name'].lower())
        
        if concerns:
            summary += f"Areas of concern include: {', '.join(concerns)}. "
        if positives:
            summary += f"Positive aspects: {', '.join(positives)}."
        
        return summary.strip()
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate personalized recommendations based on analysis."""
        recommendations = []
        
        # Overall health recommendations
        if results['overall_health']['level_index'] <= 1:
            recommendations.append("🚨 Schedule a comprehensive dental examination as soon as possible")
        
        # Decay risk recommendations
        decay = results['decay_risk']
        if decay['level_index'] == 0:
            recommendations.extend([
                "⚠️ High cavity risk detected - see a dentist within 1 week",
                "Use prescription-strength fluoride toothpaste",
                "Eliminate sugary foods and acidic drinks"
            ])
        elif decay['level_index'] == 1:
            recommendations.extend([
                "Schedule dental checkup within 2 weeks",
                "Use fluoride mouthwash daily",
                "Reduce sugar intake"
            ])
        
        # Gum health recommendations
        gum = results['gum_health']
        if gum['level_index'] <= 1:
            recommendations.extend([
                "⚠️ Gum issues detected - professional cleaning recommended",
                "Use an antiseptic mouthwash",
                "Floss gently but consistently every day"
            ])
        
        # Plaque recommendations
        plaque = results['plaque_level']
        if plaque['level_index'] <= 1:
            recommendations.extend([
                "Professional dental cleaning needed",
                "Brush for full 2 minutes, twice daily",
                "Consider an electric toothbrush for better plaque removal"
            ])
        
        # Whiteness recommendations
        whiteness = results['whiteness']
        if whiteness['level_index'] <= 1:
            recommendations.extend([
                "Consider professional whitening consultation",
                "Reduce coffee, tea, and red wine consumption",
                "Use whitening toothpaste"
            ])
        
        # Alignment recommendations
        alignment = results['alignment']
        if alignment['level_index'] <= 1:
            recommendations.append("Consider orthodontic consultation for alignment issues")
        
        # General recommendations
        recommendations.extend([
            "Brush teeth twice daily with fluoride toothpaste",
            "Floss daily to remove plaque between teeth",
            "Schedule regular dental checkups every 6 months"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        
        return unique_recs[:10]  # Limit to top 10
    
    def _identify_risk_factors(self, results: Dict) -> List[Dict]:
        """Identify risk factors from analysis."""
        risk_factors = []
        
        for category, data in results.items():
            if data['level_index'] <= 1:  # Poor or Fair
                risk_factors.append({
                    'category': data['name'],
                    'level': data['level'],
                    'severity': 'High' if data['level_index'] == 0 else 'Moderate',
                    'confidence': data['confidence']
                })
        
        return risk_factors
    
    def _identify_positive_aspects(self, results: Dict) -> List[Dict]:
        """Identify positive aspects from analysis."""
        positives = []
        
        for category, data in results.items():
            if data['level_index'] >= 2:  # Good or Excellent
                positives.append({
                    'category': data['name'],
                    'level': data['level'],
                    'confidence': data['confidence']
                })
        
        return positives
    
    def _get_follow_up_recommendation(self, overall_score: float) -> str:
        """Get follow-up recommendation based on score."""
        if overall_score < 30:
            return "🚨 URGENT: Schedule dental appointment within 1 week"
        elif overall_score < 50:
            return "⚠️ Schedule dental appointment within 2 weeks"
        elif overall_score < 70:
            return "Schedule dental checkup within 1 month"
        elif overall_score < 85:
            return "Schedule routine checkup within 3 months"
        else:
            return "Continue regular dental checkups every 6 months"


class TeethConditionTrainer:
    """Training utilities for the teeth condition model."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path('datasets/teeth_condition')
        self.model = TeethConditionModel()
        
    def prepare_dataset(self, images_dir: str, labels_file: str, 
                       validation_split: float = 0.2) -> Tuple:
        """
        Prepare dataset for training.
        
        Expected labels_file format (CSV):
        image_name,overall_health,whiteness,decay_risk,gum_health,plaque_level,alignment
        img001.jpg,2,3,1,2,1,3
        
        Values: 0=Poor, 1=Fair, 2=Good, 3=Excellent
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Load labels
        labels_df = pd.read_csv(labels_file)
        
        images = []
        labels = {cat: [] for cat in CONDITION_CATEGORIES.keys()}
        
        for _, row in labels_df.iterrows():
            img_path = Path(images_dir) / row['image_name']
            if img_path.exists():
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                
                # One-hot encode labels
                for cat in CONDITION_CATEGORIES.keys():
                    label = np.zeros(4)
                    label[int(row[cat])] = 1
                    labels[cat].append(label)
        
        # Convert to arrays
        X = np.array(images)
        y = {cat: np.array(labels[cat]) for cat in labels.keys()}
        
        # Split
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = {cat: y[cat][train_idx] for cat in y.keys()}
        y_val = {cat: y[cat][val_idx] for cat in y.keys()}
        
        return (X_train, y_train), (X_val, y_val)
    
    def train(self, train_data: Tuple, val_data: Tuple, 
              epochs: int = 50, batch_size: int = 32,
              save_path: str = None) -> Dict:
        """Train the model."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Build and compile model
        self.model.build_model()
        self.model.compile_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        if save_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True
                )
            )
        
        # Train
        history = self.model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history


def create_sample_dataset_structure():
    """Create sample dataset structure for training."""
    base_path = Path('datasets/teeth_condition')
    
    # Create directories
    (base_path / 'images').mkdir(parents=True, exist_ok=True)
    
    # Create sample labels CSV
    sample_csv = """image_name,overall_health,whiteness,decay_risk,gum_health,plaque_level,alignment
sample1.jpg,2,3,1,2,1,3
sample2.jpg,1,2,2,1,2,2
sample3.jpg,3,3,3,3,3,3
"""
    
    with open(base_path / 'labels.csv', 'w') as f:
        f.write(sample_csv)
    
    # Create README
    readme = """# Teeth Condition Dataset

## Structure
- images/: Place dental images here
- labels.csv: Condition labels for each image

## Label Format
Each row: image_name, overall_health, whiteness, decay_risk, gum_health, plaque_level, alignment

## Values
- 0: Poor / Severe Issues
- 1: Fair / Moderate Issues  
- 2: Good / Minor Issues
- 3: Excellent / Healthy

## Training
```python
from teeth_condition_analyzer import TeethConditionTrainer

trainer = TeethConditionTrainer()
train_data, val_data = trainer.prepare_dataset('datasets/teeth_condition/images', 'datasets/teeth_condition/labels.csv')
trainer.train(train_data, val_data, epochs=50, save_path='models/teeth_condition_model.h5')
```
"""
    
    with open(base_path / 'README.md', 'w') as f:
        f.write(readme)
    
    print(f"✅ Dataset structure created at: {base_path}")
    return base_path


# ============================================
# CLI INTERFACE
# ============================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='EfficientNet Teeth Condition Analyzer')
    parser.add_argument('--analyze', '-a', help='Analyze image')
    parser.add_argument('--setup', action='store_true', help='Create dataset structure')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--data', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    
    args = parser.parse_args()
    
    if args.setup:
        create_sample_dataset_structure()
    elif args.analyze:
        analyzer = TeethConditionAnalyzer()
        result = analyzer.analyze(args.analyze)
        
        print("\n🦷 TEETH CONDITION ANALYSIS")
        print("=" * 50)
        print(f"Overall Score: {result['overall_score']}/100")
        print(f"\n📋 Summary: {result['summary']}")
        
        print("\n📊 Condition Breakdown:")
        for cat, data in result['condition_breakdown'].items():
            print(f"  • {data['name']}: {data['level']} ({data['confidence']:.0%})")
        
        print("\n⚠️ Risk Factors:")
        for risk in result['risk_factors']:
            print(f"  • {risk['category']}: {risk['level']} ({risk['severity']})")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(result['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📅 Follow-up: {result['follow_up']}")
        print(f"\n⚠️ {result['disclaimer']}")
    elif args.train:
        if not args.data:
            print("Please provide --data directory")
        else:
            trainer = TeethConditionTrainer(args.data)
            train_data, val_data = trainer.prepare_dataset(
                f'{args.data}/images',
                f'{args.data}/labels.csv'
            )
            trainer.train(train_data, val_data, epochs=args.epochs,
                         save_path='models/teeth_condition_model.h5')
    else:
        parser.print_help()
        print("\n📋 Quick Start:")
        print("  1. Setup dataset:  py -3.11 teeth_condition_analyzer.py --setup")
        print("  2. Analyze image:  py -3.11 teeth_condition_analyzer.py --analyze image.jpg")
        print("  3. Train model:    py -3.11 teeth_condition_analyzer.py --train --data datasets/teeth_condition")
