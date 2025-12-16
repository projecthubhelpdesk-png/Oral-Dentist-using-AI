# Teeth Condition Dataset

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
