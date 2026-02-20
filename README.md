# Desalination Efficiency Prediction (Deep Learning)

This repository contains a TensorFlow/Keras training script that reads parameters from an Excel file and predicts desalination efficiency.

## Data split
The script uses:
- **70%** training data
- **15%** validation data
- **15%** testing data

## Usage
```bash
python train_desalination_model.py \
  --excel path/to/your_data.xlsx \
  --target desalination_efficiency \
  --sheet 0 \
  --epochs 300 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --model-out desalination_efficiency_model.keras
```

## Notes
- The target column defaults to `desalination_efficiency`.
- Only numeric feature columns are used.
- Numeric missing values are imputed with the median and then standardized.
- Early stopping is enabled using validation loss.
