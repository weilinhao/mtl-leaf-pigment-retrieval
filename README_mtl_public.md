# Public Reproducible MTL Code

This repository provides a minimal public implementation of the multi-task learning (MTL) model used for simultaneous retrieval of leaf chlorophyll and carotenoids.

## Requirements

- Python 3
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn

## Input format

The script expects plain text files for spectra and targets.

For the simulated training target file:
- column 2: chlorophyll (Cab)
- column 3: carotenoids (Car)

For each real test target file:
- column 1: chlorophyll (Cab)
- column 2: carotenoids (Car)

Spectral columns are sliced using `--x-start` and `--x-end`.

## Example

```bash
python train_mtl_reproducible_public.py \
  --train-x path/to/X_train.txt \
  --train-y path/to/y_train.txt \
  --test ANGERS path/to/X_test.txt path/to/y_test.txt False \
  --output-dir outputs
```

To add multiple test datasets, repeat `--test`.

## Outputs

The output directory will contain:

- `mtl_model.h5`: trained model
- `run_config.json`: run configuration
- `metrics.csv`: dataset-wise RMSE and R²
- `loss_curve_mtl.csv`: training history, if enabled
- `predictions/`: per-sample predictions, if enabled
