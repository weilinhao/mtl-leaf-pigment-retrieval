This public release contains only the dual-task MTL training and testing pipeline proposed in the paper. It does not depend on any external configuration file.

Fixed configuration:
- Input: 400–911 nm reflectance (512 bands)
- Shared layers: [256]
- Cab branch: [256, 128]
- Car branch: [256, 128]
- Learning rate: 1e-4
- Batch size: 128
- Epochs: 100
- L2 on shared layers: 0.01
- Loss weights: Cab = 0.3, Car = 0.7
- Training label scaling: MinMax

Example usage:
python train_mtl_public.py \
  --train-x path/to/train_X.txt \
  --train-y path/to/train_y.txt \
  --test ANGERS path/to/angers_X.txt path/to/angers_y.txt real \
  --test NX path/to/nx_X.txt path/to/nx_y.txt real

Notes:
- For the synthetic training set, the script reads Cab and Car from columns 2 and 3 of the y file by default (0-based indices 1 and 2).
- For in-situ test sets, the script reads Cab and Car from columns 1 and 2 of the y file by default (0-based indices 0 and 1).
- If a test set is also synthetic, use `sim` as the last field of the `--test` argument.
- Outputs include the trained model, scaler, training history, prediction files, and metric tables.
