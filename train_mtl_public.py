import os
import random
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import lecun_normal
from joblib import dump

SEED = 1
EPOCHS = 100
BATCH_SIZE = 128
VALID_SIZE = 0.25
EARLY_STOPPING = True
PATIENCE = 50
LEARNING_RATE = 1e-4
COMMON_HIDDEN = [256]
CAB_HIDDEN = [256, 128]
CAR_HIDDEN = [256, 128]
L2_SHARED = 0.01
L2_CAB = None
L2_CAR = None
LOSS_WEIGHT_CAB = 0.3
LOSS_WEIGHT_CAR = 0.7
EXPECTED_INPUT_DIM = 512
TRAIN_SIZE = 10000
TRAIN_TARGET_COLS = (1, 2)
REAL_TARGET_COLS = (0, 1)
SIM_TARGET_COLS = (1, 2)
DEFAULT_OUTPUT_DIR = "mtl_release_outputs"

EXAMPLE_SETTINGS = {
    "train_x": "path/to/train_X.txt",
    "train_y": "path/to/train_y.txt",
    "train_sep": None,
    "x_start": 0,
    "x_end": None,
    "test_sets": [
        {"name": "ANGERS", "x": "path/to/angers_X.txt", "y": "path/to/angers_y.txt", "is_simulated": False, "sep": None},
        {"name": "NX", "x": "path/to/nx_X.txt", "y": "path/to/nx_y.txt", "is_simulated": False, "sep": None}
    ]
}

os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

INIT = lecun_normal()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_auto(path, sep_hint=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    def read_with(sep):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine="python", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, sep=sep, header=None, engine="python", error_bad_lines=False, warn_bad_lines=False)

    if sep_hint is not None:
        return read_with(sep_hint)

    try:
        df = read_with(",")
        if df.shape[1] == 1:
            raise ValueError
        return df
    except Exception:
        return read_with(r"\s+")


def slice_x(x, x_start=0, x_end=None):
    return x[:, x_start:x_end] if x_end is not None else x[:, x_start:]


def load_xy(x_path, y_path, x_start=0, x_end=None, size=None, sep=None):
    x_df = read_auto(x_path, sep)
    y_df = read_auto(y_path, sep)
    x = x_df.values
    y = y_df.values
    if size is not None and size > 0:
        x = x[:size]
        y = y[:size]
    x = slice_x(x, x_start, x_end).astype(np.float32)
    y = y.astype(np.float32)
    return x, y


def pick_targets(y, cols):
    y1 = y[:, cols[0]].reshape(-1, 1)
    y2 = y[:, cols[1]].reshape(-1, 1)
    return y1, y2


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = inp
    for h in COMMON_HIDDEN:
        x = Dense(h, activation="selu", kernel_initializer=INIT, kernel_regularizer=l2(L2_SHARED) if L2_SHARED else None)(x)

    cab = x
    for h in CAB_HIDDEN:
        cab = Dense(h, activation="selu", kernel_initializer=INIT, kernel_regularizer=l2(L2_CAB) if L2_CAB else None)(cab)
    cab = Dense(1, activation="linear", kernel_initializer=INIT, name="cab_output")(cab)

    car = x
    for h in CAR_HIDDEN:
        car = Dense(h, activation="selu", kernel_initializer=INIT, kernel_regularizer=l2(L2_CAR) if L2_CAR else None)(car)
    car = Dense(1, activation="linear", kernel_initializer=INIT, name="car_output")(car)

    model = Model(inputs=inp, outputs=[cab, car])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={"cab_output": "mse", "car_output": "mse"},
        loss_weights={"cab_output": LOSS_WEIGHT_CAB, "car_output": LOSS_WEIGHT_CAR},
    )
    return model


def train_model(train_x_path, train_y_path, train_sep=None, x_start=0, x_end=None, output_dir=DEFAULT_OUTPUT_DIR):
    tf.keras.backend.clear_session()

    x, y = load_xy(train_x_path, train_y_path, x_start=x_start, x_end=x_end, size=TRAIN_SIZE, sep=train_sep)
    if EXPECTED_INPUT_DIM is not None and x.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(f"Expected input_dim={EXPECTED_INPUT_DIM}, but got {x.shape[1]}")

    y1, y2 = pick_targets(y, TRAIN_TARGET_COLS)
    scaler_y1 = MinMaxScaler()
    scaler_y2 = MinMaxScaler()
    y1 = scaler_y1.fit_transform(y1)
    y2 = scaler_y2.fit_transform(y2)

    tr_x, va_x, tr_y1, va_y1, tr_y2, va_y2 = train_test_split(
        x, y1, y2, test_size=VALID_SIZE, random_state=42, shuffle=True
    )

    model = build_model(x.shape[1])
    callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=2)] if EARLY_STOPPING else []

    history = model.fit(
        tr_x,
        {"cab_output": tr_y1, "car_output": tr_y2},
        validation_data=(va_x, {"cab_output": va_y1, "car_output": va_y2}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )

    ensure_dir(output_dir)
    pd.DataFrame(history.history).to_csv(os.path.join(output_dir, "train_history.csv"), index=False)
    model.save(os.path.join(output_dir, "mtl_model.h5"))
    dump(scaler_y1, os.path.join(output_dir, "scaler_cab.joblib"))
    dump(scaler_y2, os.path.join(output_dir, "scaler_car.joblib"))

    meta = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "valid_size": VALID_SIZE,
        "learning_rate": LEARNING_RATE,
        "common_hidden": COMMON_HIDDEN,
        "cab_hidden": CAB_HIDDEN,
        "car_hidden": CAR_HIDDEN,
        "l2_shared": L2_SHARED,
        "l2_cab": L2_CAB,
        "l2_car": L2_CAR,
        "loss_weight": {"cab": LOSS_WEIGHT_CAB, "car": LOSS_WEIGHT_CAR},
        "expected_input_dim": EXPECTED_INPUT_DIM,
        "train_target_cols": TRAIN_TARGET_COLS,
        "real_target_cols": REAL_TARGET_COLS,
        "sim_target_cols": SIM_TARGET_COLS,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model, scaler_y1, scaler_y2


def evaluate_one(model, scaler_y1, scaler_y2, ds, x_start=0, x_end=None, output_dir=DEFAULT_OUTPUT_DIR):
    x, y = load_xy(ds["x"], ds["y"], x_start=x_start, x_end=x_end, size=None, sep=ds.get("sep", None))
    cols = SIM_TARGET_COLS if ds.get("is_simulated", False) else REAL_TARGET_COLS
    y1, y2 = pick_targets(y, cols)

    pred_y1_s, pred_y2_s = model.predict(x, verbose=0)
    y1_true = y1.reshape(-1)
    y2_true = y2.reshape(-1)
    y1_pred = scaler_y1.inverse_transform(pred_y1_s.reshape(-1, 1)).reshape(-1)
    y2_pred = scaler_y2.inverse_transform(pred_y2_s.reshape(-1, 1)).reshape(-1)

    m1 = metrics(y1_true, y1_pred)
    m2 = metrics(y2_true, y2_pred)

    pred_dir = os.path.join(output_dir, "predictions")
    ensure_dir(pred_dir)
    pd.DataFrame({"y_true": y1_true, "y_pred": y1_pred}).to_csv(os.path.join(pred_dir, f"{ds['name']}_cab.csv"), index=False)
    pd.DataFrame({"y_true": y2_true, "y_pred": y2_pred}).to_csv(os.path.join(pred_dir, f"{ds['name']}_car.csv"), index=False)

    rows = [
        {
            "dataset": ds["name"],
            "target": "cab",
            **m1,
        },
        {
            "dataset": ds["name"],
            "target": "car",
            **m2,
        },
    ]
    return rows


def parse_datasets(args):
    items = []
    for spec in args.test:
        name, x_path, y_path, mode = spec
        items.append({
            "name": name,
            "x": x_path,
            "y": y_path,
            "is_simulated": mode.lower() == "sim",
            "sep": None,
        })
    return items


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-x", required=False)
    parser.add_argument("--train-y", required=False)
    parser.add_argument("--train-sep", default=None)
    parser.add_argument("--x-start", type=int, default=0)
    parser.add_argument("--x-end", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test", nargs=4, action="append", metavar=("NAME", "X_PATH", "Y_PATH", "MODE"))
    parser.add_argument("--print-example", action="store_true")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.print_example:
        print(json.dumps(EXAMPLE_SETTINGS, ensure_ascii=False, indent=2))
        return

    if not args.train_x or not args.train_y:
        raise ValueError("--train-x and --train-y are required")

    test_sets = parse_datasets(args)
    model, scaler_y1, scaler_y2 = train_model(
        train_x_path=args.train_x,
        train_y_path=args.train_y,
        train_sep=args.train_sep,
        x_start=args.x_start,
        x_end=args.x_end,
        output_dir=args.output_dir,
    )

    all_rows = []
    for ds in test_sets:
        all_rows.extend(evaluate_one(model, scaler_y1, scaler_y2, ds, x_start=args.x_start, x_end=args.x_end, output_dir=args.output_dir))

    if all_rows:
        pd.DataFrame(all_rows)[["dataset", "target", "rmse", "r2"]].to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)


if __name__ == "__main__":
    main()
