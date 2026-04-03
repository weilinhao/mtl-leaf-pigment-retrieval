import os
import gc
import json
import time
import random
import argparse
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import lecun_normal


PUBLIC_CONFIG = {
    'seed': 1,
    'epochs': 100,
    'batch_size': 128,
    'train_size': 10000,
    'x_start': 0,
    'x_end': 512,
    'early_stop': True,
    'early_stop_patience': 50,
    'sep_train': None,
    'sep_test': None,
    'output_dir': 'outputs',
    'save_predictions': True,
    'save_loss_curve': True,
    'net_conf': {
        'common_hidden': [256],
        'bn': {'common_bn': 0.01, 'cab_bn': None, 'car_bn': None},
        'hidden': {'cab_hidden': [256, 128], 'car_hidden': [256, 128]},
        'lr': 1e-4,
        'loss_weight': {'cab_w': 0.3, 'car_w': 0.7},
    },
}

INIT_LECUN = lecun_normal()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Public reproducible implementation of the MTL leaf pigment retrieval model.'
    )
    parser.add_argument('--train-x', required=True, help='Path to the training spectra file.')
    parser.add_argument('--train-y', required=True, help='Path to the training target file.')
    parser.add_argument(
        '--test',
        nargs=4,
        metavar=('NAME', 'X_PATH', 'Y_PATH', 'IS_SIMULATED'),
        action='append',
        default=[],
        help='Add one test dataset: NAME X_PATH Y_PATH IS_SIMULATED(True/False). Repeat this option for multiple test sets.',
    )
    parser.add_argument('--output-dir', default=PUBLIC_CONFIG['output_dir'], help='Directory for outputs.')
    parser.add_argument('--train-size', type=int, default=PUBLIC_CONFIG['train_size'], help='Number of training samples to use.')
    parser.add_argument('--x-start', type=int, default=PUBLIC_CONFIG['x_start'], help='Start column index for spectral input slicing.')
    parser.add_argument('--x-end', type=int, default=PUBLIC_CONFIG['x_end'], help='End column index for spectral input slicing.')
    parser.add_argument('--epochs', type=int, default=PUBLIC_CONFIG['epochs'], help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=PUBLIC_CONFIG['batch_size'], help='Training batch size.')
    parser.add_argument('--seed', type=int, default=PUBLIC_CONFIG['seed'], help='Random seed.')
    parser.add_argument('--sep-train', default=PUBLIC_CONFIG['sep_train'], help='Optional separator for training files.')
    parser.add_argument('--sep-test', default=PUBLIC_CONFIG['sep_test'], help='Optional separator for test files.')
    parser.add_argument('--no-early-stop', action='store_true', help='Disable early stopping.')
    parser.add_argument('--no-save-predictions', action='store_true', help='Do not save per-sample prediction files.')
    parser.add_argument('--no-save-loss-curve', action='store_true', help='Do not save the training loss curve.')
    parser.add_argument('--print-example', action='store_true', help='Print an example command and exit.')
    return parser.parse_args()


def print_example():
    example = (
        'python train_mtl_reproducible_public.py '
        '--train-x path/to/X_train.txt '
        '--train-y path/to/y_train.txt '
        '--test ANGERS path/to/X_test.txt path/to/y_test.txt False '
        '--output-dir outputs'
    )
    print(example)


def str_to_bool(value):
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def setup_deterministic_environment(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def reset_thorough(seed):
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
            tf.compat.v1.reset_default_graph()
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gc.collect()


def _ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



def _append_csv(path, row_dict):
    header = not os.path.exists(path)
    pd.DataFrame([row_dict]).to_csv(path, index=False, mode='a', header=header)


def _read_auto(path, sep_hint=None):
    if not os.path.exists(path):
        raise FileNotFoundError('Data file not found: {}'.format(path))

    def read_with(sep):
        try:
            return pd.read_csv(path, sep=sep, header=None, engine='python', on_bad_lines='skip')
        except TypeError:
            return pd.read_csv(path, sep=sep, header=None, engine='python', error_bad_lines=False, warn_bad_lines=False)

    if sep_hint is not None:
        return read_with(sep_hint)

    try:
        df = read_with(',')
        if df.shape[1] == 1:
            raise ValueError
        return df
    except Exception:
        return read_with(r'\s+')


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return float(r2), float(rmse)


def get_mtl_data(pair, x_init, x_end, size, mode,
                 sep_hint=None, scaler_y1=None, scaler_y2=None,
                 is_simulated=True):
    xf = _read_auto(pair[0], sep_hint)
    yf = _read_auto(pair[1], sep_hint)
    if size > 0:
        x = xf.values[:size, x_init:x_end]
        y = yf.values[:size, :]
    else:
        x = xf.values[:, x_init:x_end]
        y = yf.values

    if is_simulated:
        y1 = y[:, 1]
        y2 = y[:, 2]
    else:
        y1 = y[:, 0]
        y2 = y[:, 1]

    y1 = y1.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)

    if scaler_y1 is not None:
        y1 = scaler_y1.fit_transform(y1) if mode == 'train' else scaler_y1.transform(y1)
    if scaler_y2 is not None:
        y2 = scaler_y2.fit_transform(y2) if mode == 'train' else scaler_y2.transform(y2)
    return x, y1.flatten(), y2.flatten()


def build_mtl(input_dim, net_conf):
    wd_common = net_conf['bn']['common_bn']
    wd_cab = net_conf['bn']['cab_bn']
    wd_car = net_conf['bn']['car_bn']
    common_hidden = net_conf['common_hidden']
    hidden = net_conf['hidden']
    lr = net_conf['lr']
    loss_weight = net_conf['loss_weight']

    inp = Input(shape=(input_dim,))
    x = inp
    for h in common_hidden:
        x = Dense(
            h,
            activation='selu',
            kernel_initializer=INIT_LECUN,
            kernel_regularizer=l2(wd_common) if wd_common else None,
        )(x)

    cab = x
    for h in hidden['cab_hidden']:
        cab = Dense(
            h,
            activation='selu',
            kernel_initializer=INIT_LECUN,
            kernel_regularizer=l2(wd_cab) if wd_cab else None,
        )(cab)
    cab = Dense(1, activation='linear', kernel_initializer=INIT_LECUN, name='cab_output')(cab)

    car = x
    for h in hidden['car_hidden']:
        car = Dense(
            h,
            activation='selu',
            kernel_initializer=INIT_LECUN,
            kernel_regularizer=l2(wd_car) if wd_car else None,
        )(car)
    car = Dense(1, activation='linear', kernel_initializer=INIT_LECUN, name='car_output')(car)

    model = Model(inputs=inp, outputs=[cab, car])
    model.compile(
        loss={'cab_output': 'mse', 'car_output': 'mse'},
        loss_weights={'cab_output': loss_weight['cab_w'], 'car_output': loss_weight['car_w']},
        optimizer=Adam(learning_rate=lr),
        metrics=['mse'],
    )
    return model


def train_mtl(config, train_pair):
    tf.keras.backend.clear_session()
    scaler_y1, scaler_y2 = MinMaxScaler(), MinMaxScaler()

    x, y1, y2 = get_mtl_data(
        train_pair,
        config['x_start'],
        config['x_end'],
        config['train_size'],
        'train',
        sep_hint=config['sep_train'],
        scaler_y1=scaler_y1,
        scaler_y2=scaler_y2,
        is_simulated=True,
    )
    tr_x, va_x, tr_y1, va_y1, tr_y2, va_y2 = train_test_split(
        x, y1, y2, test_size=0.25, random_state=42, shuffle=True
    )
    model = build_mtl(x.shape[1], config['net_conf'])
    callbacks = []
    if config['early_stop']:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=config['early_stop_patience'],
                restore_best_weights=True,
                verbose=2,
            )
        )

    history = model.fit(
        tr_x,
        {'cab_output': tr_y1, 'car_output': tr_y2},
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_data=(va_x, {'cab_output': va_y1, 'car_output': va_y2}),
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    if config['save_loss_curve']:
        _ensure_dir(config['output_dir'])
        h = history.history
        pd.DataFrame({
            'epoch': np.arange(1, len(h['loss']) + 1),
            'loss': h['loss'],
            'val_loss': h.get('val_loss'),
            'cab_loss': h.get('cab_output_loss'),
            'val_cab_loss': h.get('val_cab_output_loss'),
            'car_loss': h.get('car_output_loss'),
            'val_car_loss': h.get('val_car_output_loss'),
        }).to_csv(os.path.join(config['output_dir'], 'loss_curve_mtl.csv'), index=False)

    return model, scaler_y1, scaler_y2


def eval_mtl(model, pair, dataset_name, config, scaler_y1, scaler_y2, is_simulated=False):
    x_s, y1_s_s, y2_s_s = get_mtl_data(
        pair,
        config['x_start'],
        config['x_end'],
        0,
        'test',
        sep_hint=config['sep_test'],
        scaler_y1=scaler_y1,
        scaler_y2=scaler_y2,
        is_simulated=is_simulated,
    )
    y1_p_s, y2_p_s = model.predict(x_s, verbose=0)

    y1_obs = scaler_y1.inverse_transform(y1_s_s.reshape(-1, 1)).flatten()
    y2_obs = scaler_y2.inverse_transform(y2_s_s.reshape(-1, 1)).flatten()
    y1_pred = scaler_y1.inverse_transform(y1_p_s.reshape(-1, 1)).flatten()
    y2_pred = scaler_y2.inverse_transform(y2_p_s.reshape(-1, 1)).flatten()

    r2_cab, rmse_cab = _metrics(y1_obs, y1_pred)
    r2_car, rmse_car = _metrics(y2_obs, y2_pred)

    metrics_path = os.path.join(config['output_dir'], 'metrics.csv')
    _append_csv(metrics_path, {
        'dataset': dataset_name,
        'target': 'cab',
        'rmse': rmse_cab,
        'r2': r2_cab,
    })
    _append_csv(metrics_path, {
        'dataset': dataset_name,
        'target': 'car',
        'rmse': rmse_car,
        'r2': r2_car,
    })

    if config['save_predictions']:
        pred_dir = os.path.join(config['output_dir'], 'predictions')
        _ensure_dir(pred_dir)
        pd.DataFrame({'y_true': y1_obs, 'y_pred': y1_pred}).to_csv(
            os.path.join(pred_dir, '{}__MTL__cab.csv'.format(dataset_name)), index=False
        )
        pd.DataFrame({'y_true': y2_obs, 'y_pred': y2_pred}).to_csv(
            os.path.join(pred_dir, '{}__MTL__car.csv'.format(dataset_name)), index=False
        )


def build_runtime_config(args):
    config = json.loads(json.dumps(PUBLIC_CONFIG))
    config['seed'] = args.seed
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['train_size'] = args.train_size
    config['x_start'] = args.x_start
    config['x_end'] = args.x_end
    config['sep_train'] = args.sep_train
    config['sep_test'] = args.sep_test
    config['output_dir'] = args.output_dir
    config['early_stop'] = not args.no_early_stop
    config['save_predictions'] = not args.no_save_predictions
    config['save_loss_curve'] = not args.no_save_loss_curve
    config['train_pair'] = (args.train_x, args.train_y)
    config['test_list'] = [
        (name, x_path, y_path, str_to_bool(is_sim)) for name, x_path, y_path, is_sim in args.test
    ]
    return config


def save_run_config(config):
    _ensure_dir(config['output_dir'])
    serializable = {
        'train_x_path': config['train_pair'][0],
        'train_y_path': config['train_pair'][1],
        'test_list': config['test_list'],
        'seed': config['seed'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'train_size': config['train_size'],
        'x_start': config['x_start'],
        'x_end': config['x_end'],
        'early_stop': config['early_stop'],
        'early_stop_patience': config['early_stop_patience'],
        'net_conf': config['net_conf'],
    }
    with open(os.path.join(config['output_dir'], 'run_config.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def run_once(config):
    save_run_config(config)
    model, sy1, sy2 = train_mtl(config, config['train_pair'])
    model.save(os.path.join(config['output_dir'], 'mtl_model.h5'))

    metrics_path = os.path.join(config['output_dir'], 'metrics.csv')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    for dataset_name, x_path, y_path, is_simulated in config['test_list']:
        eval_mtl(
            model,
            (x_path, y_path),
            dataset_name,
            config,
            sy1,
            sy2,
            is_simulated=is_simulated,
        )


def main():
    args = parse_args()
    if args.print_example:
        print_example()
        return

    config = build_runtime_config(args)
    setup_deterministic_environment(config['seed'])
    reset_thorough(config['seed'])
    run_once(config)


if __name__ == '__main__':
    main()
