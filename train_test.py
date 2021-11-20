"""
The code implementation of the paper:
A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import argparse

from pie_intent import PIEIntent
from pie_predict import PIEPredict

from pie_data import PIE

import keras.backend as K
import tensorflow as tf

from prettytable import PrettyTable

from joint_model import getRecall, getPrecision

# dim_ordering = K.image_dim_ordering()
DATA_PATH = '/home/y_feng/workspace6/datasets/PIE_dataset'

#train models with data up to critical point
#only for PIE
#train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(args):
    train_test = args.train_test
    epochs = args.intent_epochs
    batch_size = args.batch_size
    obs_len = args.obs_len
    pred_len = args.pred_len
    data_split_type = args.data_split_type
    traj_only = args.traj_only

    balance_train = args.balance_train
    balance_val = args.balance_val
    data_opts = {'fstride': 1,
            'sample_type': 'all', 
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': data_split_type,  #  kfold, random, default
            'seq_type': 'intention', #  crossing , intention
            'min_track_size': 0, #  discard tracks that are shorter
            'max_size_observe': 15,  # number of observation frames
            'max_size_predict': 5,  # number of prediction frames
            'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
            'balance': True,  # balance the training and testing samples
            'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
            'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
            'encoder_input_type': [],
            'decoder_input_type': ['bbox'],
            'output_type': ['intention_binary']
            }


    t = PIEIntent(num_hidden_units=128,
                  regularizer_val=0.001,
                  lstm_dropout=0.4,
                  lstm_recurrent_dropout=0.2,
                  convlstm_num_filters=64,
                  convlstm_kernel_size=2)

    saved_files_path = ''

    imdb = PIE(data_path=DATA_PATH)

    pretrained_model_path = 'data/pie/intention/context_loc_pretrained'

    if train_test < 2:  # Train
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        if balance_val:
            beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        if balance_train:
            beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

        saved_files_path = t.train(data_train=beh_seq_train,
                                   data_val=beh_seq_val,
                                   epochs=epochs,
                                   loss=['binary_crossentropy'],
                                   metrics=['accuracy', getRecall, getPrecision],
                                   batch_size=128,
                                   optimizer_type='rmsprop',
                                   data_opts=data_opts,
                                   gpu=1)

        print(data_opts['seq_overlap_rate'])

    if train_test > 0:  # Test
        if saved_files_path == '':
            saved_files_path = pretrained_model_path
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        acc, f1 = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False)
        
        t = PrettyTable(['Acc', 'F1'])
        t.title = 'Intention model (local_context + bbox)'
        t.add_row([acc, f1])
        
        print(t)

        K.clear_session()
        tf.reset_default_graph()
        return saved_files_path

def train_predict(dataset='pie',
                  args=None, 
                  intent_model_path='data/pie/intention/context_loc_pretrained'):
    train_test = args.train_test
    epochs = args.traj_epochs
    batch_size = args.batch_size
    obs_len = args.obs_len
    pred_len = args.pred_len
    data_split_type = args.data_split_type
    traj_only = args.traj_only

    balance_train = args.balance_train
    balance_val = args.balance_val
    balance_test = args.balance_test
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': data_split_type,  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

    t = PIEPredict()
    # pie_path = os.environ.copy()['PIE_PATH']

    if dataset == 'pie':
        imdb = PIE(data_path=DATA_PATH)

    traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': ['intention_prob', 'obd_speed'],
                       'prediction_type': ['bbox']
                       }

    speed_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['obd_speed'], 
                       'dec_input_type': [],
                       'prediction_type': ['obd_speed'] 
                       }

    traj_model_path = 'data/pie/trajectory/loc_intent_speed_pretrained'
    speed_model_path = 'data/pie/speed/speed_pretrained'

    if train_test < 2:  # Train
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        if balance_val:
            beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        if balance_train:
            beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')
        traj_model_path = t.train(beh_seq_train, beh_seq_val, epochs=epochs, traj_only=traj_only, **traj_model_opts)
        if not traj_only:
            speed_model_path = t.train(beh_seq_train, beh_seq_val, epochs=epochs, **speed_model_opts)

    if train_test > 0:  # Test
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        if balance_test:
            beh_seq_test = imdb.balance_samples_count(beh_seq_test, label_type='intention_binary')
        perf_final = t.test_final(beh_seq_test,
                                  traj_model_path=traj_model_path, 
                                  speed_model_path=speed_model_path,
                                  intent_model_path=intent_model_path,
                                  traj_only=traj_only)

        t = PrettyTable(['MSE', 'C_MSE'])
        t.title = 'Trajectory prediction model (loc + PIE_intent + PIE_speed)'
        t.add_row([perf_final['mse-45'], perf_final['c-mse-45']])
        
        print(t)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test', type=int, default=1)
    parser.add_argument('--intent_epochs', type=int, default=300)
    parser.add_argument('--traj_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--obs_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=45)

    parser.add_argument('--balance_train', type=bool, default=True)
    parser.add_argument('--balance_val', type=bool, default=False)
    parser.add_argument('--balance_test', type=bool, default=False)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--data_split_type', type=str, default='default')

    parser.add_argument('--traj_only', type=bool, default=False)

    args = parser.parse_args()
    if not args.traj_only and args.intent_epochs > 0:
        intent_model_path = train_intent(args)
        train_predict(args=args, intent_model_path=intent_model_path)
    else:
        train_predict(args=args)


if __name__ == '__main__':

    main()