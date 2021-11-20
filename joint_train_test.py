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

from joint_model import PIEIntent
from pie_predict import PIEPredict

from pie_data import PIE

import keras.backend as K
import tensorflow as tf

from prettytable import PrettyTable

from joint_model import getRecall, getPrecision

# dim_ordering = K.image_dim_ordering()
DATA_PATH = '/home/y_feng/workspace6/datasets/PIE_dataset'


def train_joint(args):
    train_test = args.train_test
    epochs = args.epochs
    batch_size = args.batch_size
    obs_len = args.obs_len
    pred_len = args.pred_len
    traj_loss_weight = args.traj_loss_weight
    intent_loss_weight = args.intent_loss_weight
    speed_loss_weight = args.speed_loss_weight
    balance_train = args.balance_train
    balance_val = args.balance_val
    balance_test = args.balance_test
    early_stop = args.early_stop
    data_split_type = args.data_split_type
    traj_only = args.traj_only
    test_by_val = args.test_by_val

    
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': data_split_type,  # kfold, random, default. default: set03 for test
                 'seq_type': 'intention',  # crossing , intention
                 'min_track_size': 0,  # discard tracks that are shorter
                 'max_size_observe': obs_len,  # number of observation frames
                 'max_size_predict': pred_len,  # number of prediction frames
                 'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                 'balance': True,  # balance the training and testing samples
                 'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                 'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                 'seq_type': 'trajectory',
                 'encoder_input_type': ['bbox', 'obd_speed'],
                 'decoder_input_type': [],
                 'output_type': ['intention_binary', 'bbox']
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
        if test_by_val:
            beh_seq_val = imdb.generate_data_trajectory_sequence('test', **data_opts)
        else:
            beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        if balance_val:
            beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

        # beh_seq_train: dict{img path, ped ID, bbox, ...}
        #    img path: list
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        if balance_train:
            beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

        if traj_only:
            saved_files_path = t.train_traj_only(data_train=beh_seq_train,
                                                 data_val=beh_seq_val,
                                                 epochs=epochs,
                                                 loss=['mse'],
                                                 metrics=['mse'],
                                                 batch_size=batch_size,
                                                 optimizer_type='rmsprop',
                                                 data_opts=data_opts,
                                                 gpu=1,
                                                 early_stop=early_stop)
        else:
            saved_files_path = t.train(data_train=beh_seq_train,
                                       data_val=beh_seq_val,
                                       epochs=epochs,
                                       loss={'intent_fc': 'binary_crossentropy','speed_dec_fc': 'mse', 'traj_dec_fc': 'mse'},
                                       loss_weights={'intent_fc': intent_loss_weight,'speed_dec_fc': speed_loss_weight, 'traj_dec_fc': traj_loss_weight},
                                       metrics={'intent_fc': ['acc', getRecall, getPrecision],'speed_dec_fc': 'mse', 'traj_dec_fc': 'mse'},
                                       batch_size=batch_size,
                                       optimizer_type='rmsprop',
                                       data_opts=data_opts,
                                       gpu=1,
                                       early_stop=early_stop)

        print(data_opts['seq_overlap_rate'])
        print('Training done, joint model saved in ' + saved_files_path)
    if train_test > 0:  # Test
        if saved_files_path == '':
            saved_files_path = pretrained_model_path
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        if balance_test:
            beh_seq_test = imdb.balance_samples_count(beh_seq_test, label_type='intention_binary')

        intent_acc, intent_recall, intent_precision, speed_mse, traj_mse = t.test(beh_seq_test, data_opts, saved_files_path, False)

        t = PrettyTable(['acc', 'recall', 'precision', 'speed', 'traj'])
        t.title = 'Joint model (intent + speed + bbox)'
        t.add_row([intent_acc, intent_recall, intent_precision, speed_mse, traj_mse])

        print(t)

        K.clear_session()
        tf.reset_default_graph()
        return saved_files_path


def train_predict(dataset='pie',
                  train_test=2, 
                  intent_model_path='data/pie/intention/context_loc_pretrained'):
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

    t = PIEPredict()
    pie_path = ''

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

    # train
    if train_test < 2:
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        traj_model_path = t.train(beh_seq_train, beh_seq_val, **traj_model_opts)
        speed_model_path = t.train(beh_seq_train, beh_seq_val, **speed_model_opts)

    # test
    if train_test > 0:
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

        perf_final = t.test_final(beh_seq_test,
                                  traj_model_path=traj_model_path, 
                                  speed_model_path=speed_model_path,
                                  intent_model_path=intent_model_path)

        t = PrettyTable(['MSE', 'C_MSE'])
        t.title = 'Trajectory prediction model (loc + PIE_intent + PIE_speed)'
        t.add_row([perf_final['mse-45'], perf_final['c-mse-45']])
        
        print(t)

#train models with data up to critical point
#only for PIE
#train_test = 0 (train only), 1 (train-test), 2 (test only)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--obs_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=45)

    parser.add_argument('--traj_loss_weight', type=float, default=1)
    parser.add_argument('--intent_loss_weight', type=float, default=1)
    parser.add_argument('--speed_loss_weight', type=float, default=1)

    parser.add_argument('--balance_train', type=bool, default=True)
    parser.add_argument('--balance_val', type=bool, default=False)
    parser.add_argument('--balance_test', type=bool, default=False)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--data_split_type', type=str, default='default')

    parser.add_argument('--traj_only', type=bool, default=False)
    parser.add_argument('--test_by_val', type=bool, default=False)

    args = parser.parse_args()
    train_joint(args=args)
    
    #   train_predict(dataset=dataset, train_test=train_test, intent_model_path=intent_model_path)
      # train_predict(dataset=dataset, train_test=train_test)


if __name__ == '__main__':
    main()
