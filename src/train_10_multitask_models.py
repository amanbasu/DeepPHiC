import json
import argparse
from utils import *
import numpy as np
import tensorflow as tf
from models.DeepPHiC import DeepPHiCFusion

np.random.seed(0)

def get_common_features(inp1, inp2, inp3, inp4, inp5, inp6, idx, share_model):
    return share_model.predict(
        [inp1[idx], inp2[idx], inp3[idx], inp4[idx], inp5[idx], inp6[idx]]
    )

def train(tissues, args):

    for tissue in tissues:

        ########## load files ##########
        print(tissue, 'loading files...')
        x1_seq, x2_seq, x1_read, x2_read, x1_dist, x2_dist, y = get_features(
            tissue, 
            args.type
        )

        N = len(x1_seq)
        aucs = {'auroc': [], 'auprc': []}
        share_model = tf.keras.models.load_model(
            '../res/shared_model/DeepPHiC_{}_{}.h5'.format(tissue, args.type)
        )
        share_model = tf.keras.models.Model(
            inputs=share_model.input, outputs=share_model.layers[-3].output
        )

        for iter in range(10):
            print('****** fold - {} ******'.format(iter+1))

            ########## train-test-val split ##########
            # 70% train, 15% val, 15% test
            index = list(range(N))
            np.random.seed(iter)
            np.random.shuffle(index)
            train_idx, val_idx, test_idx = get_split(index, N)

            ########## data normalization ##########
            x1_read = normalize(x1_read, train_idx)
            x2_read = normalize(x2_read, train_idx)

            x1_dist = normalize(x1_dist, train_idx)
            x2_dist = normalize(x2_dist, train_idx)

            ########## common features from shared model ##########
            common_feature_train = get_common_features(
                x1_seq, x2_seq, x1_read, x2_read, x1_dist, x2_dist, train_idx,
                share_model
            )
            common_feature_val = get_common_features(
                x1_seq, x2_seq, x1_read, x2_read, x1_dist, x2_dist, val_idx,
                share_model
            )
            common_feature_test = get_common_features(
                x1_seq, x2_seq, x1_read, x2_read, x1_dist, x2_dist, test_idx, 
                share_model
            )

            ########## train models ##########
            print('training DeepPHiCFusion...')
            np.random.seed(iter)
            model = DeepPHiCFusion(learning_rate=args.lr, dropout=args.dropout)
            model.fit(
                x1_seq[train_idx], x2_seq[train_idx], 
                x1_read[train_idx], x2_read[train_idx],
                x1_dist[train_idx], x2_dist[train_idx], 
                common_feature_train, y[train_idx],
                validation_data=(
                    [x1_seq[val_idx], x2_seq[val_idx],
                    x1_read[val_idx], x2_read[val_idx],
                    x1_dist[val_idx], x2_dist[val_idx],
                    common_feature_val], y[val_idx]
                ),
                epochs=args.epochs
            )
            y_hat = model.predict(
                x1_seq[test_idx], x2_seq[test_idx], 
                x1_read[test_idx], x2_read[test_idx],
                x1_dist[test_idx], x2_dist[test_idx], common_feature_test
            )
            stats = get_stats(y[test_idx], y_hat)
            aucs['auroc'].append(stats['auroc'])
            aucs['auprc'].append(stats['auprc'])

        ########## save results ########## 
        RESULT_FILE = '../res/DeepPHiC_10fold_multitask_{}_{}.json'.format(
            tissue, args.type
        )
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(aucs, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument(
        '--type', default='pe', type=str, choices=['pe', 'pp'],
        help='interaction type'
    )
    parser.add_argument(
        '--epochs', default=200, type=int, help='maximum training epochs'
    )
    parser.add_argument(
        '--lr', default=1e-4, type=float, help='learning rate'
    )
    parser.add_argument(
        '--dropout', default=0.2, type=float, help='dropout'
    )
    args = parser.parse_args()

    with open('../res/tissues.json', 'r') as f:
        if args.type == 'pe':
            tissues = json.load(f)['pe']
        else:
            tissues = json.load(f)['pp']
            
    train(tissues, args)
    print('done')