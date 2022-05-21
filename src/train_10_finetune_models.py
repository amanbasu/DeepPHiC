import json
import argparse
import numpy as np
from utils import *
import tensorflow as tf
from models.DeepPHiC import DeepPHiC

np.random.seed(0)

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
        base_model = tf.keras.models.load_model(
            '../res/shared_model/DeepPHiC_{}_{}.h5'.format(tissue, args.type)
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

            ########## train models ##########
            print(f'training DeepPHiC...')
            np.random.seed(iter)
            model = DeepPHiC(learning_rate=5e-4)
            model.model.set_weights(base_model.get_weights()) 

            # only fine-tune the classifier
            if not args.train_full:
                for layer in model.model.layers[:-3]:
                    layer.trainable = False

            model.fit(
                x1_seq[train_idx], x2_seq[train_idx], 
                x1_read[train_idx], x2_read[train_idx],
                x1_dist[train_idx], x2_dist[train_idx], y[train_idx],
                validation_data=(
                    [x1_seq[val_idx], x2_seq[val_idx],
                    x1_read[val_idx], x2_read[val_idx],
                    x1_dist[val_idx], x2_dist[val_idx]], y[val_idx]
                ),
                epochs=args.epochs
            )
            y_hat = model.predict(
                x1_seq[test_idx], x2_seq[test_idx], 
                x1_read[test_idx], x2_read[test_idx],
                x1_dist[test_idx], x2_dist[test_idx]
            )
            stats = get_stats(y[test_idx], y_hat)
            aucs['auroc'].append(stats['auroc'])
            aucs['auprc'].append(stats['auprc'])

        ########## save results ########## 
        RESULT_FILE = '../res/DeepPHiC_10fold_finetune_{}_{}.json'.format(
            tissue, args.type
        )
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(aucs, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    TYPES = ['po', 'pp']    

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument(
        '--type', default='po', type=str, choices=TYPES, help='feature type'
    )
    parser.add_argument(
        '--epochs', default=200, type=int, help='maximum training epochs'
    )
    parser.add_argument(
        '--train_full', default=True, type=bool, 
        help='whether to finetune only the classifier or entire model'
    )
    args = parser.parse_args()

    if args.type == 'po':
        tissues = [
            'AD2', 'AO', 'BL1', 'CM', 'EG2', 'FT2', 'GA', 'GM', 'H1', 'HCmerge', 
            'IMR90', 'LG', 'LI11', 'LV', 'ME', 'MSC', 'NPC', 'OV2', 'PA', 'PO3', 
            'RA3', 'RV', 'SB', 'SX', 'TB', 'TH1', 'X5628FC'
        ]
    elif args.type == 'pp':
        tissues = [
            'AD2', 'AO', 'BL1', 'CM', 'GA', 'GM', 'H1', 'HCmerge', 'IMR90', 
            'LG', 'LI11', 'LV', 'ME', 'MSC', 'NPC', 'OV2', 'PA', 'PO3', 'RV',
            'SB', 'SG1', 'SX', 'TB', 'TH1', 'X5628FC'
        ]

    train(tissues, args)
    print('done')