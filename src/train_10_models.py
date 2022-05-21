import json
import argparse
import numpy as np
from utils import *
from models.ChINN import ChINN
from models.DeepMILO import DeepMILO
from models.DeepPHiC import DeepPHiC
from models.DeepTACT import DeepTACT
from models.SPEID import SPEID

np.random.seed(0)

def get_model(name, dropout=0.5):
    if name == 'ChINN':
        return ChINN(dropout=dropout)
    elif name == 'DeepMILO':
        return DeepMILO(dropout=dropout)
    elif name == 'DeepPHiC':
        return DeepPHiC(dropout=dropout)
    elif name == 'DeepTACT':
        return DeepTACT(dropout=dropout)
    elif name == 'SPEID':
        return SPEID(dropout=dropout)
    else:
        return None

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
            print(f'training {args.model}...')
            np.random.seed(iter)
            model = get_model(args.model)
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
        RESULT_FILE = '../res/{}_10fold_{}_{}.json'.format(
            args.model, tissue, args.type
        )
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(aucs, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    TYPES = ['po', 'pp']    
    MODELS = ['ChINN', 'DeepMILO', 'DeepPHiC', 'DeepTACT', 'SPEID']

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument(
        '--model', default='DeepPHiC', type=str, choices=MODELS, 
        help='model for training'
    )
    parser.add_argument(
        '--type', default='po', type=str, choices=TYPES, help='feature type'
    )
    parser.add_argument(
        '--epochs', default=200, type=int, help='maximum training epochs'
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