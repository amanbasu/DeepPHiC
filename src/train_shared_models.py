import argparse
from utils import *
import numpy as np
from models.ChINN import ChINN
from models.DeepMILO import DeepMILO
from models.DeepPHiC import DeepPHiC
from models.DeepTACT import DeepTACT
from models.SPEID import SPEID

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
    # Leave the tissue that you want to train the model for, otherwise, training
    # data might leak into the testing data during fine-tuning. 
    for tissue_of_concern in tissues:

        print('training model for', tissue_of_concern)

        features_train = {
            'x1_seq': [], 'x2_seq': [],
            'x1_read': [], 'x2_read': [],
            'x1_dist': [], 'x2_dist': []
        }
        features_val = {
            'x1_seq': [], 'x2_seq': [],
            'x1_read': [], 'x2_read': [],
            'x1_dist': [], 'x2_dist': []
        }
        label_train, label_val = [], []

        tissue_to_train = tissues[:]
        tissue_to_train.remove(tissue_of_concern)

        for tissue in tissue_to_train:

            ########## load files ##########
            print(tissue, 'loading files...')
            x1_seq, x2_seq, x1_read, x2_read, x1_dist, x2_dist, y = get_features(
                tissue, args.type
            )

            ########## split into train and val ##########
            N = len(x1_seq)        
            index = list(range(N))
            np.random.seed(0)
            np.random.shuffle(index)
            train_idx, val_idx, test_idx = get_split(index, N)

            # merge train and test data for shared model
            # effective split - train : val = 0.85 : 0.15
            train_idx = np.hstack([train_idx, test_idx])                            

            ########## normalize data ##########
            x1_read = normalize(x1_read, train_idx)
            x2_read = normalize(x2_read, train_idx)

            x1_dist = normalize(x1_dist, train_idx)
            x2_dist = normalize(x2_dist, train_idx)

            ########## bring the data together ##########
            for dict_, split_, label_ in zip(
                [features_train, features_val],
                [train_idx, val_idx],
                [label_train, label_val]
            ):

                dict_['x1_seq'].append(x1_seq[split_])
                dict_['x2_seq'].append(x2_seq[split_])
                dict_['x1_read'].append(x1_read[split_])
                dict_['x2_read'].append(x2_read[split_])
                dict_['x1_dist'].append(x1_dist[split_])
                dict_['x2_dist'].append(x2_dist[split_])
                label_.append(y[split_])

        ########## combine tissue features ##########
        for dict_ in [features_train, features_val]:
            for key in dict_.keys():
                dict_[key] = np.vstack(dict_[key])

        label_train = np.vstack(label_train)
        label_val = np.vstack(label_val)

        ########## train shared model ##########
        print(f'training shared {args.model}...')
        model = get_model(name=args.model, dropout=0.5)                         # a higher dropout to prevent overfitting
        model.fit(
            features_train['x1_seq'], features_train['x2_seq'], 
            features_train['x1_read'] , features_train['x2_read'],
            features_train['x1_dist'] , features_train['x2_dist'], label_train,
            validation_data=(
                [features_val['x1_seq'], features_val['x2_seq'], 
                features_val['x1_read'], features_val['x2_read'],
                features_val['x1_dist'], features_val['x2_dist']], label_val
            ),
            epochs=args.epochs
        )
        model.save_model('../res/shared_model/{}_{}_{}.h5'.format(
            args.model, tissue_of_concern, args.type
        ))

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