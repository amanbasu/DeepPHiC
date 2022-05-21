import numpy as np
import pandas as pd
import argparse
import h5py

BASE_PATH = '/N/slate/amanagar/interaction'
TYPE = None
hf = None

# convert to one-hot encoding, each channel represents either A, T, G, or C.
def encode_seq(x):
    encode = x.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
    encode = encode.replace('N', '0')                                           # mark N as A, they are very few in number
    return list(encode)

def save_file(key, data):
    hf.create_dataset(key, data=data)

# read sequences from txt file and convert to array
def get_sequences(feat, tissue, type='uint8'):
    filename = '{}/r/{}_{}_{}.txt'.format(BASE_PATH, feat, tissue, TYPE)
    with open(filename, 'r') as file:
        f = file.read()
    f = f.replace('\n', '')
    sequences = list(map(encode_seq, f[1:].split('>')))
    sequences = np.array(sequences).astype(type)
    save_file(feat, tissue, sequences)

def get_reads(feat, tissue, type='int32'):
    filename = '{}/r/{}_{}_{}.txt'.format(BASE_PATH, feat, tissue, TYPE)
    reads = pd.read_csv(filename, delimiter=' ')
    reads = reads.drop(labels=['Unnamed: 0'], axis=1)
    reads = reads.values.astype(type)
    save_file(feat, tissue, reads)

if __name__ == '__main__':
    TYPES = ['po', 'pp']    

    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument(
        '--type', default='po', type=str, choices=TYPES, help='feature type'
    )
    args = parser.parse_args()
    TYPE = args.type

    if TYPE == 'po':
        tissues = [
            'AD2', 'AO', 'BL1', 'CM', 'EG2', 'FT2', 'GA', 'GM', 'H1', 'HCmerge', 
            'IMR90', 'LG', 'LI11', 'LV', 'ME', 'MSC', 'NPC', 'OV2', 'PA', 'PO3', 
            'RA3', 'RV', 'SB', 'SX', 'TB', 'TH1', 'X5628FC'
        ]
    elif TYPE == 'pp':
        tissues = [
            'AD2', 'AO', 'BL1', 'CM', 'GA', 'GM', 'H1', 'HCmerge', 'IMR90', 
            'LG', 'LI11', 'LV', 'ME', 'MSC', 'NPC', 'OV2', 'PA', 'PO3', 'RV',
            'SB', 'SG1', 'SX', 'TB', 'TH1', 'X5628FC'
        ]

    for tissue in tissues:
        print(tissue)
        filename = '{}/h5/features_{}_{}.h5'.format(BASE_PATH, tissue, TYPE)
        hf = h5py.File(filename, 'w')
        for i in ['neg', 'pos']:
            for j in [1, 2]:
                get_sequences('seq{}_{}'.format(j, i), tissue, type='uint8')
                get_reads('read{}_{}'.format(j, i), tissue, type='int16')
                get_reads('dist{}_{}'.format(j, i), tissue, type='int32')
        hf.close()
    print('done')