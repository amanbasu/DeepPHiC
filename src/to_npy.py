import numpy as np
import pandas as pd
import argparse
import h5py

BASE_PATH = '/N/slate/amanagar/interaction'
hf = None
TYPE = None

# convert to one-hot encoding, each channel represents either A, T, G, or C.
def encode_seq(x):
    encode = x.replace('A', '0').replace('C', '1').replace('G', '2').replace('T', '3')
    encode = encode.replace('N', '0')                                           # mark N as A, they are very few in number
    return list(encode)

def save_file(key, data):
    hf.create_dataset(key, data=data)

# read sequences from txt file and convert to array
def get_sequences(feat, tissue, type='uint8'):
    filename = '{}/r/{}_{}_{}.txt'.format(BASE_PATH, feat, tissue, TYPE.replace('pe', 'po'))
    with open(filename, 'r') as file:
        f = file.read()
    f = f.replace('\n', '')
    sequences = list(map(encode_seq, f[1:].split('>')))
    sequences = np.array(sequences).astype(type)
    save_file(feat.replace('_2k', ''), sequences[:, :2000])

def get_reads(feat, tissue, type='int32'):
    filename = '{}/r/{}_{}_{}.txt'.format(BASE_PATH, feat, tissue, TYPE.replace('pe', 'po'))
    reads = pd.read_csv(filename, delimiter=' ')
    reads = reads.drop(labels=['Unnamed: 0'], axis=1)
    reads = reads.values.astype(type)
    if type=='int16':
        save_file(feat.replace('anchor', 'read').replace('_2k', ''), reads)
    else: 
        save_file(feat.replace('anchor', 'dist').replace('_dist_2k', ''), reads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument(
        '--type', default='pe', type=str, choices=['pe', 'pp'], 
        help='feature type'
    )
    args = parser.parse_args()

    TYPE = args.type
    tissues = ['AO', 'CM', 'LV', 'RV']

    for tissue in tissues:
        print(tissue)
        filename = '{}/h5/{}/features_{}.h5'.format(BASE_PATH, TYPE, tissue)
        hf = h5py.File(filename, 'w')
        for i in ['neg', 'pos']:
            for j in [1, 2]:
                get_sequences('seq{}_{}_2k'.format(j, i), tissue, type='uint8')
                get_reads('anchor{}_{}_2k'.format(j, i), tissue, type='int16')
                get_reads('anchor{}_{}_dist_2k'.format(j, i), tissue, type='int32')
        hf.close()
    print('done')