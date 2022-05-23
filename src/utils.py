import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

BASE_PATH = '/N/slate/amanagar/interaction/h5'

'''
Stack positive and negative samples
'''
def combine_features(feat):
    # sequence -> [1 neg, 1 pos, 2 neg, 2 pos]
    x1 = np.vstack(feat[:2])
    x2 = np.vstack(feat[2:])
    y = np.vstack([np.zeros((len(feat[0]), 1)), np.ones((len(feat[1]), 1))])
    return x1, x2, y
    
'''
Standardize the features
'''
def normalize(x, idx=None, stats=False):

    # standardize using known mean and std, especially for val and test data
    if isinstance(stats, (list, tuple)):
        return (x - stats[0]) / stats[1]

    # idx is meant to be the training set indexes
    m = x[idx].mean(axis=0)
    s = x[idx].std(axis=0)
    
    if stats:
        return (x - m) / s, m.tolist(), s.tolist()

    return (x - m) / s

'''
Calculates accuracy, FPR, TPR, auroc, and auprc of the predictions
'''
def get_stats(y, y_hat):

    # threshold the predictions and calculate the accuracy
    acc = np.mean((y_hat>0.5)==y)

    # get area under the curve (auc) score
    fpr, tpr, _ = roc_curve(y, y_hat)
    auroc = roc_auc_score(y, y_hat)

    precision, recall, _ = precision_recall_curve(y, y_hat)
    auprc = auc(recall, precision)

    stats = {'acc': acc, 'auroc': auroc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auprc': auprc}
    print('****** AUROC: {:.3f} - AUPRC: {:.3f} ******'.format(stats['auroc'], stats['auprc']))

    return stats

'''
Combine negative and positive samples into one.
'''
def get_features(tissue, type):
    seq, read, dist = [], [], []
    hf = h5py.File('{}/{}/features_{}.h5'.format(BASE_PATH, type, tissue), 'r')
    for i in [1, 2]:
        for j in ['neg', 'pos']:
            seq.append(np.array(hf[f'seq{i}_{j}']))
            read.append(np.array(hf[f'read{i}_{j}']))
            dist.append(np.array(hf[f'dist{i}_{j}']))
    hf.close()

    x1_seq, x2_seq, y = combine_features(seq)
    print('x1-seq, x2-seq, and y shape:', x1_seq.shape, x2_seq.shape, y.shape)

    x1_read, x2_read, _ = combine_features(read)
    print('x1-read and x2-read shape:', x1_read.shape, x2_read.shape)

    x1_dist, x2_dist, _ = combine_features(dist)
    print('x1-dist and x2-dist shape:', x1_dist.shape, x2_dist.shape)

    return x1_seq, x2_seq, x1_read, x2_read, x1_dist, x2_dist, y

'''
Splits the index into train, val, and test set.
Train : val : test = 0.7 : 0.15 : 0.15
'''
def get_split(index, N):
    cut = [int(0.7*N), int(0.85*N)]
    index = np.array(index)

    train_idx = index[list(range(0, cut[0]))]
    val_idx = index[list(range(cut[0], cut[1]))]
    test_idx = index[list(range(cut[1], N))]

    return train_idx, val_idx, test_idx