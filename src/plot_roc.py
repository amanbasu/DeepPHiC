import json
import argparse
import matplotlib.pyplot as plt

def plot_roc(model, label, args):
    fprs, tprs, aucs = [], [], []
    for mod in model:
        with open('../results/stats/{}_{}_{}.json'.format(
            mod, args.tissue, args.type
        ), 'r') as file:
            stats = json.load(file)
            fpr, tpr, auroc = stats['fpr'], stats['tpr'], stats['auroc']
            fprs += [fpr]
            tprs += [tpr]
            aucs += [round(auroc, 2)]

    plt.figure(figsize=(9, 6))
    COLOR = ['#b10026', '#fc4e2a', '#feb24c']
    for i in range(len(model)):
        plt.plot(fprs[i], tprs[i], lw=2, label=f'{label[i]} (auc = {aucs[i]})', color=COLOR[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc="lower right")
    plt.title('ROC Curve - {} {}'.format(args.tissue, args.type), fontsize=15)
    plt.savefig('../results/plots/roc_curve_{}_{}.jpg'.format(args.tissue, args.type), dpi=250)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for training.')
    parser.add_argument(
        '--type', default='pp', type=str, choices=['pe', 'pp'],
        help='interaction type'
    )
    parser.add_argument(
        '--tissue', default='LV', type=str, choices=['AO', 'CM', 'LV', 'RV'],
        help='tissue/cell type'
    )
    args = parser.parse_args()

    model = ['DeepPHiC_base', 'DeepPHiC_multitask', 'DeepPHiC_finetune']
    label = ['DeepPHiC-Base', 'DeepPHiC-ML', 'DeepPHiC-TL']

    plot_roc(model, label, args)