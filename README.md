# DeepPHiC: Predicting promoter-centered chromatin interactions using a deep learning approach

We developed a supervised multi-modal deep learning model, which utilizes a comprehensive set of features including genomic sequence, epigenetic signals and anchor distance to predict tissue/cell type-specific genome-wide promoter-enhancer and promoter-promoter interactions. We further extend the deep learning model in a multi-task learning and transfer learning framework. We demonstrate that the proposed approach outperforms state-of-the-art deep learning approaches and robust to the inclusion of anchor distance. In addition, we find that the proposed approach can achieve comparable prediction performance using biologically relevant tissues/cell types compared to using all tissues/cell types especially for predicting promoter-enhancer interactions.

<p align="center"><img src="res/overview.png"/></p>

## Requirements

DeepPHiC is solely implemented on TensorFlow `2.6.0` using the Keras framework and Python `3.9.6`. The other requirements are:

```
h5py==3.1.0
matplotlib==3.4.3
numpy==1.19.3
numpy-utils==0.1.6
scikit-learn==0.24.2
tensorflow==2.6.0
```

## Usage

Download DeepPHiC

```
git clone https://github.com/amanbasu/DeepPHiC.git
```

Install requirements

```
pip install -r requirements.txt
```

You can learn more about the script arguments using the `-h` command for individual files
```
$ python train_base.py -h
usage: train_base.py [-h] [--type {pe,pp}] [--epochs EPOCHS] [--lr LR]
                     [--dropout DROPOUT] [--test TEST]

Arguments for training.

optional arguments:
  -h, --help         show this help message and exit
  --type {pe,pp}     interaction type
  --epochs EPOCHS    maximum training epochs
  --lr LR            learning rate
  --dropout DROPOUT  dropout
  --test TEST        test flag to work on sample data
```

### Train base model
### Train shared model
### Train fin-tune model
### Train multi-task model
### Plot ROC curve
