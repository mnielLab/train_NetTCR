# NetTCR - Sequence-based prediction of peptide-TCR interacions
NetTCR is a deep learning model used to predict TCR specificity. NetTCR uses convolutional neural networks (CNN) to predict whether a given TCR binds a specific peptide. 

The scripts in this repo allow training and testing of models. It is possible to train/test using CDR3 only (with `train_nettcr_cdr3.py` and `test_nettcr_cdr3.py`) or all the CDRs (with `train_nettcr_cdr123.py` and `test_nettcr_cdr123.py`). It is also possible to choose, with the `--chain` option, which chains of the TCRs to use for training.

## Data
The input datasets shoud contain the CDRs and peptide sequences. For the CDR3 training/testing, at least the columns `peptide`, `A3` and/or `B3` should be present (with headers). For CDR123, the columns should be `peptide`, `A1`,`A2`,`A3`, and/or `B1`, `B2`, `B3`. All the input files shoud be comma-separated.

## Network training

The inputs files for the training scripts are the training dataset and the validation data, used for early stopping.

Example:
`python train_nettcr_cdr123.py --train_file test/train_data --val_data test/val_data --outdir test/ --chain ab`

This will generate and save a `.pt` file with the the traiend model. The model will be saved  `nettcr_prediction_cdr123_<chain>.csv`. The directory has to be specified with the option `--outdir`.

## Network testing 

