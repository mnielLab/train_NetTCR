# NetTCR-2.1 - Sequence-based prediction of peptide-TCR interacions using CDR1, CDR2 and CDR3 loops 
NetTCR-2.1 is a deep learning model used to predict TCR specificity. NetTCR-2.1 uses convolutional neural networks (CNN) to predict whether a given TCR binds a specific peptide.

The scripts in this repo allow training and testing of models. It is possible to train/test using CDR3 only (with `train_nettcr_cdr3.py` and `test_nettcr_cdr3.py`) or all the CDRs (with `train_nettcr_cdr123.py` and `test_nettcr_cdr123.py`). It is also possible to choose, with the `--chain` option, which chains of the TCRs to use for training.

## Data
The input datasets shoud contain the CDRs and peptide sequences. For the CDR3 training/testing, at least the columns `peptide`, `A3`, `B3` should be present (with headers). For CDR123, the columns should be `peptide`, `A1`,`A2`,`A3`, `B1`, `B2`, `B3`. All the input files shoud be comma-separated.

See `test/train_data` as an example.

## Network training

The inputs files for the training scripts are the training dataset and the validation data, used for early stopping.

Example:

`python src/train_nettcr_cdr3.py --train_data test/train_data --val_data test/val_data --outdir test/models/ --chain ab`

This will generate and save a `.pt` file with the the traiend model. The directory has to be specified with the option `--outdir`.

The other input arguments to the script are `--epochs`, `--learning_rate`, `--verbose`. If a GPU is available, the scritp will detect it and use it. 

## Network testing 
The test scripts can be used to make predictions of test TCRs, using a pre-trained model.

Example:

`python src/test_nettcr_cdr3.py --test_data test/test_data --trained_model test/models/trained_model_cdr3_ab.pt --outdir test/models/ --chain ab`

This will generate and save a `.csv` file with the prediction. The file will be saved in the specified output directory. 
