# NetTCR - Sequence-based prediction of peptide-TCR interacions
NetTCR is a deep learning model used to predict TCR specificity. NetTCR uses convolutional neural networks (CNN) to predict whether a given TCR binds a specific peptide. 

## Data


## Network training

NetTCR model can be trained using `train_nettcr_cdr3.py` and `train_nettcr_cdr123.py`. In the first case, only CDR3 sequencesare be used; in the latter, all the CDRs are input to the network. 
The TCR chains to use can be selectd with the option `--chain`.

`python train_nettcr_cdr123.py --train_file test/train_data --val_data test/val_data --outdir test/ --chain ab`

This will generate and save a `.pt` file with the the traiend model. The model will be saved  `nettcr_prediction_cdr123_<chain>.csv`. The directory has to be specified with the option `--outdir`.

The inputs to the script are the training dataset and the validation data, used for early stopping.


## Network testing 

Both training and test set should be a comma-separated CSV files. The files should have the following columns (with headers): CDR3a, CDR3b, peptide, binder (the binder coulmn is not required in the test file). 
See test/sample_train.csv and test/sample_test.csv as an example.
