import fire
from CLI_assistant_func import folds_exist

import os
import torch
import csv
import numpy as np
import lightning as L
import data_utils.augmentations as augs
from data_utils.datamodules import AlignedMicroDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import sys
sys.path.append('..')
from alignment import alignment_utils as utils
from alignment.AlignCCA import AlignCCA
from models import Seq2SeqRNN

def data_prep(data_path = '../data/pt_decoding_data_S62.pkl', 
              fold_data_path = '../data/training_data_pooled', 
              patient_id = 'S14', 
              batch_size = 500, 
              n_folds = 20, 
              val_size = 0.1):
    
    data_filename = os.path.expanduser(data_path)
    pt_data = utils.load_pkl(data_filename)

    phoneme_index = 1 # All samples center around phoneme 1 onset.
    lab_type = 'phon'
    algn_type = 'phon_seq'
    tar_data, pre_data = utils.decoding_data_from_dict(pt_data, patient_id, phoneme_index,
                                                    lab_type=lab_type,
                                                    algn_type=algn_type)

    augmentations = [
        augs.time_shifting, augs.noise_jitter, augs.scaling,
        augs.time_warping, augs.time_masking, augs.time_shifting
    ]
    data = torch.Tensor(tar_data[0])
    labels = torch.Tensor(tar_data[1]).long().unsqueeze(1) - 1 
    align_labels = torch.Tensor(tar_data[2]).long() - 1
    
    # Keep original 3-column structure for DataModule compatibility
    # Label smoothing will be handled in the model loss function
    
    pool_data = [(torch.Tensor(p[0]), torch.Tensor(p[2]).long() - 1, torch.Tensor(p[2]).long() - 1) for p in pre_data]

    if folds_exist(fold_data_path, n_folds):
        print("✅ All folds found, reusing existing DataModule...")
        dm = AlignedMicroDataModule(
            data, align_labels, align_labels, pool_data, AlignCCA,
            batch_size=batch_size, folds=n_folds, val_size=val_size,
            augmentations=augmentations, data_path=fold_data_path
        )
    else:
        print("⚙️ Generating new folds...")
        dm = AlignedMicroDataModule(
            data, align_labels, align_labels, pool_data, AlignCCA,
            batch_size=batch_size, folds=n_folds, val_size=val_size,
            augmentations=augmentations, data_path=fold_data_path
        )
        dm.setup()
    
    torch.save(
        {"patient_id": patient_id, 
         "n_folds": n_folds, 
         "data": data,
         "labels": labels,
         "align_labels": align_labels,
         "pool_data": pool_data,
         "batch_size": batch_size,
         "val_size": val_size,
         "augmentations": augmentations}, 
         f"{fold_data_path}/{patient_id}_prep.pt")

    pass


def train(n_filters = 100,          # Number of CNN filters
          hidden_size = 128,         # RNN hidden layer size
          cnn_dropout = 0.3,         # CNN dropout rate
          rnn_dropout = 0.3,         # RNN dropout rate
          learning_rate = 0.1,      # Training learning rate (starting point, will be exp decayed)
          l2_reg = 1e-5,             # L2 regularization strength
          n_iters = 20,              # Number of training iterations (total # of training trials = n_fold * n_iters)
          fold_data_path = '../data/training_data_pooled',  # Path to the parent folder of fold_data
          patient_id = 'S14',        # Patient ID to train on
          label_smoothing = 0.1):    # Label smoothing strength (10% probability is distributed among the untrue classes)
    
    saved = torch.load(f"{fold_data_path}/{patient_id}_prep.pt")
    n_folds, data = saved["n_folds"], saved["data"]
    
    # Recreate DataModule with current fixed code
    dm = AlignedMicroDataModule(
        saved["data"], saved["labels"], saved["align_labels"], 
        saved["pool_data"], AlignCCA,
        batch_size=saved["batch_size"], folds=saved["n_folds"], 
        val_size=saved["val_size"], augmentations=saved["augmentations"], 
        data_path=fold_data_path
    )

    gclip_val = 0.5
    fs = 200
    in_channels = data.shape[-1] # 111
    num_classes = 9
    kernel_time = 50  # ms
    kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
    stride_time = 50  # ms
    stride = int(stride_time * fs / 1000)  # stride length in samples
    padding = 0
    n_enc_layers = 2
    n_dec_layers = 1
    activ = False
    model_type = 'gru'

    max_epochs = 500
    log_dir = os.path.join("..", "data", "transformer_logs")
    context_prefix = "pooled"

    # specify a folder to save all accuracy data
    acc_dir = os.path.join("..", "data", "accuracy_data_post_training")
    iter_accs = []  # store all accuracies across iterations

    for i in range(n_iters):
        print(f'##### Setting up data module for iteration {i+1} #####')
        dm.setup()  # prepare dataset (splits, etc.)
        
        fold_accs = []  # store results for this iteration’s folds

        # go through each fold
        for fold in range(n_folds):
            dm.set_fold(fold)
            in_channels = dm.get_data_shape()[-1]  # number of input features (111 channels)

            # build a fresh Seq2SeqRNN for this fold
            model = Seq2SeqRNN(in_channels, n_filters, hidden_size, num_classes, n_enc_layers,
                            n_dec_layers, kernel_size, stride, padding, cnn_dropout, rnn_dropout, model_type,
                            learning_rate, l2_reg, activation=activ, decay_iters=max_epochs, label_smoothing=label_smoothing)

            # callbacks: save best model + log learning rate
            callbacks = [
                ModelCheckpoint(monitor='val_acc', mode='max'),
                LearningRateMonitor(logging_interval='epoch'),
            ]

            # trainer controls training loop
            trainer = L.Trainer(default_root_dir=log_dir,
                                max_epochs=max_epochs,
                                gradient_clip_val=gclip_val,   # prevent exploding gradients
                                accelerator='auto',            # GPU if available
                                callbacks=callbacks,
                                logger=True,
                                enable_model_summary=False,
                                enable_progress_bar=False)

            # train and validate for all epochs
            trainer.fit(model=model, 
                        train_dataloaders=dm.train_dataloader(), 
                        val_dataloaders=dm.val_dataloader())

            # print training/validation metrics
            print(trainer.logged_metrics)

            # take the epoch with the best validation checkpoint, test it
            trainer.test(model=model, 
                        dataloaders=dm.test_dataloader(), 
                        ckpt_path='best')

            # store test accuracy for this fold (just one number)
            fold_accs.append(trainer.logged_metrics['test_acc'])
        
        # store fold accuracies for this iteration (n_fold numbers)
        iter_accs.append(fold_accs)

        # ensure save folder exists
        os.makedirs(os.path.join(acc_dir, context_prefix), exist_ok=True)

        # save partial results (up to this iteration) as CSV
        with open(os.path.join(acc_dir, f"{context_prefix}/{patient_id}_{context_prefix}_seq2seq_rnn_accs_iter{i+1}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(iter_accs)

        # print average accuracy across folds for this iteration
        print(np.mean(fold_accs))

    # after all iterations: save final results
    print(iter_accs)

    # save all accuracies into one CSV
    with open(os.path.join(acc_dir, f"{context_prefix}/{patient_id}_{context_prefix}_seq2seq_rnn_accs.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(iter_accs)

    # also save as .npy (easy reload in Python)
    np.save(os.path.join(acc_dir, f"{context_prefix}/{patient_id}_{context_prefix}_seq2seq_rnn_accs.npy"),
            np.array(iter_accs))

    pass


if __name__ == "__main__":
    # Fire will expose functions as CLI commands
    fire.Fire({
        "data_prep": data_prep,
        "train": train
    })
