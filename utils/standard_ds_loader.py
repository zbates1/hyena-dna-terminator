# This will be using several function defined within 'utils' and eventually the ipynb file for mmseqs in the root dir
# It will load the three common datasets I have used for training:
# 1. Standard Shalem 15
# 2. MMseqs shalem_15
# 3. shalem_15 + n50c with cutoff, transformation, mmseqs

import os

import pandas as pd
import numpy as np
import torch

from .seq_processer import process_and_save_data
from .n50c_ds_loader import generate_n50c_ds
from .transform_datasets import generate_transformed_data
from .remove_similar_seqs import generate_mmseqs_results

class StandardDatasetLoader:
    def __init__(self):
        self.shalem_15_ds_path = './data/shalem_15.txt'
        self.shalem_15_processed_ds_path = './data/shalem_15_processed.txt'
        
        self.shalem_13_val_ds_path = './data/shalem_13.txt'
        self.shalem_13_val_processed_ds_path = './data/shalem_13_processed.txt'
        
        self.curran_15_val_ds_path = './data/curran_15.txt'
        self.curran_15_val_processed_ds_path = './data/curran_15_processed.txt'
        
        self.n50c_pre_processed_ds_path = './data/n50c_pre_processed.txt'
        self.n50c_processed_ds_path = './data/n50c_processed.txt'
        
        # This is for the shalem_15 + n50c dataset, which requires transformation because the two labels datasets are following different distributions
        self.shalem_n50c_transformed_ds_path = './data/shalem_n50c_transformed.txt'
        
        # The mmseqs is meant to manually create a held-out validation dataset, thus the two paths are different
        self.shalem_n50c_mmseqs_ds_path = './data/shalem_n50c_mmseqs.txt'
        self.shalem_n50c_mmseqs_val_ds_path = './data/shalem_n50c_mmseqs_val.txt'
        
        self.path_to_generated_embeddings = os.path.expanduser(f'./data/ds_fine_tuned_embeddings/')

        
    def smart_read_csv(self, filepath):
        """Read a CSV file with either comma or tab as delimiter."""
        with open(filepath, 'r') as file:
            snippet = file.read(1024)  # read the first 1024 bytes to determine the delimiter
            
        if '\t' in snippet and ',' not in snippet:
            delimiter = '\t'
        else:
            delimiter = ','  # default to comma if both are present or neither are present
        
        return pd.read_csv(filepath, delimiter=delimiter)
        
    def chop_pad_sequences(self, ds_path, save_path):
        
        assert os.path.isfile(ds_path), f'{ds_path} does not exist'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        process_and_save_data(ds_path, save_path)
        
    def load_shalem_15(self):
        self.chop_pad_sequences(self.shalem_15_ds_path, self.shalem_15_processed_ds_path)
        
        return self.smart_read_csv(self.shalem_15_processed_ds_path)
    
    def load_n50c(self):
        
        self.n50c_raw_ds_path = generate_n50c_ds(self.n50c_pre_processed_ds_path) # We actually don't even need the raw dataset path, but its here just in case. 
        
        self.chop_pad_sequences(self.n50c_pre_processed_ds_path, self.n50c_processed_ds_path)
        
        return self.smart_read_csv(self.n50c_processed_ds_path)
    
    def transform_shalem_n50c(self):
        
        self.OUTPUT_DIR = "./histograms/"
        self.LOW_TPM_CUTOFF = 1.0
        
        generate_transformed_data(self.shalem_15_processed_ds_path, self.n50c_processed_ds_path, self.shalem_n50c_transformed_ds_path, self.OUTPUT_DIR, self.LOW_TPM_CUTOFF)
        
        return self.smart_read_csv(self.shalem_n50c_transformed_ds_path)
    
    def remove_similar_seqs(self):
        # Sample Usage (Modify as needed)
        PREFIX = ''
        self.data_files = self.shalem_n50c_transformed_ds_path
        # LETS EDIT THE DATA FILES, WE NEED TO HAVE THE TARGETS BE TRANSFORMED, WHICH MEANS WE SHOULD JUST BE ABLE TO PASS THE './transformed_combined_ds.txt' output file from 'transform_datasets.py'
        self.grouped_ds_output_file = f'./data/mmseqs_output/{PREFIX}grouped_ds_seqs.fasta' # Used as output for the fasta file creation, and then the input for mmseqs
        self.mmseqs_output = f'./data/mmseqs_output/{PREFIX}clusterRes'
        self.mmseqs_min_seq_id = 0.9
        self.mmseqs_c = 0.5
        
        self.mmseqs_fasta_filename = f'{self.mmseqs_output}{PREFIX}_all_seqs.fasta' # This is the fasta file path specifically, which will take a modified path of mmseqs_output

        # Run the transform_shalem_n50c function to get the combined dataset for mmseqs input
        _ = self.transform_shalem_n50c()

        val_df, training_df = generate_mmseqs_results(self.data_files, self.grouped_ds_output_file, self.mmseqs_output, self.mmseqs_min_seq_id, self.mmseqs_c, self.mmseqs_fasta_filename, self.shalem_n50c_mmseqs_ds_path, self.shalem_n50c_mmseqs_val_ds_path)

        return self.shalem_n50c_mmseqs_val_ds_path, training_df

    def load_hyena_data_process_settings(self):
        # This function is for the [2] notebook. It establishes settings for the hyena data process. 
        
        self.n_folds = 10

        self.model_dict = {
    'lr': {"fit_intercept": True}, 
    'xgb': {"learning_rate": 0.22, "max_depth": 10, "reg_lambda": 1.8, "reg_alpha": 0.89, "n_jobs": -1}
}

        self.model_list = ['lr', 'xgb']

        self.PATH_TO_GENERATED_EMBEDDINGS = self.path_to_generated_embeddings # Base path for two below
        self.SEQS_EMBEDDINGS_PATH_NO_FINE_TUNING = f'{self.PATH_TO_GENERATED_EMBEDDINGS}train_1.pth' #no fine tuning, feel free to change this
        self.FINE_TUNED_SEQS_EMBEDDINGS_PATH = f'{self.PATH_TO_GENERATED_EMBEDDINGS}train_58.pth' #fine tuning here, feel free to change the path

        self.OPTIMAL_SEQS_DATA_PATH = './hyena_embeds_pe/optimal_sequences.csv'
        
        self.LABEL_DATA_PATH = './data/train_val_test_splits/train.csv'

        
        return self.n_folds, self.model_dict, self.model_list, self.SEQS_EMBEDDINGS_PATH_NO_FINE_TUNING, self.FINE_TUNED_SEQS_EMBEDDINGS_PATH, self.OPTIMAL_SEQS_DATA_PATH, self.LABEL_DATA_PATH