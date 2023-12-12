import os

import pandas as pd
import numpy as np
import torch

# from ..utils.standard_ds_loader import StandardDatasetLoader
import hyenadaterminator
from hyendaterminator import StandardDatasetLoader

standard_ds_loader = StandardDatasetLoader()

def test_shalem_15_loader():
    fine_tuning_dataset = standard_ds_loader.load_shalem_15()
    
    assert os.path.isfile(standard_ds_loader.shalem_15_processed_ds_path), f'{standard_ds_loader.shalem_15_processed_ds_path} does not exist'
    assert fine_tuning_dataset.shape == (14171, 2), f'{fine_tuning_dataset.shape.shape} does not match (14171, 2), the standard expected shape'
    
def test_n50c_loader():
    fine_tuning_dataset = standard_ds_loader.load_n50c()
    
    assert os.path.isfile(standard_ds_loader.n50c_processed_ds_path), f'{standard_ds_loader.n50c_processed_ds_path} does not exist'
    assert fine_tuning_dataset.shape == (590024, 2), f'{fine_tuning_dataset.shape} does not match (590024, 2), the standard expected shape'
    
def test_shalem_15_n50c_loader():
    val_df, fine_tuning_dataset = standard_ds_loader.remove_similar_seqs()
    
    assert os.path.isfile(standard_ds_loader.shalem_n50c_transformed_ds_path), f'{standard_ds_loader.shalem_n50c_transformed_ds_path} does not exist'
    assert fine_tuning_dataset.shape == (596884, 3), f'{fine_tuning_dataset.shape} does not match (596884, 3), the standard expected shape'
    