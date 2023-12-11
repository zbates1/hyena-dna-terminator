# Useful Medium article on visualizing distribution of data
# https://towardsdatascience.com/10-examples-to-master-distribution-plots-with-python-seaborn-4ea2ceea906a

# Standard library imports
import argparse
import json
import os

# Related third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import StandardScaler, Normalizer


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
def smart_read_csv(filepath):
    """Read a CSV file with either comma or tab as delimiter."""
    
    # Try with comma first
    with open(filepath, 'r') as file:
        snippet = file.read(1024)  # read the first 1024 bytes to determine the delimiter
        
    # Determine the delimiter based on the snippet
    if '\t' in snippet and ',' not in snippet:
        delimiter = '\t'
    else:
        delimiter = ','  # default to comma if both are present or neither are present
    
    return pd.read_csv(filepath, delimiter=delimiter)

def read_data(filepath, filter, LOW_TPM_CUTOFF):
    data = smart_read_csv(filepath)
    if filter:
        data = data[data.iloc[:, 1] > LOW_TPM_CUTOFF]
    else:
        data = data
    return data

def log_transform(data):
    return np.log(data + 1)  # We add 1 to handle zeros

def scale_data(target, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        # scaler = Normalizer()
    target_numpy = target.to_numpy().reshape(-1, 1)
    scaled = scaler.fit_transform(target_numpy)
    # Convert back to series
    scaled = pd.Series(scaled.reshape(-1))
    return scaled

def plot_distributions(ds_1_target, ds_2_target, title, OUTPUT_DIR):
    # Create a DataFrame to hold the data from both datasets
    df = pd.DataFrame({
        'Value': pd.concat([ds_1_target, ds_2_target], ignore_index=True),
        'Dataset': ['ds_1'] * len(ds_1_target) + ['ds_2'] * len(ds_2_target)
    })

    # Plot histogram using matplotlib
    plt.figure(figsize=(10, 6))
    plt.hist(ds_1_target, bins=50, alpha=0.5, label='ds_1')
    plt.hist(ds_2_target, bins=50, alpha=0.5, label='ds_2')
    plt.title(f"{title} - Histogram")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}_histogram.png"))
    plt.show()

    # Plot ECDF using seaborn
    sns.displot(
        data=df,
        x="Value",
        kind="ecdf",
        hue="Dataset",
        height=6,
        aspect=1.4,
        stat="proportion"
    )
    plt.title(f"{title} - ECDF")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}_ecdf.png"))
    plt.show()

def save_combined_data(ds_1, ds_2, output_path):
    print('Len ds_1:', len(ds_1))
    print('Len ds_2:', len(ds_2))
    
    combined_data = pd.concat([ds_1, ds_2], ignore_index=True)
    print(f'Shape of combined data: {combined_data.shape}')
    combined_data.to_csv(output_path, index=False, sep='\t')
    print(f'\nCombined DF Save and the head: \n{combined_data.head()}')
    
    # Save the lengths to a JSON file
    lengths = {
        "ds_1_length": len(ds_1),
        "ds_2_length": len(ds_2)
    }
    json_path = os.path.join(os.path.dirname(output_path), "dataset_lengths.json")
    with open(json_path, 'w') as json_file:
        json.dump(lengths, json_file)

def generate_transformed_data(ds_1_path, ds_2_path, output_path, OUTPUT_DIR, LOW_TPM_CUTOFF):
    create_output_dir(os.path.dirname(output_path))
    
    ds_1_data = read_data(ds_1_path, filter=True, LOW_TPM_CUTOFF=LOW_TPM_CUTOFF) # This filter is meant to be applied to Shalem_15 because of the TPM's that are below 1.0. These duplicates are hindering the machine learning process.
    ds_2_data = read_data(ds_2_path, filter=False, LOW_TPM_CUTOFF=LOW_TPM_CUTOFF)
    print(f'Shape of ds_1_data in transform_datasets.py!: {ds_1_data.shape}')
    print(f'Shape of ds_2_data in transform_datasets.py!: {ds_2_data.shape}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_distributions(ds_1_data.iloc[:, 1], ds_2_data.iloc[:, 1], "Original", OUTPUT_DIR)

    # ds_1_log = log_transform(ds_1_data.iloc[:, 1])
    # ds_2_log = log_transform(ds_2_data.iloc[:, 1])
    # plot_distributions(ds_1_log, ds_2_log, "Log")

    # Assuming the boxcox and yeojohnson transformations follow the same pattern
    # Add these transforms if they exist in your original code

    ds_1_scaled = scale_data(ds_1_data.iloc[:, 1])
    ds_2_scaled = scale_data(ds_2_data.iloc[:, 1])
    plot_distributions(ds_1_scaled, ds_2_scaled, "Scaled", OUTPUT_DIR)

    ds_1_data.iloc[:, 1] = ds_1_scaled
    ds_2_data.iloc[:, 1] = ds_2_scaled
    
    print(f'Shape of ds_1_data: {ds_1_data.shape}')
    print(f'Shape of ds_2_data: {ds_2_data.shape}')
    
    ds_1_data.columns = ["sequence", "expression"]
    ds_2_data.columns = ["sequence", "expression"]

    save_combined_data(ds_1_data, ds_2_data, output_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process two datasets and visualize distributions.')
    parser.add_argument('--ds_1', type=str, default='./data/shalem_15.txt', help='Path to the first dataset.')
    parser.add_argument('--ds_2', type=str, default='./data/N50C_processed.txt', help='Path to the second dataset.')
    parser.add_argument('--output_path', type=str, default='./data/transformed_combined_ds.txt', help='Path to save the combined scaled dataset.')
    
    args = parser.parse_args()
    
    OUTPUT_DIR = "./histograms/"
    LOW_TPM_CUTOFF = 1.0

    generate_transformed_data(args.ds_1, args.ds_2, args.output_path, OUTPUT_DIR, LOW_TPM_CUTOFF)
