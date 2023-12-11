import json
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Utility Functions
def get_epoch_number(filename):
    """Extract the epoch number from the filename."""
    try:
        return int(filename.rstrip(".json").split("_")[-1])
    except ValueError:
        return 0


def retrieve_and_sort_json_files(checkpoint_path):
    """Retrieve and sort JSON files by their epoch number."""
    pattern = os.path.join(checkpoint_path, 'cv_results_*.json')
    json_files = glob.glob(pattern)
    return sorted(json_files, key=get_epoch_number)


def parse_json_file(json_file):
    """Parse and process JSON file to extract metrics."""
    with open(json_file) as f:
        cv_dict = json.load(f)

    metrics = {
        'avg_test_neg_mse': np.mean(cv_dict['test_neg_mean_squared_error']),
        'avg_train_neg_mse': np.mean(cv_dict['train_neg_mean_squared_error']),
        'avg_test_r2': np.mean(cv_dict['test_r2']),
        'avg_train_r2': np.mean(cv_dict['train_r2'])
    }

    return metrics


def create_and_populate_dataframe(json_files):
    """Create a DataFrame and populate it with data from JSON files."""
    df = pd.DataFrame(columns=['avg_test_neg_mse', 'avg_train_neg_mse', 'avg_test_r2', 'avg_train_r2'])
    print(f'Head of df:\n{df.head()}')
    for json_file in json_files:
        metrics = parse_json_file(json_file)
        df_row = pd.DataFrame(metrics, index=[0])
        df = pd.concat([df, df_row], ignore_index=True)

    return df


def plot_metrics(epoch_df, base_path):
    """Plot the metrics and save the figures."""
    figures_dir = os.path.join(base_path, 'figure_results')
    os.makedirs(figures_dir, exist_ok=True)

    # Create a range of x_values based on the number of epochs in the DataFrame
    x_values = range(len(epoch_df))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=x_values, y=epoch_df['avg_test_neg_mse'], label='Avg Test Negative MSE', ax=ax)
    sns.lineplot(x=x_values, y=epoch_df['avg_train_neg_mse'], label='Avg Train Negative MSE', ax=ax)
    sns.lineplot(x=x_values, y=epoch_df['avg_test_r2'], label='Avg Test R2', ax=ax)
    sns.lineplot(x=x_values, y=epoch_df['avg_train_r2'], label='Avg Train R2', ax=ax)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metrics vs. Epochs')
    ax.legend()
    metrics_plot_path = os.path.join(figures_dir, 'metrics_vs_epochs.png')
    plt.savefig(metrics_plot_path)
    plt.close(fig)  # Close the figure to avoid displaying it in non-interactive environments


# Data Loading and Processing Functions
def load_and_combine_csv_files(csv_paths):
    """Load CSV files and combine them into a single DataFrame."""      
    print(f'CSV File paths passed into load_and_combine_csv_files: {csv_paths}')  
    all_dfs = [pd.read_csv(file) for path in csv_paths for file in glob.glob(os.path.join(path, 'cv_results/*.csv'))]
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.sort_values(by='Epoch', inplace=True, ignore_index=True)
    return combined_df

def plot_val_metrics(dataframe, metric, base_path, title):
    """
    Plots the given metric from the dataframe and saves the plot to the specified path.

    Args:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - metric (str): The metric to plot ('MSE' or 'R2').
    - base_path (str): The base directory path where the figure will be saved.
    - title (str): The title of the plot.
    - ylabel (str): The label for the y-axis.
    """
    
    ylabel = metric
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dataframe, x='Epoch', y=metric, hue='Validation Dataset', marker="o")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(title='Validation Dataset')
    plt.grid(True)

    # Check if the directory exists, if not, create it
    figures_path = os.path.join(base_path, 'figure_results')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    # Save the figure
    plt.savefig(os.path.join(figures_path, f'val_{metric}_plot.png'))
    plt.close()  # Close the plot to free memory


def main(json_list_path, val_ds_embeds_base_path, val_ds_basename_list):
    # JSON_LIST_PATH = './fine_tuned_checkpoints/embeddings/cv_results/' # This is for ds_1 in checkpoint_evals.py
    # VAL_DS_EMBEDS_BASE_PATH = './epoch_eval_val_ds_embeds'
    # val_ds_basename_list = ['shalem_n50c_val', 'shalem_13_processed', 'curran_15_processed'] # Change this based on the input to checkpoint_evals.py

    JSON_LIST_PATH = json_list_path
    VAL_DS_EMBEDS_BASE_PATH = val_ds_embeds_base_path
    val_ds_basename_list = val_ds_basename_list
    print(f'val_ds_basename_list passed from checkpoint_evals.py: {val_ds_basename_list}')

    # ==============Processing JSON Files================
    print('Loading Cross Validation Results from JSON Files...')
    json_files = retrieve_and_sort_json_files(JSON_LIST_PATH)
    print(f'Found {len(json_files)} JSON files.')
    print('Processing JSON Files...')
    epoch_df = create_and_populate_dataframe(json_files)
    
    # Plot and save metrics figures
    print('Plotting Metrics...')
    plot_metrics(epoch_df, VAL_DS_EMBEDS_BASE_PATH)
    print('Finished Plotting Metrics.')
    # ====================================================
    
    
    # ==============Processing CSV (VAL) Files================
    CSV_LIST_PATHS = [os.path.join(VAL_DS_EMBEDS_BASE_PATH, basename) for basename in val_ds_basename_list]
    print(f'CSV_LIST_PATHS: {CSV_LIST_PATHS}')
    
    assert len(val_ds_basename_list) == len(CSV_LIST_PATHS), \
        'Length of val_ds_basename_list and CSV_LIST_PATHS do not match'

    # Load, combine and process CSV files
    val_df = load_and_combine_csv_files(CSV_LIST_PATHS)
    print('Shape of val_df:', val_df.shape)
    print('\nHead of val_df:\n', val_df.head())

    # Plot and save metrics figures
    plot_val_metrics(val_df, 'MSE', VAL_DS_EMBEDS_BASE_PATH, 'Epoch Eval Val Dataset, MSE')
    plot_val_metrics(val_df, 'R2', VAL_DS_EMBEDS_BASE_PATH, 'Epoch Eval Val Dataset, R2')

if __name__ == '__main__':
    main()