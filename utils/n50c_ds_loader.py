import os

import pandas as pd
import zipfile
import requests
from io import BytesIO

def download_and_unzip(url, extract_to='.'):
    """
    Download a zip file from the given URL and unzip it in the specified directory.
    """
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as thezip:
        thezip.extractall(path=extract_to)
        return thezip.namelist()[0]  # Assuming the first file/folder in the zip is what you need

# Function Definitions
def process_n50_ds(n50c_df, save_path, constant_seqs_after_n50):
    
    CONSTANT_SEQ = 'CAAATTTTTCTTTTTTTTCTGTACAGACGCGTGTACGCATGTAACATTATACTGAAAACCTTGCTTGAGAAGGTTTTGGGACGCTCGAAGGCTTTAATTTGCGGCCG'
    assert isinstance(constant_seqs_after_n50, bool), f'constant_seqs_after_n50 should be a bool, but it is {type(constant_seqs_after_n50)}'
    
    n50c_df.iloc[:, 0] = n50c_df.iloc[:, 0].apply(lambda x: x + CONSTANT_SEQ if constant_seqs_after_n50 else CONSTANT_SEQ + x)

    def limit_seq_length(df_column):
        over_150 = sum(df_column.str.len() > 150)
        avg_len = df_column.str.len().mean()
        message = 'Constant seqs are placed after N50 library' if constant_seqs_after_n50 else 'Constant seqs are placed before N50 library'
        print(message)
        print(f"Number of sequences initially over 150: {over_150} with average length {avg_len}")
        return df_column.apply(lambda x: x[:150])

    print('Applying the limit_seq_length function...')
    n50c_df.iloc[:, 0] = limit_seq_length(n50c_df.iloc[:, 0])
    print(f'Avg Seq Length: {n50c_df.iloc[:, 0].str.len().mean()}')
    n50c_df.to_csv(save_path, index=False, header=False)
    print(f'File saved at: {save_path}!')

def generate_n50c_ds(N50C_SAVE_PATH):
    
    constant_seqs_after_n50 = True # This means the constant UTR sequence used in the N50 library will be placed after the N50 library for each sequence, as opposed to before the customized sequence.
    
    # Download and Unzip Data
    url = "https://figshare.com/ndownloader/files/30853609"
    # Coming from: https://figshare.com/articles/dataset/Source_Data_for_Savinov_et_al_2021_3_UTRs/16664143
    resulting_path = download_and_unzip(url)
    root_path, _ = os.path.split(resulting_path)
    print(f'Root path: {root_path}')

    n50c_file_path = os.path.join('.', root_path, 'N50C_data.csv')
    print(f"File path: {n50c_file_path}")
    # Now you can use n50c_file_path for your further processing

    # Load and Process Dataset
    results_df = pd.read_csv(n50c_file_path)
    print(f'Head of original data: \n{results_df.head()}')
    filtered_df = results_df[['UTR', 'log2_score']]
    print('\n\nHead of filtered data:\n', filtered_df.head())
    
    assert filtered_df.shape[1] == 2, f'Expected 2 columns, got {filtered_df.shape[1]}'
    
    avg_seq_length = filtered_df.iloc[0].str.len().mean()
    print('\nMean sequence length before processing...:', avg_seq_length)
    print(f'\nShape of filtered data: {filtered_df.shape}')

    # Process and Save Data with Constant Gene Sequence
    process_n50_ds(filtered_df, N50C_SAVE_PATH, constant_seqs_after_n50)
    
    return n50c_file_path

if __name__ == "__main__":
    
    # Example Usage
    N50C_SAVE_PATH = './data/n50c_constant_sequences.txt'
    n50c_data_path = generate_n50c_ds(N50C_SAVE_PATH)