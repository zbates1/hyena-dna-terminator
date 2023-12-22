# Standard library imports
import os
import shutil
import subprocess
import sys
import urllib.request

# Related third party imports
import numpy as np
import pandas as pd

def smart_read_csv(filepath):
    """Read a CSV file with either comma or tab as delimiter."""
    
    with open(filepath, 'r') as file:
        snippet = file.read(1024)  # read the first 1024 bytes to determine the delimiter
        
    if '\t' in snippet and ',' not in snippet:
        delimiter = '\t'
    else:
        delimiter = ','  # default to comma if both are present or neither are present
    
    return pd.read_csv(filepath, delimiter=delimiter)

def create_grouped_fasta(df_list, output_filename):
    """Generate a fasta file based on a list of csv/tsv dataframes."""
    
    # Assert output_filename ends with .fasta
    assert output_filename.endswith('.fasta'), f'Output filename {output_filename} should end with .fasta'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, 'w') as fasta_file:
        for idx, df_path in enumerate(df_list):
            df = smart_read_csv(df_path)
            print(f'Shape of df {idx}: {df.shape}')
            assert df.shape[1] == 2, f'Each df in df_list should have 2 columns, containing sequence and tpm in that order'
            
            for index, row in df.iterrows():
                sequence = row[0]
                tpm = row[1]
                fasta_file.write(f'>seq_{idx + index}_tpm_{tpm}\n{sequence}\n')
            
def setup_mmseqs_in_cwd():
    """Setup mmseqs2 in the current working directory."""
    # Download and extract mmseqs2
    subprocess.run(['wget', 'https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz'])
    subprocess.run(['tar', 'xvfz', 'mmseqs-linux-avx2.tar.gz'])

    # Update PATH
    mmseqs_directory = os.path.join(os.getcwd(), 'mmseqs', 'bin')
    os.environ['PATH'] = f"{mmseqs_directory}:{os.environ['PATH']}"

    return os.path.join(mmseqs_directory, 'mmseqs')

def run_mmseqs(fasta_input, mmseqs_output, mmseqs_min_seq_id, mmseqs_c):
    # Check if mmseqs2 is in the PATH
    mmseqs_path = shutil.which("mmseqs")
    print(f'Happy to report that {mmseqs_path} is in your PATH')
    
    
    # If not in the PATH, set it up in the cwd
    if mmseqs_path is None:
        print("mmseqs is not found in system's PATH. Setting up in current directory...")
        mmseqs_path = setup_mmseqs_in_cwd()
    
    mmseqs_output = mmseqs_output
    mmseqs_min_seq_id = mmseqs_min_seq_id
    mmseqs_c = mmseqs_c
    
    mmseqs_cmd = f'{mmseqs_path} easy-cluster {fasta_input} {mmseqs_output} tmp --min-seq-id {mmseqs_min_seq_id} -c {mmseqs_c} --cov-mode 1'
    
    # Example bash command: /home/zbates/hyena-dna/mmseqs/bin/mmseqs easy-cluster /home/zbates/hyena-dna/data/mmseqs_output/output.fasta  ClusterRes tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1
    
    os.system(mmseqs_cmd)
    
def read_grouped_fasta(filename):
    data = []
    with open(filename, 'r') as f:
        lines = iter(f)  # Convert the file object to an iterator
        
        representative_sequence_id = None
        for line_current in lines:
            line_current = line_current.strip()
            
            # If the line starts with '>', it's either a representative sequence or a new sequence with a TPM value
            if line_current.startswith('>'):
                # If there's a next line and it also starts with '>', then current line is a representative sequence id
                line_next = next(lines, '').strip()
                if line_next.startswith('>'):
                    representative_sequence_id = line_current[1:]
                    print(f'\nFound a representative sequence id: {representative_sequence_id}')
                    continue  # Move to next iteration of the loop
                print(f'\nCurrent line and next line: {line_current}, {line_next[:10]}')
                # Otherwise, the next line is the sequence data for the current sequence id
                tpm = float(line_current.split('_')[-1])
                sequence_data = line_next
                print(f'\nFound sequence data and tpm that belong to representative sequence id: {representative_sequence_id}, at tpm: {tpm} and sequence data: {sequence_data[:10]}')
                
                # Assert sequence_data is containing only ACGTUN (dna/rna/padding characters)
                assert all(c in 'ACGTUN' for c in sequence_data), f'Sequence data {sequence_data} does not contain only ACGTUN'
                
                data.append([sequence_data, tpm, representative_sequence_id])
            else:
                print(f'\nUnexpected format encountered in line (just first 10 characters): {str(line_current)[:10]}')

    return pd.DataFrame(data, columns=['sequence', 'tpm', 'representative_sequence'])

def gather_stats(df):
    
    df_shape = df.shape
    df_columns = df.columns
    df_unique = len(df.representative_sequence.unique())
    df_counts = df.representative_sequence.value_counts()
    
    return df_shape, df_columns, df_unique, df_counts


def create_held_out_dataset(df, dir_name, category_column, num_categories_to_keep, ratio_to_keep, training_ds_path, held_out_val_ds_path):
    # Ensure the directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Calculate Value Counts for Each Category (sorted by frequency)
    category_counts = df[category_column].value_counts().nlargest(num_categories_to_keep)

    # Select the top num_categories_to_keep Categories
    top_categories = category_counts.index.tolist()

    # Initialize the held-out and remaining dataframes
    held_out_df = pd.DataFrame(columns=df.columns)
    remaining_df = pd.DataFrame(columns=df.columns)

    # Process each category to hold out the specified ratio
    for category in top_categories:
        # Filter the DataFrame for the current category
        category_df = df[df[category_column] == category]
        
        # Determine the number of data points to hold out for the current category
        hold_out_size = int(len(category_df) * ratio_to_keep)

        # Hold out the data points
        category_held_out = category_df.sample(n=hold_out_size)
        category_remaining = category_df.drop(category_held_out.index)
        
        # Append the held out and remaining data points to their respective dataframes
        held_out_df = pd.concat([held_out_df, category_held_out])
        remaining_df = pd.concat([remaining_df, category_remaining])

    # Append the data from categories not in the top to the remaining dataframe
    remaining_df = pd.concat([remaining_df, df[~df[category_column].isin(top_categories)]])

    # Save both DataFrames to the provided file paths
    held_out_df.to_csv(held_out_val_ds_path, index=False)
    remaining_df.to_csv(training_ds_path, index=False)
    
    # Print out information about the dataframes
    print(f"Held-out data saved to {held_out_val_ds_path}, with shape {held_out_df.shape}")
    print(f"Training data saved to {training_ds_path}, with shape {remaining_df.shape}")
    
    # Assertions to ensure data integrity
    assert len(held_out_df) + len(remaining_df) == len(df), \
        f'Lengths of held-out and remaining dataframes should add up to length of original dataframe, which is {len(df)}, but they add up to {len(held_out_df) + len(remaining_df)}'

    # Return the dataframes in case they need to be used in the current session
    return held_out_df, remaining_df

def generate_mmseqs_results(data_files, grouped_ds_output_file, mmseqs_output, mmseqs_min_seq_id, mmseqs_c, mmseqs_fasta_filename, training_ds_path, held_out_val_ds_path):
    
    # If data_files is a single file, wrap it in a list
    if isinstance(data_files, str):
        data_files = [data_files]
    
    create_grouped_fasta(data_files, grouped_ds_output_file)
    print(f'Created grouped fasta file: {grouped_ds_output_file}, from data files: {data_files}')
    print(f'Running mmseqs...')
    run_mmseqs(grouped_ds_output_file, mmseqs_output, mmseqs_min_seq_id, mmseqs_c)
    print(f'Finished running mmseqs2')
    
    print(f'Reading grouped fasta file: {mmseqs_output}')
    df = read_grouped_fasta(mmseqs_fasta_filename)
    print(f'Finished reading grouped fasta file: {mmseqs_output}')
    
    df_shape, df_columns, df_unique, df_counts = gather_stats(df)
    print(f'The following statistics are gathered from the dataframe: shape, columns, unique, counts')
    print(f'\nShape: {df_shape}')
    print(f'\nColumns: {df_columns}')
    print(f'\nNumber of Unique: {df_unique}')
    print(f'\nCounts: {df_counts}')
    
    
    # CREATE_HELD_OUT = True
    CATEGORY_COLUMN = 'representative_sequence'
    NUM_CATEGORIES_TO_KEEP = len(df.representative_sequence.unique()) # The number of top categories to hold out, or you could do just a constant
    RATIO_TO_KEEP = 0.1 # The ratio of top categories to keep, closer to 1 means the whole category will be held out

    
    val_df, training_df = create_held_out_dataset(df, dir_name=os.path.dirname(mmseqs_output), category_column=CATEGORY_COLUMN, num_categories_to_keep=NUM_CATEGORIES_TO_KEEP, ratio_to_keep=RATIO_TO_KEEP, training_ds_path=training_ds_path, held_out_val_ds_path=held_out_val_ds_path)
    print(f'Created held-out dataset')

    return val_df, training_df
    

if __name__ == '__main__':
    # Sample Usage (Modify as needed)
    PREFIX = ''

    # LETS EDIT THE DATA FILES, WE NEED TO HAVE THE TARGETS BE TRANSFORMED, WHICH MEANS WE SHOULD JUST BE ABLE TO PASS THE './transformed_combined_ds.txt' output file from 'transform_datasets.py'
    data_files = ['./data/transformed_combined_ds.txt']
    
    grouped_ds_output_file = f'./data/mmseqs_output/{PREFIX}grouped_ds_seqs.fasta' # Used as output for the fasta file creation, and then the input for mmseqs
    
    mmseqs_output = f'./data/mmseqs_output/{PREFIX}clusterRes'
    mmseqs_min_seq_id = 0.9
    mmseqs_c = 0.5
    
    mmseqs_fasta_filename = f'{mmseqs_output}{PREFIX}_all_seqs.fasta' # This is the fasta file path specifically, which will take a modified path of mmseqs_output
    
    val_df, training_df = generate_mmseqs_results(data_files, grouped_ds_output_file, mmseqs_output, mmseqs_min_seq_id, mmseqs_c, mmseqs_fasta_filename)