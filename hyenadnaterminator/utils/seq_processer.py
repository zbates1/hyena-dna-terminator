# seq_processor.py
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

def chop_and_pad_sequences(dataframe):
    max_length = 150  # Set the maximum length to 150
    print('max sequence length:', max_length)
    print('Number of nans in each dataframe:')
    print(dataframe.isna().sum())
    print('Shape before dropping nans:', dataframe.shape)
    dataframe.dropna(inplace=True)
    print('Shape after dropping nans:', dataframe.shape)
    
    def process_sequence(seq):
        if len(seq) > max_length:
            seq = seq[:max_length]
        elif len(seq) < max_length:
            seq = seq + 'N' * (max_length - len(seq))
        return seq
    
    dataframe.iloc[:, 0] = dataframe.iloc[:, 0].apply(process_sequence)
    print(f'\n\nFirst rows after processing: {dataframe.iloc[:5, 0]}')
    return dataframe

def process_and_save_data(ds_path, save_path):
    df = smart_read_csv(ds_path)
    print(f'First rows to be processed from {ds_path}:, {df.iloc[:5, 0]}')
    df = chop_and_pad_sequences(df)
    df.to_csv(save_path, sep='\t', header=None, index=None)
    print(f'Data saved to {save_path}')

def main():
    # Example usage
    ds_path = 'path_to_your_dataset.csv'  # Replace with your dataset path
    save_path = 'path_to_save_processed_data.tsv'  # Replace with your save path

    process_and_save_data(ds_path, save_path)

if __name__ == "__main__":
    main()