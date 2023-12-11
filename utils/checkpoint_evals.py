import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import json
import glob as glob
import argparse

from huggingface import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer

from epoch_eval import main as epoch_eval

CHECKPOINT_BASE_PATH = './fine_tuned_checkpoints'


# ==============================================Defining Functions=================================================
def load_pretrained_model(epoch_checkpoint_path):
    # we need these for the decoder head, if using
    use_head = False
    n_classes = 1
    backbone_cfg = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen', 
                                    'hyenadna-small-32k-seqlen', 
                                    'hyenadna-medium-160k-seqlen', 
                                    'hyenadna-medium-450k-seqlen', 
                                    'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )


    # Instantiate the model (make sure it's the same architecture as the trained model)
    model = HyenaDNAPreTrainedModel.from_pretrained(
        './checkpoints',
        pretrained_model_name,
        download=True,
        config=backbone_cfg,
        device=device,
        use_head=use_head,
        n_classes=n_classes,
    )

    # Load the trained parameters
    # model_load_path = "./trained_model.pth"  # Path where the trained model parameters are saved
    model_load_path = epoch_checkpoint_path  # Path where the trained model parameters are saved
    model.load_state_dict(torch.load(model_load_path, map_location=device), strict=False)

    # Set the model to evaluation mode
    model.eval()

    # Now you can use the model for inference
    return model

def load_tokenizer(max_seq_len):
    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_seq_len, 
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )
    return tokenizer

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

# Function to generate embeddings/run inference
def run_inference(model, tokenizer, sequence):
        # sequence = sequences_list[i]  # Get the current sequence
        tok_seq = tokenizer(sequence)
        tok_seq = tok_seq["input_ids"]  # grab ids
        
        # place on device, convert to tensor
        tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
        tok_seq = tok_seq.to(device)
        # print('tok_seq shape:', tok_seq.shape)

        # prep model and forward
        model.to(device)
        model.eval()
        
        with torch.inference_mode():
            embeddings_single = model(tok_seq)

        # print('embeddings_single shape:', embeddings_single.shape)  # embeddings here!
        # print('Type of embeddings single:', type(embeddings_single))

        return embeddings_single


def generate_embeddings(model, tokenizer, sequences_list):
    embeddings = None
    
    for i in range(len(sequences_list)):
        # print('Iteration:', i)
        sequence = sequences_list[i]  # Get the current sequence
        
        embeddings_single = run_inference(model, tokenizer, sequence)
        
        # Now I need to stack them, if I am on the first item, I need to just set embeddings to embeddings_single
        if embeddings is None:
            embeddings = embeddings_single
            # print('initial embeddings shape:', embeddings.shape)
            # print('initial type of embeddings:', type(embeddings))
        else:
            # Stack the embeddings along the batch dimension
            embeddings = torch.cat((embeddings, embeddings_single), dim=0)
            # print('embeddings shape:', embeddings.shape)
            # print('Type of embeddings:', type(embeddings))
    
    return embeddings

def save_embeddings(embeddings, embeddings_path):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(embeddings_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    print('Saving embeddings to:', embeddings_path)
    torch.save(embeddings, embeddings_path)
    # torch.save(embeddings, 'shalem_embeddings_fine_tuned.pth')

def load_sequences(ds_path):
    print('Dataset path:', ds_path)
    sequence_ds = smart_read_csv(ds_path)
    assert sequence_ds.shape[1] == 2, f'Sequence dataset should have two columns, not {sequence_ds.shape[1]} for {ds_path}'
    sequence_list = sequence_ds.iloc[:, 0].tolist()
    print('Max sequence length:', len(max(sequence_list, key=len)))

    # Assert the the sequence list contains strings
    assert all(isinstance(sequence, str) for sequence in sequence_list), f'Sequences must be strings, not {type(sequence_list)}, for {ds_path}. Seqs should be in first column with no headers'
    return sequence_list

def glob_checkpoint_path(ckpt_path_list=CHECKPOINT_BASE_PATH):
    print(f'Globing Checkpoint Path at {ckpt_path_list}')
    checkpoint_path_list = glob.glob(f'{ckpt_path_list}/model_*.pth')
    checkpoint_path_list.sort()
    print(f'Globing Checkpoint Path at {ckpt_path_list} and found {len(checkpoint_path_list)} checkpoints')

    return checkpoint_path_list

def generate_embeddings_path_name(ckpt_path, embeddings_path):
    print('ckpt_path:', ckpt_path)
    epoch = int(ckpt_path.split('_')[-1].split('.')[0])
    # Check if './fine_tuned_checkpoints_embeddings' exists, if not, create it
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)
    embedding_path_name = f'{embeddings_path}model_{epoch}_embeddings.pth'
    return embedding_path_name


def generate_val_embeddings_path_name(ckpt_path, val_embeddings_path, epoch):
    '''This is for generating the path name for the validation embeddings, 
    when running the k-fold cross validation for the shalem_15 embeddings. 
    '''
    
    print('**generate_val_embeddings_path_name**: ckpt_path:', ckpt_path)
    print('**generate_val_embeddings_path_name**: epoch:', epoch)
    print('**generate_val_embeddings_path_name**: val_embeddings_path:', val_embeddings_path)

    embeds_epoch = int(ckpt_path.split('/')[-1].split('_')[1].split('.')[0])
    assert embeds_epoch == int(epoch), f'Checkpoint epoch {embeds_epoch} does not match the epoch {epoch} in the path {ckpt_path}'
    
    # Check if './fine_tuned_checkpoints_embeddings' exists, if not, create it
    if not os.path.exists(val_embeddings_path):
        os.makedirs(val_embeddings_path)
    embedding_path_name = f'{val_embeddings_path}/model_{epoch}_embeddings.pth'
    return embedding_path_name   

def perform_dimensionality_reduction(ckpt_embedding_path):
    X = torch.load(ckpt_embedding_path)
    print('X shape:', X.shape, type(X))
    if X.is_cuda:
        X = X.cpu()
    X_reduced = torch.mean(X, dim=2).numpy()
    print('X_reduced shape:', X_reduced.shape, type(X_reduced))
    return X_reduced

def train_and_evaluate_xgboost(X, y):
    
    hyperparams = {
        'reg_lambda': 1.8,
        'reg_alpha': 0.89,
        'learning_rate': 0.22,
        'max_depth': 8,
    }
        
    xgb_regressor = xgb.XGBRegressor(**hyperparams)
    kf = KFold(n_splits=10, shuffle=True, random_state=21)  # Adjust number of splits as needed
    cv_dict = cross_validate(xgb_regressor, X, y, cv=kf, scoring=['neg_mean_squared_error', 'r2'], return_train_score=True)
    
    return cv_dict
def run_datasets_hyena(ds_path, embeddings_path, override):
    # Get the list of existing embedding files and extract their epoch numbers

    existing_embedding_files = glob.glob(f'{embeddings_path}/model_*_embeddings.pth')
    print(f'Found {len(existing_embedding_files)} existing embeddings in {embeddings_path}. If you want to override and generate new embeddings, set override to True.')
        
    existing_epochs = [int(file.split('_')[-2]) for file in existing_embedding_files]
    
    # Get the list of checkpoint files and extract their epoch numbers
    checkpoint_path_list = glob_checkpoint_path()
    checkpoint_epochs = [int(file.split('_')[-1].split('.')[0]) for file in checkpoint_path_list]
    
    # Find the missing epochs
    missing_epochs = set(checkpoint_epochs) - set(existing_epochs) if not override else checkpoint_epochs
    
    if not missing_epochs and not override:
        print('All embeddings have been generated.')
        return
    
    print(f'Epochs to process: {missing_epochs}')
    
    # Filter out the checkpoints that correspond to the epochs to be processed
    checkpoint_path_list = [file for file in checkpoint_path_list if int(file.split('_')[-1].split('.')[0]) in missing_epochs]
    
    for ckpt_path in checkpoint_path_list:
        print('Current checkpoint:', ckpt_path)
        single_ckpt_embeddings_filename = generate_embeddings_path_name(ckpt_path, embeddings_path)
        if os.path.exists(single_ckpt_embeddings_filename):
            print(f'Embedding file {single_ckpt_embeddings_filename} already exists. Skipping...')
            continue
        
        # Load the model and tokenizer 
        model = load_pretrained_model(ckpt_path)
        tokenizer = load_tokenizer(max_seq_len=150)

        sequence_list = load_sequences(ds_path)
        embeddings = generate_embeddings(model, tokenizer, sequence_list)
        print('Embeddings Generated')
        save_embeddings(embeddings, embeddings_path=single_ckpt_embeddings_filename)
        
        
#=============================Now lets Load the embeddings, perform dimensionality reduction, train and evaluate XGBoost for all newly generated embeddings=============================

def glob_ckpt_embeddings(ckpt_path_list):
    embeddings_ckpt_path_list = glob.glob(f'{ckpt_path_list}/model_*.pth')
    embeddings_ckpt_path_list.sort()
    return embeddings_ckpt_path_list

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

def get_epoch_cv_dict(embeddings_ckpt_path_list):
    # Split the path by underscores and extract the second-to-last element (epoch number)
    parts = embeddings_ckpt_path_list.split('_')
    if len(parts) >= 2:
        epoch_str = parts[-2]
        try:
            epoch_number = int(epoch_str)
            print(f"Epoch Number: {epoch_number}")
        except ValueError:
            print("Epoch number is not a valid integer.")
    else:
        print("Invalid path format.")
    return epoch_str

def create_json_path(epoch_number, cv_results_path):
    # Create the directory ./fine_tuned_checkpoints_embeddings/cv_results/ if it does not exist
    if not os.path.exists(cv_results_path):
        os.makedirs(cv_results_path)
    return os.path.join(cv_results_path, f'cv_results_epoch_{epoch_number}.json')

# Custom serializer for NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def process_val_predictions(model, single_ckpt_embedding_path, epoch_number, val_datasets_paths, val_dataset_embeddings_paths):

    for val_ds_path, val_embeds_path in zip(val_datasets_paths, val_dataset_embeddings_paths):
        print('Validation Dataset:', val_ds_path)

        cv_results_path = os.path.join(val_embeds_path, 'cv_results')
        
        # LETS GLOB THE FILES AND CHECK IF THEY ALREADY EXIST
        existing_cv_files = glob.glob(f'{cv_results_path}/*results.csv')
        if len(existing_cv_files) > 100:
            print(f'Found {len(existing_cv_files)} existing CV results in {cv_results_path}. Skipping...')
            
        else:
        
            os.makedirs(cv_results_path, exist_ok=True)
            print('CV Results Path:', cv_results_path)
            val_embeds_path = val_embeds_path
            assert os.path.exists(val_embeds_path), f'Embeddings path {val_embeds_path} does not exist'
            print('Val Embeds Path:', val_embeds_path)
            val_name = os.path.basename(val_ds_path).split('.')[0]  # Extracting the dataset name from the file path
                
            xgb_regressor = model
            
            print(f'Val embeds path being fed into generate_val_embeddings_path_name: {val_embeds_path}')
            val_embeddings_path = generate_val_embeddings_path_name(single_ckpt_embedding_path, val_embeds_path, epoch_number)
            print(f'Val Embeddings Path: {val_embeddings_path}')
            X_val_reduced = perform_dimensionality_reduction(ckpt_embedding_path=val_embeddings_path)
            print('X_val_reduced shape:', X_val_reduced.shape, type(X_val_reduced))
            print(f'Val Dataset: {val_ds_path}')
            # y_val = pd.read_csv(val_ds_path, sep=',').iloc[:, 1].to_numpy().ravel() # Replace with pd.read_csv() if this doesn't work
            y_val = smart_read_csv(val_ds_path).iloc[:, 1].to_numpy().ravel() # Replace with pd.read_csv() if this doesn't work

            print('y_val shape:', y_val.shape, type(y_val))
            y_val_predictions = xgb_regressor.predict(X_val_reduced)
            print('y_val_predictions shape:', y_val_predictions.shape, type(y_val_predictions))
            
            df = pd.DataFrame({
                'y_val': y_val,
                'y_val_predictions': y_val_predictions
            })
            
            # Create scatterplot of y_val vs predictions
            # plt.scatter(y_val, y_val_predictions)
            plt.figure()
            plt.scatter(df['y_val'], df['y_val_predictions'])
            # Add labels and title
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Predictions vs. True Values for Dataset: {val_name}, Epoch {epoch_number}')
            # Save the plot
            fig_path_name = os.path.join(cv_results_path, f"val_{val_name}_epoch_{epoch_number}_predictions_vs_true_values.png")
            print(f"Saving plot to {fig_path_name}")
            plt.savefig(fig_path_name)
            
            
            predictions = xgb_regressor.predict(X_val_reduced)
            mse = mean_squared_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            
            print(f"Epoch {epoch_number}, Validation Dataset: {val_name}, MSE: {mse}, R2: {r2}")

            results_df = pd.DataFrame({
                "Epoch": [epoch_number],
                "Validation Dataset": [val_name],
                "MSE": [mse],
                "R2": [r2]
            })
            
            save_csv_results_path = os.path.join(cv_results_path, f"{val_name}_epoch_{epoch_number}_results.csv")
            print(f"Saving results to {save_csv_results_path}")
            # Save the results to a CSV file
            results_df.to_csv(save_csv_results_path, index=False)       

def evaluate_ckpt_embeddings_with_validation(ckpt_path_list, 
                                             train_variable_path,
                                             val_datasets_paths,
                                             val_dataset_embeddings_paths,
                                             validation=True):
    
    cv_results_path = './fine_tuned_checkpoints/embeddings/cv_results/'
    
    # Glob the cv_results_path and check if the files already exist
    existing_json_files = glob.glob(f'{cv_results_path}/*.json')

    # This function evaluates the XGBoost models trained on the train/test datasets
    # on the validation datasets for every epoch and saves results to a CSV
    print('Evaluating XGBoost models...')
    print(f'Target Path: {train_variable_path}')
    embeddings_ckpt_path_list = glob_ckpt_embeddings(ckpt_path_list)
    assert embeddings_ckpt_path_list, f'No embeddings found in {ckpt_path_list}'
    
    for single_ckpt_embedding_path in embeddings_ckpt_path_list:
        X_train_reduced = perform_dimensionality_reduction(ckpt_embedding_path=single_ckpt_embedding_path)
        print(f'Train variable path: {train_variable_path}')
        y_train = pd.read_csv(train_variable_path, sep=',').iloc[:, 1].to_numpy().ravel()

        
        if validation:
            
            hyperparams = {
            'reg_lambda': 1.8,
            'reg_alpha': 0.89,
            'learning_rate': 0.22,
            'max_depth': 8,
                }
            xgb_regressor = xgb.XGBRegressor(**hyperparams)
            print('Training XGBoost model...')
            print('X_train_reduced shape:', X_train_reduced.shape, type(X_train_reduced))
            print('y_train shape:', y_train.shape, type(y_train))
            xgb_regressor.fit(X_train_reduced, y_train)
            print('XGBoost model trained.')
            
            epoch_number = get_epoch_cv_dict(embeddings_ckpt_path_list=single_ckpt_embedding_path)
            print(f'Epoch Number: {epoch_number}')
            process_val_predictions(xgb_regressor, single_ckpt_embedding_path, epoch_number, val_datasets_paths, val_dataset_embeddings_paths)
        
        # Train and evaluate XGBoost
        cv_dict = train_and_evaluate_xgboost(X_train_reduced, y_train)
        print('cv_dict:', cv_dict, type(cv_dict))
        epoch_number = get_epoch_cv_dict(embeddings_ckpt_path_list=single_ckpt_embedding_path)
        # Save dictionary to a JSON file
        cv_file_name = create_json_path(epoch_number, cv_results_path) # I COULD CHANGE THIS TO SAVE TO /TRAIN_VAL_TEST_EPOCH_EVAL/TRAIN (DEPENDING ON THE PATH INPUT TO CHECKPOINT_EMBEDDINGS AND TRAINING VARIABLE), BUT ITS NOT A BIG DEAL
        print(f"Saving results to {cv_file_name}")

        # Dump to JSON using the custom serializer
        with open(cv_file_name, 'w') as file:
            json.dump(cv_dict, file, cls=NumpyEncoder)

def chop_and_pad_sequence(seq, max_length=150):
    """Chop or pad a single sequence to the specified max_length."""
    if len(seq) > max_length:
        return seq[:max_length]
    elif len(seq) < max_length:
        return seq + 'N' * (max_length - len(seq))
    return seq

def process_dataframe(df, max_length=150):
    """Process the sequences in a dataframe without affecting other columns."""
    df_copy = df.copy()  # Ensure we don't modify the original DataFrame
    target_column = df_copy.iloc[:, 1]
    df_copy.iloc[:, 0] = df_copy.iloc[:, 0].apply(lambda seq: chop_and_pad_sequence(seq, max_length))
    print(f"Processed data shape: {df_copy.shape}")
    return df_copy

def process_and_save_dataset(ds_path, save_path, max_length=150):
    """Load a dataset, process it, and save the result."""
    df = pd.read_csv(ds_path, sep=',', header=None)
    print(f"Original data shape: {df.shape}")
    processed_df = process_dataframe(df, max_length)
    processed_df.to_csv(save_path, sep='\t', header=None, index=None)
    print(f"Processed data saved to {save_path}")
def process_multiple_datasets(ds_path_list, save_path_list, max_length=150):
    """Process multiple datasets in one go."""
    for ds_path, save_path in zip(ds_path_list, save_path_list):
        process_and_save_dataset(ds_path, save_path, max_length)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the data paths for datasets and embeddings.')
    parser.add_argument('--checkpoint_base_path', type=str, default='./fine_tuned_checkpoints', help='Base path for checkpoints from [1] fine-tuning')
    parser.add_argument('--ds_1_target_path', type=str, default='./data/train_val_test_splits/train.csv', help='Dataset 1 target path')
    parser.add_argument('--ds_2_val_target_path', type=str, default='./data/curran_15_processed.txt', help='Dataset 2 validation target path')
    parser.add_argument('--ds_3_val_target_path', type=str, default='./data/shalem_13_processed.txt', help='Dataset 3 validation target path')
    parser.add_argument('--ds_4_val_target_path', type=str, default='./data/train_val_test_splits/val.csv', help='Dataset 4 validation target path')
    parser.add_argument('--epoch_eval_val_ds_embeds_base_path', type=str, default='./epoch_eval_val_ds_embeds/', help='Base path for validation datasets embeddings')

    args = parser.parse_args()

    CHECKPOINT_BASE_PATH = args.checkpoint_base_path
    EPOCH_EVAL_VAL_DS_EMBEDS_BASE_PATH = args.epoch_eval_val_ds_embeds_base_path

    # Assuming the function 'run_datasets_hyena' and 'evaluate_ckpt_embeddings_with_validation' are defined elsewhere
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths setup
    ds_1_embeddings_path = f'{CHECKPOINT_BASE_PATH}/embeddings/'
    ds_2_embeddings_path = f'{EPOCH_EVAL_VAL_DS_EMBEDS_BASE_PATH}/{os.path.basename(args.ds_2_val_target_path).split(".")[0]}/'
    ds_3_embeddings_path = f'{EPOCH_EVAL_VAL_DS_EMBEDS_BASE_PATH}/{os.path.basename(args.ds_3_val_target_path).split(".")[0]}/'
    ds_4_embeddings_path = f'{EPOCH_EVAL_VAL_DS_EMBEDS_BASE_PATH}/{os.path.basename(args.ds_4_val_target_path).split(".")[0]}/'

    run_datasets_hyena(args.ds_1_target_path, ds_1_embeddings_path, override=True)
    run_datasets_hyena(args.ds_2_val_target_path, ds_2_embeddings_path, override=True)
    run_datasets_hyena(args.ds_3_val_target_path, ds_3_embeddings_path, override=True)
    run_datasets_hyena(args.ds_4_val_target_path, ds_4_embeddings_path, override=True)
    print('Finished running datasets.')

    val_datasets_paths = [args.ds_2_val_target_path, args.ds_3_val_target_path, args.ds_4_val_target_path]
    val_dataset_embeddings_paths = [ds_2_embeddings_path, ds_3_embeddings_path, ds_4_embeddings_path]

    print('Beginning evaluation...')
    evaluate_ckpt_embeddings_with_validation(ds_1_embeddings_path,
                                             train_variable_path=args.ds_1_target_path,
                                             val_datasets_paths=val_datasets_paths,
                                             val_dataset_embeddings_paths=val_dataset_embeddings_paths,
                                             validation=True)

    print('Finished evaluation. Beginning epoch_eval.py script...')    
    # NEED TO JOIN AND EDIT VARIOUS PATHS TO GET THIS TO WORK
    epoch_eval(os.path.join(CHECKPOINT_BASE_PATH, 'embeddings/cv_results/'), EPOCH_EVAL_VAL_DS_EMBEDS_BASE_PATH, val_ds_basename_list = [os.path.basename(path).split('.')[0] for path in val_datasets_paths])