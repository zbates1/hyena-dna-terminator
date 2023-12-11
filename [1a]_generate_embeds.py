# [1a]_generate_embeds.py

import os

import torch
import pandas as pd
import numpy as np

from .utils.standard_ds_loader import StandardDatasetLoader
from huggingface import HyenaDNAPreTrainedModel
from standalone_hyenadna import CharacterTokenizer

def load_model_and_tokenizer(CHECKPOINT_EPOCH_LOAD_NUMBER):

        # Setting for loading model. These should not be changed
        pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
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
        model_load_path = f'./fine_tuned_checkpoints/model_epoch_{CHECKPOINT_EPOCH_LOAD_NUMBER}.pth'  # Path where the trained model parameters are saved
        model.load_state_dict(torch.load(model_load_path, map_location=device), strict=False)

        # Set the model to evaluation mode
        model.eval()

        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=1000, 
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )
        
        return model, tokenizer, device
    
# Function to generate embeddings/run inference
def run_inference(model, tokenizer, sequence, device):
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

def generate_embeddings(model, tokenizer, sequences_list, device, ds_path_list):
    embeddings = None
    
    for i in range(len(sequences_list)):
        # print('Iteration:', i)
        sequence = sequences_list[i]  # Get the current sequence
        # assert the length of the sequence is less than 1000
        if len(sequence) > 1000:
            print(f'Sequence is too long: {len(sequence)} in row {i} in dataset {ds_path_list}')
            # cut the sequence end to 1000
            sequence = sequence[:1000]
            continue
        
        # print('sequence:', sequence)
        embeddings_single = run_inference(model, tokenizer, sequence, device)
        # embeddings_single = embeddings_single.to('cpu') #THIS IS NEW, TRYING TO DEAL WITH THE OOM MESSAGES==================================================================

        # Now I need to stack them, if I am on the first item, I need to just set embeddings to embeddings_single
        if embeddings is None:
            embeddings = embeddings_single
            # print('initial embeddings shape:', embeddings.shape)
            # print('initial type of embeddings:', type(embeddings))
        else:
            # Stack the embeddings along the batch dimension
            embeddings = torch.cat((embeddings, embeddings_single), dim=0)
            # Push to CPU
            # embeddings = embeddings.to('cpu') #THIS IS NEW, TRYING TO DEAL WITH THE OOM MESSAGES==================================================================
            # print('embeddings shape:', embeddings.shape)
            # print('Type of embeddings:', type(embeddings))
    
    return embeddings

def load_embeddings(embeddings_path):
    embeddings = torch.load(embeddings_path)
    return embeddings

def save_embeddings(embeddings, embeddings_path, embeddings_base_name):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(embeddings_path)
    print(f'Saving embeddings to directory: {directory}')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    full_embeddings_path = os.path.join(directory, embeddings_base_name)
    print('Saving embeddings to file path:', full_embeddings_path)
    torch.save(embeddings, full_embeddings_path)
    # torch.save(embeddings, 'shalem_embeddings_fine_tuned.pth')

def load_sequences(ds_path):
    print('Dataset path:', ds_path)
    sequence_ds = standard_ds_loader.smart_read_csv(ds_path)
    sequence_list = sequence_ds.iloc[:, 0].tolist()
    # print('Sequence list:', sequence_list)
    # Assert the the sequence list contains strings
    assert all(isinstance(sequence, str) for sequence in sequence_list), f'Sequences must be strings, not {type(sequence_list)}, for {ds_path}. Seqs should be in first column with no headers'
    return sequence_list

def run_datasets_hyena(ds_path_list, embeddings_folder):
    
    """
    Run datasets hyena.

    Args:
        ds_path_list (list): A list of dataset paths.
        embeddings_folder (str): Path to the folder where the embeddings will be saved.

    Returns:
        None
    """
    
    os.makedirs(embeddings_folder, exist_ok=True)

    
    for ds_path in ds_path_list:
        print('Current dataset:', ds_path)
        sequence_list = load_sequences(ds_path)
        embeddings = generate_embeddings(model, tokenizer, sequence_list)
        print(f'Embeddings generated with shape: {embeddings.shape}')
        save_embeddings(embeddings, embeddings_path=embeddings_folder, embeddings_base_name=f'{os.path.basename(ds_path).split(".")[0]}_{CHECKPOINT_EPOCH_LOAD_NUMBER}.pth')
        
        
# For creating your own datapaths, use the following:
ds_load_path_list = [test_data_path, train_data_path, val_data_path]
PATH_TO_GENERATED_EMBEDDINGS = standard_ds_loader.path_to_generated_embeddings

run_datasets_hyena(ds_path_list=ds_load_path_list, embeddings_folder=PATH_TO_GENERATED_EMBEDDINGS)