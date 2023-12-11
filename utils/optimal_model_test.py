# optimal_model_test.py

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import pacmap
from sklearn.decomposition import PCA
import torch

# Define your model mapping here
model_mapping = {
    'RandomForest': RandomForestRegressor,
    'Ridge': Ridge,
    'XGBoost': XGBRegressor
}

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

def find_best_model_config(json_file_path, specific_model):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Sort the configurations based on the validation score
    best_config = max(data.items(), key=lambda item: item[1]['Value'])
    # If specific model is not None, use that string to filter out the models that don't match in item[0]
    if specific_model:
        best_config = [key for key in best_config if specific_model in key][0]
        
    best_config_params = best_config[1]['Params']
    
    X_train_path = best_config[0]
    
    return best_config, best_config_params, X_train_path

def process_embeddings(X_path):
    
    X = torch.load(X_path)
    
    # Check if embeddings is a PyTorch tensor, and if so, convert to NumPy array
    if isinstance(X, torch.Tensor):
        # Push to the CPU if it's on a GPU
        if X.is_cuda:
            X = X.cpu()
        
        # Convert to numpy array
        X = X.numpy()

    # If it's already a numpy array, proceed directly to reshaping
    if isinstance(X, np.ndarray):
        X_reshaped = X.reshape(len(X), -1)
        print('X shape:', X_reshaped.shape)
        
    else:
        print("The provided embeddings are neither a PyTorch tensor nor a NumPy array.")

    return X_reshaped

def min_max_scaler(X):
    # Calculate the max and min along the last dimension (128)
    # Calculate the max and min along the last dimension (128)
    max_array = torch.max(X, dim=2)[0].numpy()
    min_array = torch.min(X, dim=2)[0].numpy()

    # Stack max and min arrays along the last dimension (2)
    max_min_array = np.stack([max_array, min_array], axis=2)

    # Reshape the max and min arrays to be 2D arrays
    min_max_array_reshaped = max_min_array.reshape(len(max_min_array), -1)
    
    return min_max_array_reshaped
    
def mean_scaler(X):
    # Calculate the mean along the last dimension (128)
    mean_array = torch.mean(X, dim=2).numpy()
    return mean_array


def master_dim_reducer(X_val, dim_reduct_method, num_dims):
    if dim_reduct_method == 'pca':
        reducer = PCA(n_components=num_dims)
        X_reduced = reducer.fit_transform(X_val)
    elif dim_reduct_method == 'pacmap':
        reducer = pacmap.PaCMAP(n_components=num_dims)
        X_reduced = reducer.fit_transform(X_val)
    elif dim_reduct_method == 'min_max':
        X_reduced = min_max_scaler(X_val)
    elif dim_reduct_method == 'average':
        X_reduced = mean_scaler(X_val)
    else:
        raise ValueError(f"Invalid dimensionality reduction method: {dim_reduct_method}")
    
    return X_reduced
    
def perform_dimensionality_reduction(X_val, config):
    
    X_val_reshaped = process_embeddings(X_val)
        
    # Extract the dimensionality reduction method
    file_name = config[0]
    dim_reduct_method_search = ['pca', 'pacmap', 'min_max', 'average']
    dim_reduct_method = [method for method in dim_reduct_method_search if method in file_name]
    assert len(dim_reduct_method) == 1, f"Invalid file name: {file_name}, expected one of {dim_reduct_method_search}"
    dim_reduct_method = dim_reduct_method[0]
    
    num_dims = int(file_name.split('.')[1].split('_')[-1])
    
    return master_dim_reducer(X_val_reshaped, dim_reduct_method, num_dims)
        
def train_best_model(params, X_train_path, y_train_path):
    X_train = np.load(X_train_path)
    y_train = smart_read_csv(y_train_path).iloc[:, 1].to_numpy().ravel()  # Adjust according to how you store labels
    
    # Initialize the model using the best parameters
    model_class = model_mapping.get(params['classifier'])
    if not model_class:
        raise ValueError(f"Classifier {params['classifier']} is not recognized.")
    model_params = {k: v for k, v in params.items() if k != 'classifier'}
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_val, y_val_path):
    y_val = smart_read_csv(y_val_path).iloc[:, 1].to_numpy().ravel()
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    return r2, mse

def run_pipeline(json_file_path, y_train_path, X_val_path, y_val_path, specific_model): # The json file has the X-train path in it.
    """
    The function `run_pipeline` takes in various file paths and a specific model, finds the best model
    configuration based on a JSON file, performs dimensionality reduction on the validation data, trains
    the best model, and evaluates its performance.
    
    :param json_file_path: The path to the JSON file that contains the configuration information for the
    pipeline, these are the raw results from optuna hyperparameter tuning, this will also be used to obtain the X_train_path
    :param y_train_path: The path to the file containing the target variable (y) values for the training
    data
    :param X_val_path: The path to the validation dataset features
    :param y_val_path: The `y_val_path` parameter is the file path to the validation set labels. It is
    used to evaluate the performance of the trained model on the validation set
    :param specific_model: The specific_model parameter is used to specify the type of model you want to
    train. It should be a string, representing the name of the model, such as "xgboost". You could also pass None, which would result
    highest scoring model to be chosen. 
    :return: the trained model, the R2 score, and the mean squared error (MSE) of the model's
    predictions on the validation data.
    """
     
    best_config, best_params, X_train_path = find_best_model_config(json_file_path, specific_model)
    X_val_reduced = perform_dimensionality_reduction(X_val_path, best_config)
    model = train_best_model(best_params, X_train_path, y_train_path)
    r2, mse = evaluate_model(model, X_val_reduced, y_val_path)

    return model, r2, mse

if __name__ == "__main__":
    # json_path = "path_to_json.json"
    # y_train_path = "path_to_y_train.npy"
    # X_val_path = "path_to_X_val.npy"
    # y_val_path = "path_to_y_val.npy"
    
    best_model, r2_score, mse_score = run_pipeline(json_path, y_train_path, X_val_path, y_val_path)
    print(f"Best Model: {best_model}")
    print(f"R2 Score: {r2_score}")
    print(f"MSE: {mse_score}")