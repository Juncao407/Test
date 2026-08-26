import warnings

warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import os
import gc
from os import listdir
from os.path import isfile, join
import pickle
import time
from tqdm import tqdm
import random

# TensorFlow logging configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across all libraries used.

    Args:
        seed (int): The seed value to use for all random number generators
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TensorFlow
    tf.random.set_seed(seed)

    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


# PyGOD imports
from pygod.metric import eval_roc_auc
from pythresh.thresholds.iqr import IQR

# Set random seeds for reproducibility
set_random_seeds(42)

# sklearn imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, auc, f1_score

# PyTorch Geometric imports
from torch_geometric import data
import torch_geometric.transforms as T

# Local imports
from .training_utils import MinMaxScaler, recall_at_k, Dataset
from models.GCNAE import GCNAE
from models.TGCAE import TGCAEDetector
from models.AE import AE
from models.LSTMAE import LSTMAE


def train_eval_gcnae(graph_path, batch_size=0, attribute_dims=None):
    start = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')

    transform = T.Compose([T.GCNNorm()])

    # Initialize lists to store the evaluation metrics
    auc_roc_list = []
    auc_pr_list = []
    r_k_list = []
    f1_list = []
    hit_rate_list = []
    recall_at_k_list_per_type = []  # Initialize a list to store the Recall@k per type

    files = [f for f in listdir(graph_path) if isfile(join(graph_path, f))]

    # Extract contamination rate from graph_path for dynamic k value
    import re
    contamination_match = re.search(r'outlier_([0-9.]+)prc', graph_path)
    if contamination_match:
        contamination_rate = float(contamination_match.group(1))
        # Convert to percentage and determine k
        k_value = int(contamination_rate * 100)  # e.g., 0.05 -> 5, 0.1 -> 10
        k_value = max(1, k_value)  # Ensure at least 1
    else:
        k_value = 10  # Default to 10

    for filename in tqdm(files):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        file_path = graph_path + filename

        glist = pickle.load(open(file_path, 'rb'))

        data_batch = data.Batch().from_data_list(glist)

        data_batch = data_batch.to(device)

        # Create y variable
        data_batch.y = data_batch.x[:, -1]

        # Store node IDs and original y values
        nodes_y_df = pd.DataFrame({"node_id": range(data_batch.y.size(0)), "original_y": data_batch.y.cpu().numpy()})

        # Save original y values before transforming to binary
        original_y = data_batch.y.cpu().numpy()

        # Transform y values to binary (0, 1)
        data_batch.y = (data_batch.y > 0).long()
        data_batch.x = data_batch.x[:, 1:-1]

        # Column-wise normalization
        data_batch.x = (data_batch.x - data_batch.x.mean(dim=0)) / data_batch.x.std(dim=0)
        data_batch = transform(data_batch)

        # Fit Model
        model = GCNAE(gpu=0, batch_size=0, encoder_layers=2, decoder_layers=1, attribute_dims=attribute_dims)
        model.fit(data_batch)

        decision_scores = model.decision_score_.cpu().numpy() if isinstance(model.decision_score_,
                                                                            torch.Tensor) else model.decision_score_

        # Handle NaN values
        if np.isnan(decision_scores).any():
            print(f"  Warning: NaN detected in {filename}, replacing with 0")
            median_val = np.nanmedian(decision_scores)
            if np.isnan(median_val):
                median_val = 0.0
            decision_scores = np.nan_to_num(decision_scores, nan=median_val)

        thres = IQR()
        labels = thres.eval(decision_scores)

        # Metrics
        auc_roc = eval_roc_auc(data_batch.y.cpu().numpy(), decision_scores)
        precision, recall, _ = precision_recall_curve(data_batch.y.cpu().numpy(), decision_scores)
        auc_pr = auc(recall, precision)
        recall_at_k_val = recall_at_k(data_batch.y.cpu().numpy(), decision_scores, k_value)
        f1 = f1_score(data_batch.y.cpu().numpy(), labels)

        # Append the evaluation metrics to the corresponding lists
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        r_k_list.append(recall_at_k_val)
        f1_list.append(f1)

        # Add predicted binary labels to the DataFrame
        nodes_y_df["predicted_binary"] = labels
        nodes_y_df["correct"] = ((nodes_y_df["predicted_binary"] == 1) & (nodes_y_df["original_y"] > 0)) | (
                (nodes_y_df["predicted_binary"] == 0) & (nodes_y_df["original_y"] == 0))

        # Calculate hit rates for each original value and store them in a dictionary
        hit_rates = nodes_y_df.groupby("original_y")["correct"].mean().to_dict()
        hit_rate_list.append(hit_rates)

        # Compute Recall @ k for each type
        # Step 1: Determine the threshold for the top k%
        threshold_percentile = 100 - k_value
        threshold = np.percentile(decision_scores, threshold_percentile)

        # Step 2: Use the threshold to set positive/negative predictions
        predicted_binary = (decision_scores >= threshold).astype(int)

        # Step 3: Compute Recall for each class
        recalls_for_each_class = {}

        original_y = np.array(original_y)
        predicted_binary = np.array(predicted_binary)

        unique_classes = np.unique(original_y)  # Get the unique classes
        for cls in unique_classes:
            true_positives = np.sum((original_y == cls) & (predicted_binary == 1))
            actual_positives = np.sum(original_y == cls)
            recall = true_positives / (actual_positives + 1e-10)  # Add a small value to prevent division by zero
            recalls_for_each_class[cls] = recall

        recall_at_k_list_per_type.append(recalls_for_each_class)

    results_dict = {'F1': f1_list,
                    'AUC ROC': auc_roc_list,
                    'AUC Precision-Recall': auc_pr_list,
                    'Recall @ k': r_k_list
                    }

    results_df = pd.DataFrame(results_dict)

    results_mean = results_df.mean()
    results_std = results_df.std()

    # Compute mean and standard deviation of hit rates
    hit_rate_df = pd.DataFrame(hit_rate_list)
    hit_rate_mean = hit_rate_df.mean()
    hit_rate_std = hit_rate_df.std()

    # Compute mean and standard deviation of hit rates
    recall_at_k_df_per_type = pd.DataFrame(recall_at_k_list_per_type)
    recall_at_k_mean_per_type = recall_at_k_df_per_type.mean()
    recall_at_k_std_per_type = recall_at_k_df_per_type.std()

    # Print the results as a table
    print('Evaluation metrics:')
    print('-------------------')
    print('{:<25s} {:<10s} {:<10s}'.format('', 'Mean', 'Std'))
    for col in results_df.columns:
        print('{:<25s} {:<10.1f} {:<10.1f}'.format(col, results_mean[col] * 100, results_std[col] * 100))

    # Save the results to a CSV file
    # Extract dataset name and contamination rate from path
    path_parts = graph_path.strip('/').split('/')
    dataset_name = path_parts[-2] if len(path_parts) > 1 else 'unknown'
    cont_folder = path_parts[-1] if len(path_parts) > 0 else 'unknown'
    filename_prefix = f'{dataset_name}_{cont_folder}'

    results_df.to_csv(f'results/{filename_prefix}_gcnae_results.csv', index=False)
    hit_rate_df.to_csv(f'results/{filename_prefix}_gcnae_hitrate.csv', index=False)
    recall_at_k_df_per_type.to_csv(f'results/{filename_prefix}_gcnae_recall{k_value}_per_type.csv', index=False)

    # Print hit rate mean and standard deviation
    print('')
    for col in hit_rate_df.columns:
        print("Hit Rate {:.0f}: {:.2f} ± {:.2f}".format(col, hit_rate_mean[col] * 100, hit_rate_std[col] * 100))

    # Print Recall @ k mean and standard deviation per type
    print(f"\nRecall @ k for each type:")
    for col in recall_at_k_df_per_type.columns:
        print("R@k-T{:.0f}: {:.2f} ± {:.2f}".format(col, recall_at_k_mean_per_type[col] * 100,
                                                    recall_at_k_std_per_type[col] * 100))

    end = time.time()
    print('')
    print('{:<25s} {:<10.1f} {:<10.1f}'.format('Time: Total / Average', end - start, (end - start) / 10))


def train_eval_ae(tab_path, batch_size=8192, attribute_dims=None):
    start = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')

    # Initialize lists to store the evaluation metrics
    auc_roc_list = []
    auc_pr_list = []
    r_10_list = []
    f1_list = []
    hit_rate_list = []
    recall_at_k_list_per_type = []  # Initialize a list to store the Recall@k per type

    files = [f for f in listdir(tab_path) if isfile(join(tab_path, f))]

    # Extract contamination rate from tab_path for dynamic k value
    import re
    contamination_match = re.search(r'outlier_([0-9.]+)prc', tab_path)
    if contamination_match:
        contamination_rate = float(contamination_match.group(1))
        # Convert to percentage and determine k
        k_value = int(contamination_rate * 100)  # e.g., 0.05 -> 5, 0.1 -> 10
        k_value = max(1, k_value)  # Ensure at least 1
    else:
        k_value = 10  # Default to 10

    for filename in tqdm(files):
        file_path = tab_path + filename

        df_tab = pd.read_csv(file_path, dtype={'exec_id': int, 'y': int})
        df_tab = df_tab.sort_values(by=['exec_id', 'elapsed_time'])
        original_y = df_tab[['event_id', 'exec_id', 'y']]
        df_tab.y = (df_tab.y > 0).astype(int)

        X = df_tab.drop(columns=['event_id', 'elapsed_time', 'y'])
        X = X.pivot_table(index='exec_id', columns=X.groupby('exec_id').cumcount())
        X = X.sort_index(axis='columns', level=1)
        X.columns = X.columns.map('{0[0]}|{0[1]}'.format)
        X_mask = X.isnull()
        X = X.fillna(0)

        # Convert DataFrame to tensor
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(data=X_scaled, columns=X.columns)

        # Prepare Dataset Object
        dataset = Dataset(X, X.iloc[:, :1], df_tab.groupby('exec_id').size().max(), attribute_dims, X_mask)

        # Instantiate AE class
        ae = AE()

        # Construct the model with your dataset
        model, features, _ = ae.model_fn(dataset, **ae.config)

        # Train the model
        model.fit(features, features, epochs=100, batch_size=batch_size, verbose=False)

        # Assign the trained model to the DAE instance
        ae.model = model

        # Now you can use your trained model to detect anomalies
        decision_scores = ae.detect(dataset)

        # Apply threshold
        thres = IQR()
        labels = thres.eval(decision_scores)

        # Metrics
        try:
            auc_roc = eval_roc_auc(df_tab.y, decision_scores)
        except Exception as e:
            print(f"  Warning: Error calculating AUC-ROC in {filename}: {e}, using default value")
            auc_roc = 0.5

        try:
            precision, recall, _ = precision_recall_curve(df_tab.y, decision_scores)
            auc_pr = auc(recall, precision)
        except Exception as e:
            print(f"  Warning: Error calculating AUC-PR in {filename}: {e}, using default value")
            auc_pr = 0.5

        try:
            # Compute Recall @ k with k depending on contamination rate
            recall_at_k_val = recall_at_k(df_tab.y, decision_scores, k_value)
        except Exception as e:
            print(f"  Warning: Error calculating Recall@{k_value} in {filename}: {e}, using default value")
            recall_at_k_val = 0.0

        try:
            f1 = f1_score(df_tab.y, labels)
        except Exception as e:
            print(f"  Warning: Error calculating F1 in {filename}: {e}, using default value")
            f1 = 0.0

        # Append the evaluation metrics to the corresponding lists
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        r_10_list.append(recall_at_k_val)
        f1_list.append(f1)

        # Add predicted binary labels to the DataFrame
        original_y["predicted_binary"] = labels
        original_y["correct"] = ((original_y["predicted_binary"] == 1) & (original_y["y"] > 0)) | (
                (original_y["predicted_binary"] == 0) & (original_y["y"] == 0))

        # Calculate hit rates for each original value and store them in a dictionary
        hit_rates = original_y.groupby("y")["correct"].mean().to_dict()
        hit_rate_list.append(hit_rates)

        # Compute Recall @ k for each type
        # Step 1: Determine the threshold for the top k%
        threshold_percentile = 100 - k_value
        threshold = np.percentile(decision_scores, threshold_percentile)

        # Step 2: Use the threshold to set positive/negative predictions
        predicted_binary = (decision_scores >= threshold).astype(int)

        # Step 3: Compute Recall for each class
        recalls_for_each_class = {}

        original_y_values = original_y['y'].values
        predicted_binary = np.array(predicted_binary)

        unique_classes = np.unique(original_y_values)  # Get the unique classes
        for cls in unique_classes:
            true_positives = np.sum((original_y_values == cls) & (predicted_binary == 1))
            actual_positives = np.sum(original_y_values == cls)
            recall = true_positives / (actual_positives + 1e-10)  # Add a small value to prevent division by zero
            recalls_for_each_class[cls] = recall

        recall_at_k_list_per_type.append(recalls_for_each_class)

        gc.collect()

    results_dict = {'F1': f1_list,
                    'AUC ROC': auc_roc_list,
                    'AUC Precision-Recall': auc_pr_list,
                    'Recall @ k': r_10_list
                    }

    results_df = pd.DataFrame(results_dict)

    results_mean = results_df.mean()
    results_std = results_df.std()

    # Compute mean and standard deviation of hit rates
    hit_rate_df = pd.DataFrame(hit_rate_list)
    hit_rate_mean = hit_rate_df.mean()
    hit_rate_std = hit_rate_df.std()

    # Compute mean and standard deviation of hit rates
    recall_at_k_df_per_type = pd.DataFrame(recall_at_k_list_per_type)
    recall_at_k_mean_per_type = recall_at_k_df_per_type.mean()
    recall_at_k_std_per_type = recall_at_k_df_per_type.std()

    # Print the results as a table
    print('Evaluation metrics:')
    print('-------------------')
    print('{:<25s} {:<10s} {:<10s}'.format('', 'Mean', 'Std'))
    for col in results_df.columns:
        print('{:<25s} {:<10.1f} {:<10.1f}'.format(col, results_mean[col] * 100, results_std[col] * 100))

    # Save the results to a CSV file
    results_df.to_csv('results/' + tab_path.strip('/').split('/')[-1] + '_ae_results.csv', index=False)
    hit_rate_df.to_csv('results/' + tab_path.strip('/').split('/')[-1] + '_ae_hitrate.csv', index=False)
    recall_at_k_df_per_type.to_csv(
        'results/' + tab_path.strip('/').split('/')[-1] + f'_ae_recall{k_value}_per_type.csv',
        index=False)

    # Print hit rate mean and standard deviation
    print('')
    for col in hit_rate_df.columns:
        print("Hit Rate {:.0f}: {:.2f} ± {:.2f}".format(col, hit_rate_mean[col] * 100, hit_rate_std[col] * 100))

    # Print Recall @ k mean and standard deviation per type
    print(f"\nRecall @ k for each type:")
    for col in recall_at_k_df_per_type.columns:
        print("R@k-T{:.0f}: {:.2f} ± {:.2f}".format(col, recall_at_k_mean_per_type[col] * 100,
                                                    recall_at_k_std_per_type[col] * 100))

    end = time.time()
    print('')
    print('{:<25s} {:<10.1f} {:<10.1f}'.format('Time: Total / Average', end - start, (end - start) / 10))


def train_eval_lstmae(tab_path, batch_size=8192, attribute_dims=None):
    start = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')

    # Initialize lists to store the evaluation metrics
    auc_roc_list = []
    auc_pr_list = []
    r_10_list = []
    f1_list = []
    hit_rate_list = []
    recall_at_k_list_per_type = []  # Initialize a list to store the Recall@k per type

    files = [f for f in listdir(tab_path) if isfile(join(tab_path, f))]

    # Extract contamination rate from tab_path for dynamic k value
    import re
    contamination_match = re.search(r'outlier_([0-9.]+)prc', tab_path)
    if contamination_match:
        contamination_rate = float(contamination_match.group(1))
        # Convert to percentage and determine k
        k_value = int(contamination_rate * 100)  # e.g., 0.05 -> 5, 0.1 -> 10
        k_value = max(1, k_value)  # Ensure at least 1
    else:
        k_value = 10  # Default to 10

    for filename in tqdm(files):
        file_path = tab_path + filename

        df_tab = pd.read_csv(file_path, dtype={'exec_id': int, 'y': int})
        df_tab = df_tab.sort_values(by=['exec_id', 'elapsed_time'])
        original_y = df_tab[['event_id', 'exec_id', 'y']]
        df_tab.y = (df_tab.y > 0).astype(int)

        X = df_tab.drop(columns=['event_id', 'elapsed_time', 'y'])
        X = X.pivot_table(index='exec_id', columns=X.groupby('exec_id').cumcount())
        X = X.sort_index(axis='columns', level=1)
        X.columns = X.columns.map('{0[0]}|{0[1]}'.format)
        X_mask = X.isnull()
        X = X.fillna(0)

        # Convert DataFrame to tensor
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(data=X_scaled, columns=X.columns)

        # Prepare Dataset Object
        dataset = Dataset(X, X.iloc[:, :1], df_tab.groupby('exec_id').size().max(), attribute_dims, X_mask)

        # Instantiate LSTMAE class
        lstmae = LSTMAE()

        # Construct the model with your dataset
        model, features, _ = lstmae.model_fn(dataset)

        # Train the model
        model.fit(features, features, epochs=100, batch_size=batch_size, verbose=False)

        # Assign the trained model to the DAE instance
        lstmae.model = model

        # Now you can use your trained model to detect anomalies
        decision_scores = lstmae.detect(dataset)

        # Apply threshold
        thres = IQR()
        labels = thres.eval(decision_scores)

        # Metrics
        try:
            auc_roc = eval_roc_auc(df_tab.y, decision_scores)
        except Exception as e:
            print(f"  Warning: Error calculating AUC-ROC in {filename}: {e}, using default value")
            auc_roc = 0.5

        try:
            precision, recall, _ = precision_recall_curve(df_tab.y, decision_scores)
            auc_pr = auc(recall, precision)
        except Exception as e:
            print(f"  Warning: Error calculating AUC-PR in {filename}: {e}, using default value")
            auc_pr = 0.5

        try:
            # Compute Recall @ k with k depending on contamination rate
            recall_at_k_val = recall_at_k(df_tab.y, decision_scores, k_value)
        except Exception as e:
            print(f"  Warning: Error calculating Recall@{k_value} in {filename}: {e}, using default value")
            recall_at_k_val = 0.0

        try:
            f1 = f1_score(df_tab.y, labels)
        except Exception as e:
            print(f"  Warning: Error calculating F1 in {filename}: {e}, using default value")
            f1 = 0.0

        # Append the evaluation metrics to the corresponding lists
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        r_10_list.append(recall_at_k_val)
        f1_list.append(f1)

        # Add predicted binary labels to the DataFrame
        original_y["predicted_binary"] = labels
        original_y["correct"] = ((original_y["predicted_binary"] == 1) & (original_y["y"] > 0)) | (
                (original_y["predicted_binary"] == 0) & (original_y["y"] == 0))

        # Calculate hit rates for each original value and store them in a dictionary
        hit_rates = original_y.groupby("y")["correct"].mean().to_dict()
        hit_rate_list.append(hit_rates)

        # Compute Recall @ k for each type
        # Step 1: Determine the threshold for the top k%
        threshold_percentile = 100 - k_value
        threshold = np.percentile(decision_scores, threshold_percentile)

        # Step 2: Use the threshold to set positive/negative predictions
        predicted_binary = (decision_scores >= threshold).astype(int)

        # Step 3: Compute Recall for each class
        recalls_for_each_class = {}

        original_y_values = original_y['y'].values
        predicted_binary = np.array(predicted_binary)

        unique_classes = np.unique(original_y_values)  # Get the unique classes
        for cls in unique_classes:
            true_positives = np.sum((original_y_values == cls) & (predicted_binary == 1))
            actual_positives = np.sum(original_y_values == cls)
            recall = true_positives / (actual_positives + 1e-10)  # Add a small value to prevent division by zero
            recalls_for_each_class[cls] = recall

        recall_at_k_list_per_type.append(recalls_for_each_class)

    results_dict = {'F1': f1_list,
                    'AUC ROC': auc_roc_list,
                    'AUC Precision-Recall': auc_pr_list,
                    'Recall @ k': r_10_list,
                    }

    results_df = pd.DataFrame(results_dict)

    results_mean = results_df.mean()
    results_std = results_df.std()

    # Compute mean and standard deviation of hit rates
    hit_rate_df = pd.DataFrame(hit_rate_list)
    hit_rate_mean = hit_rate_df.mean()
    hit_rate_std = hit_rate_df.std()

    # Compute mean and standard deviation of hit rates
    recall_at_k_df_per_type = pd.DataFrame(recall_at_k_list_per_type)
    recall_at_k_mean_per_type = recall_at_k_df_per_type.mean()
    recall_at_k_std_per_type = recall_at_k_df_per_type.std()

    # Print the results as a table
    print('Evaluation metrics:')
    print('-------------------')
    print('{:<25s} {:<10s} {:<10s}'.format('', 'Mean', 'Std'))
    for col in results_df.columns:
        print('{:<25s} {:<10.1f} {:<10.1f}'.format(col, results_mean[col] * 100, results_std[col] * 100))

    # Save the results to a CSV file
    results_df.to_csv('results/' + tab_path.strip('/').split('/')[-1] + '_lstmae_results.csv', index=False)
    hit_rate_df.to_csv('results/' + tab_path.strip('/').split('/')[-1] + '_lstmae_hitrate.csv', index=False)
    recall_at_k_df_per_type.to_csv(
        'results/' + tab_path.strip('/').split('/')[-1] + f'_lstmae_recall{k_value}_per_type.csv',
        index=False)

    # Print hit rate mean and standard deviation
    print('')
    for col in hit_rate_df.columns:
        print("Hit Rate {:.0f}: {:.2f} ± {:.2f}".format(col, hit_rate_mean[col] * 100, hit_rate_std[col] * 100))

    # Print Recall @ k mean and standard deviation per type
    print(f"\nRecall @ k for each type:")
    for col in recall_at_k_df_per_type.columns:
        print("R@k-T{:.0f}: {:.2f} ± {:.2f}".format(col, recall_at_k_mean_per_type[col] * 100,
                                                    recall_at_k_std_per_type[col] * 100))

    end = time.time()
    print('')
    print('{:<25s} {:<10.1f} {:<10.1f}'.format('Time: Total / Average', end - start, (end - start) / 10))


def train_eval_tgcae(graph_path, batch_size=0, attribute_dims=None, temporal_blend_ratio=0.0,
                      type1_weight=1.2, type2_weight=0.5, type3_weight=0.0):
    start = time.time()

    if not os.path.exists('results'):
        os.makedirs('results')

    transform = T.Compose([T.GCNNorm()])

    # Initialize lists to store the evaluation metrics
    auc_roc_list = []
    auc_pr_list = []
    r_k_list = []
    f1_list = []
    hit_rate_list = []
    recall_at_k_list_per_type = []  # Initialize a list to store the Recall@k per type

    files = [f for f in listdir(graph_path) if isfile(join(graph_path, f))]

    # Extract contamination rate from graph_path for dynamic k value
    import re
    contamination_match = re.search(r'outlier_([0-9.]+)prc', graph_path)
    if contamination_match:
        contamination_rate = float(contamination_match.group(1))
        # Convert to percentage and determine k
        k_value = int(contamination_rate * 100)  # e.g., 0.05 -> 5, 0.1 -> 10
        k_value = max(1, k_value)  # Ensure at least 1
    else:
        k_value = 10  # Default to 10

    for filename in tqdm(files):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        file_path = graph_path + filename

        glist = pickle.load(open(file_path, 'rb'))

        data_batch = data.Batch().from_data_list(glist)

        data_batch = data_batch.to(device)

        # Create y variable
        data_batch.y = data_batch.x[:, -1]

        # Store node IDs and original y values
        nodes_y_df = pd.DataFrame({"node_id": range(data_batch.y.size(0)), "original_y": data_batch.y.cpu().numpy()})

        # Save original y values before transforming to binary
        original_y = data_batch.y.cpu().numpy()

        # Transform y values to binary (0, 1)
        data_batch.y = (data_batch.y > 0).long()

        # Extract and normalize timestamp
        timestamps = data_batch.x[:, 0]  # First column is timestamp
        # Use robust normalization to handle outliers in timestamps
        from sklearn.preprocessing import RobustScaler
        timestamp_scaler = RobustScaler()
        timestamps_norm = timestamp_scaler.fit_transform(timestamps.cpu().numpy().reshape(-1, 1))
        data_batch.timestamps = torch.from_numpy(timestamps_norm).squeeze().to(device)

        # Remove timestamp column from features (keep only non-timestamp features)
        data_batch.x = data_batch.x[:, 1:-1]

        # Enhanced column-wise normalization with NaN handling
        x_vals = data_batch.x
        # Check for NaN or infinite values before normalization
        if torch.isnan(x_vals).any() or torch.isinf(x_vals).any():
            # Replace NaN with 0 and infinite values with large finite numbers
            x_vals = torch.nan_to_num(x_vals, nan=0.0, posinf=1e6, neginf=-1e6)

        # Calculate mean and std with protection against zero std
        x_mean = x_vals.mean(dim=0)
        x_std = x_vals.std(dim=0)

        # Replace zero std with 1 to avoid division by zero
        x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)

        # Normalize
        data_batch.x = (x_vals - x_mean) / x_std
        data_batch = transform(data_batch)

        # Fit Model - 使用增强的 TGCN 模型配置
        model = TGCAEDetector(gpu=0, batch_size=0, encoder_layers=5, decoder_layers=1, attribute_dims=attribute_dims,
                               backbone_type="GIN", dropout=0.1, hid_dim=128,  # 增加编码器层数以提升表达能力
                               type1_weight=type1_weight, type2_weight=type2_weight, type3_weight=type3_weight)
        model.fit(data_batch)

        decision_scores = model.decision_score_.cpu().numpy() if isinstance(model.decision_score_,
                                                                            torch.Tensor) else model.decision_score_

        # Enhanced NaN values handling
        if np.isnan(decision_scores).any():
            print(f"  Warning: NaN detected in {filename}, replacing with median value")
            # Use median instead of 0 for more robust replacement
            median_val = np.nanmedian(decision_scores)
            if np.isnan(median_val):
                median_val = 0.0
            decision_scores = np.nan_to_num(decision_scores, nan=median_val)

        # Handle infinite values as well
        if np.isinf(decision_scores).any():
            print(f"  Warning: Infinite values detected in {filename}, replacing with min/max finite values")
            # Replace infinite values with min/max of finite values
            finite_vals = decision_scores[np.isfinite(decision_scores)]
            if len(finite_vals) > 0:
                max_finite = np.max(finite_vals)
                min_finite = np.min(finite_vals)
                decision_scores = np.nan_to_num(decision_scores, posinf=max_finite, neginf=min_finite)
            else:
                decision_scores = np.zeros_like(decision_scores)

        thres = IQR()
        labels = thres.eval(decision_scores)

        # Metrics with additional safety checks
        try:
            y_true = data_batch.y.cpu().numpy()
            # Ensure we have both classes for ROC calculation
            if len(np.unique(y_true)) < 2:
                print(f"  Warning: Less than 2 classes in {filename}, using default AUC")
                auc_roc = 0.5  # Random guess AUC
            else:
                auc_roc = eval_roc_auc(y_true, decision_scores)
        except Exception as e:
            print(f"  Warning: Error calculating AUC-ROC in {filename}: {e}, using default value")
            auc_roc = 0.5

        try:
            precision, recall, _ = precision_recall_curve(y_true, decision_scores)
            auc_pr = auc(recall, precision)
        except Exception as e:
            print(f"  Warning: Error calculating AUC-PR in {filename}: {e}, using default value")
            auc_pr = 0.5

        try:
            # Compute Recall @ k with k depending on contamination rate
            recall_at_k_val = recall_at_k(y_true, decision_scores, k_value)
        except Exception as e:
            print(f"  Warning: Error calculating Recall@{k_value} in {filename}: {e}, using default value")
            recall_at_k_val = 0.0

        try:
            f1 = f1_score(y_true, labels)
        except Exception as e:
            print(f"  Warning: Error calculating F1 in {filename}: {e}, using default value")
            f1 = 0.0

        # Append the evaluation metrics to the corresponding lists
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        r_k_list.append(recall_at_k_val)
        f1_list.append(f1)

        # Add predicted binary labels to the DataFrame
        nodes_y_df["predicted_binary"] = labels
        nodes_y_df["correct"] = ((nodes_y_df["predicted_binary"] == 1) & (nodes_y_df["original_y"] > 0)) | (
                (nodes_y_df["predicted_binary"] == 0) & (nodes_y_df["original_y"] == 0))

        # Calculate hit rates for each original value and store them in a dictionary
        hit_rates = nodes_y_df.groupby("original_y")["correct"].mean().to_dict()
        hit_rate_list.append(hit_rates)

        # Compute Recall @ k for each type
        # Step 1: Determine the threshold for the top k%
        threshold_percentile = 100 - k_value
        threshold = np.percentile(decision_scores, threshold_percentile)

        # Step 2: Use the threshold to set positive/negative predictions
        predicted_binary = (decision_scores >= threshold).astype(int)

        # Step 3: Compute Recall for each class
        recalls_for_each_class = {}

        original_y = np.array(original_y)
        predicted_binary = np.array(predicted_binary)

        unique_classes = np.unique(original_y)  # Get the unique classes
        for cls in unique_classes:
            true_positives = np.sum((original_y == cls) & (predicted_binary == 1))
            actual_positives = np.sum(original_y == cls)
            recall = true_positives / (actual_positives + 1e-10)  # Add a small value to prevent division by zero
            recalls_for_each_class[cls] = recall

        recall_at_k_list_per_type.append(recalls_for_each_class)

    results_dict = {'F1': f1_list,
                    'AUC ROC': auc_roc_list,
                    'AUC Precision-Recall': auc_pr_list,
                    'Recall @ k': r_k_list
                    }

    results_df = pd.DataFrame(results_dict)

    results_mean = results_df.mean()
    results_std = results_df.std()

    # Compute mean and standard deviation of hit rates
    hit_rate_df = pd.DataFrame(hit_rate_list)
    hit_rate_mean = hit_rate_df.mean()
    hit_rate_std = hit_rate_df.std()

    # Compute mean and standard deviation of hit rates
    recall_at_k_df_per_type = pd.DataFrame(recall_at_k_list_per_type)
    recall_at_k_mean_per_type = recall_at_k_df_per_type.mean()
    recall_at_k_std_per_type = recall_at_k_df_per_type.std()

    # Print the results as a table
    print('Evaluation metrics:')
    print('-------------------')
    print('{:<25s} {:<10s} {:<10s}'.format('', 'Mean', 'Std'))
    for col in results_df.columns:
        print('{:<25s} {:<10.1f} {:<10.1f}'.format(col, results_mean[col] * 100, results_std[col] * 100))

    # Save the results to a CSV file
    # Extract dataset name and contamination rate from path
    path_parts = graph_path.strip('/').split('/')
    dataset_name = path_parts[-2] if len(path_parts) > 1 else 'unknown'
    cont_folder = path_parts[-1] if len(path_parts) > 0 else 'unknown'
    filename_prefix = f'{dataset_name}_{cont_folder}'

    results_df.to_csv(f'results/{filename_prefix}_tgcae_results.csv', index=False)
    hit_rate_df.to_csv(f'results/{filename_prefix}_tgcae_hitrate.csv', index=False)
    recall_at_k_df_per_type.to_csv(f'results/{filename_prefix}_tgcae_recall{k_value}_per_type.csv', index=False)

    # Print hit rate mean and standard deviation
    print('')
    for col in hit_rate_df.columns:
        print("Hit Rate {:.0f}: {:.2f} ± {:.2f}".format(col, hit_rate_mean[col] * 100, hit_rate_std[col] * 100))

    # Print Recall @ k mean and standard deviation per type
    print(f"\nRecall @ k for each type:")
    for col in recall_at_k_df_per_type.columns:
        print("R@k-T{:.0f}: {:.2f} ± {:.2f}".format(col, recall_at_k_mean_per_type[col] * 100,
                                                    recall_at_k_std_per_type[col] * 100))

    end = time.time()
    print('')
    print('{:<25s} {:<10.1f} {:<10.1f}'.format('Time: Total / Average', end - start, (end - start) / 10))


def train_eval_tgcae_ablation(graph_path, batch_size=0, attribute_dims=None, variant='full'):
    """训练和评估TGCAE消融变体
    
    Args:
        graph_path: 图数据路径
        batch_size: 批次大小
        attribute_dims: 属性维度
        variant: 变体类型 ('full', 'no_time', 'no_gate', 'fixed_weight')
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GIN
    
    # 定义消融变体模型
    class TemporalGCNAE_NoTime(nn.Module):
        def __init__(self, in_dim, hid_dim, encoder_layers=2, decoder_layers=1, dropout=0., backbone=GIN):
            super().__init__()
            self.encoder = backbone(in_dim, hid_dim, encoder_layers, dropout=dropout, act=F.relu)
            self.decoder = backbone(hid_dim, in_dim, decoder_layers, dropout=dropout, act=F.relu)
            self.emb = None
        
        def forward(self, x, edge_index, timestamps=None):
            h = self.encoder(x, edge_index)
            self.emb = h
            x_recon = self.decoder(h, edge_index)
            return x_recon
    
    class TemporalGCNAE_NoGate(nn.Module):
        def __init__(self, in_dim, hid_dim, encoder_layers=2, decoder_layers=1, dropout=0., backbone=GIN):
            super().__init__()
            self.encoder = backbone(in_dim, hid_dim, encoder_layers, dropout=dropout, act=F.relu)
            self.time_encoder = nn.Sequential(
                nn.Linear(1, hid_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hid_dim // 2, hid_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hid_dim, hid_dim), nn.Tanh()
            )
            self.decoder = backbone(hid_dim, in_dim, decoder_layers, dropout=dropout, act=F.relu)
            self.emb = None
        
        def forward(self, x, edge_index, timestamps=None):
            h = self.encoder(x, edge_index)
            self.emb = h
            if timestamps is not None:
                time_emb = self.time_encoder(timestamps.unsqueeze(-1))
                h_final = h + time_emb  # 直接相加，无门控
            else:
                h_final = h
            x_recon = self.decoder(h_final, edge_index)
            return x_recon
    
    class TemporalGCNAE_FixedWeight(nn.Module):
        def __init__(self, in_dim, hid_dim, encoder_layers=2, decoder_layers=1, dropout=0., backbone=GIN):
            super().__init__()
            self.encoder = backbone(in_dim, hid_dim, encoder_layers, dropout=dropout, act=F.relu)
            self.time_encoder = nn.Sequential(
                nn.Linear(1, hid_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hid_dim // 2, hid_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hid_dim, hid_dim), nn.Tanh()
            )
            self.fusion = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim), nn.LayerNorm(hid_dim),
                nn.ReLU(), nn.Dropout(dropout)
            )
            self.decoder = backbone(hid_dim, in_dim, decoder_layers, dropout=dropout, act=F.relu)
            self.emb = None
            self.fixed_gate = 1
        
        def forward(self, x, edge_index, timestamps=None):
            h = self.encoder(x, edge_index)
            self.emb = h
            if timestamps is not None:
                time_emb = self.time_encoder(timestamps.unsqueeze(-1))
                h_combined = torch.cat([h, time_emb], dim=-1)
                h_fused = self.fusion(h_combined)
                h_final = (1 - self.fixed_gate) * h + self.fixed_gate * h_fused
            else:
                h_final = h
            x_recon = self.decoder(h_final, edge_index)
            return x_recon
    
    # 创建对应的Detector类
    class TGCAEDetector_Variant(TGCAEDetector):
        def __init__(self, variant_type, *args, **kwargs):
            self.variant_type = variant_type
            super().__init__(*args, **kwargs)
        
        def init_model(self, **kwargs):
            if self.variant_type == 'no_time':
                return TemporalGCNAE_NoTime(self.in_dim, self.hid_dim, self.encoder_layers,
                                           self.decoder_layers, self.dropout, GIN)
            elif self.variant_type == 'no_gate':
                return TemporalGCNAE_NoGate(self.in_dim, self.hid_dim, self.encoder_layers,
                                           self.decoder_layers, self.dropout, GIN)
            elif self.variant_type == 'fixed_weight':
                return TemporalGCNAE_FixedWeight(self.in_dim, self.hid_dim, self.encoder_layers,
                                                self.decoder_layers, self.dropout, GIN)
            else:  # full
                from models.TGCAE import TemporalGCNAE
                return TemporalGCNAE(self.in_dim, self.hid_dim, self.encoder_layers,
                                    self.decoder_layers, self.dropout, GIN)
    
    start = time.time()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    transform = T.Compose([T.GCNNorm()])
    
    auc_roc_list, auc_pr_list, r_k_list, f1_list, recall_at_k_list_per_type = [], [], [], [], []
    files = [f for f in listdir(graph_path) if isfile(join(graph_path, f))]
    
    import re
    contamination_match = re.search(r'outlier_([0-9.]+)prc', graph_path)
    k_value = int(float(contamination_match.group(1)) * 100) if contamination_match else 10
    k_value = max(1, k_value)
    
    for filename in tqdm(files):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        glist = pickle.load(open(graph_path + filename, 'rb'))
        data_batch = data.Batch().from_data_list(glist).to(device)
        
        data_batch.y = data_batch.x[:, -1]
        original_y = data_batch.y.cpu().numpy()
        data_batch.y = (data_batch.y > 0).long()
        
        timestamps = data_batch.x[:, 0]
        timestamp_scaler = RobustScaler()
        timestamps_norm = timestamp_scaler.fit_transform(timestamps.cpu().numpy().reshape(-1, 1))
        data_batch.timestamps = torch.from_numpy(timestamps_norm).squeeze().to(device)
        
        data_batch.x = data_batch.x[:, 1:-1]
        x_vals = torch.nan_to_num(data_batch.x, nan=0.0, posinf=1e6, neginf=-1e6)
        x_mean, x_std = x_vals.mean(dim=0), x_vals.std(dim=0)
        x_std = torch.where(x_std == 0, torch.ones_like(x_std), x_std)
        data_batch.x = (x_vals - x_mean) / x_std
        data_batch = transform(data_batch)
        
        model = TGCAEDetector_Variant(variant, gpu=0, batch_size=0, encoder_layers=5, decoder_layers=1,
                                      attribute_dims=attribute_dims, backbone_type="GIN", dropout=0.1, hid_dim=128)
        model.fit(data_batch)
        
        decision_scores = model.decision_score_.cpu().numpy() if isinstance(model.decision_score_, torch.Tensor) else model.decision_score_
        decision_scores = np.nan_to_num(decision_scores, nan=np.nanmedian(decision_scores) if not np.isnan(np.nanmedian(decision_scores)) else 0.0)
        
        if np.isinf(decision_scores).any():
            finite_vals = decision_scores[np.isfinite(decision_scores)]
            if len(finite_vals) > 0:
                decision_scores = np.nan_to_num(decision_scores, posinf=np.max(finite_vals), neginf=np.min(finite_vals))
        
        thres = IQR()
        labels = thres.eval(decision_scores)
        y_true = data_batch.y.cpu().numpy()
        
        try:
            auc_roc = eval_roc_auc(y_true, decision_scores) if len(np.unique(y_true)) >= 2 else 0.5
        except:
            auc_roc = 0.5
        
        try:
            precision, recall, _ = precision_recall_curve(y_true, decision_scores)
            auc_pr = auc(recall, precision)
        except:
            auc_pr = 0.5
        
        try:
            recall_at_k_val = recall_at_k(y_true, decision_scores, k_value)
        except:
            recall_at_k_val = 0.0
        
        try:
            f1 = f1_score(y_true, labels)
        except:
            f1 = 0.0
        
        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        r_k_list.append(recall_at_k_val)
        f1_list.append(f1)
        
        threshold = np.percentile(decision_scores, 100 - k_value)
        predicted_binary = (decision_scores >= threshold).astype(int)
        
        recalls_for_each_class = {}
        for cls in np.unique(original_y):
            tp = np.sum((original_y == cls) & (predicted_binary == 1))
            ap = np.sum(original_y == cls)
            recalls_for_each_class[cls] = tp / (ap + 1e-10)
        
        recall_at_k_list_per_type.append(recalls_for_each_class)
    
    results_df = pd.DataFrame({'F1': f1_list, 'AUC ROC': auc_roc_list, 'AUC Precision-Recall': auc_pr_list, 'Recall @ k': r_k_list})
    recall_at_k_df_per_type = pd.DataFrame(recall_at_k_list_per_type)
    
    results_mean = results_df.mean()
    results_std = results_df.std()
    
    print('Evaluation metrics:')
    print('-------------------')
    print('{:<25s} {:<10s} {:<10s}'.format('', 'Mean', 'Std'))
    for col in results_df.columns:
        print('{:<25s} {:<10.1f} {:<10.1f}'.format(col, results_mean[col] * 100, results_std[col] * 100))
    
    path_parts = graph_path.strip('/').split('/')
    dataset_name = path_parts[-2] if len(path_parts) > 1 else 'unknown'
    cont_folder = path_parts[-1] if len(path_parts) > 0 else 'unknown'
    filename_prefix = f'{dataset_name}_{cont_folder}'
    
    results_df.to_csv(f'results/{filename_prefix}_tgcae_{variant}_results.csv', index=False)
    recall_at_k_df_per_type.to_csv(f'results/{filename_prefix}_tgcae_{variant}_recall{k_value}_per_type.csv', index=False)
    
    print(f'\nRecall @ k for each type:')
    for col in recall_at_k_df_per_type.columns:
        print(f"R@k-T{col:.0f}: {recall_at_k_df_per_type[col].mean() * 100:.2f} ± {recall_at_k_df_per_type[col].std() * 100:.2f}")
    
    print(f'\nTime: {time.time() - start:.1f}s')
