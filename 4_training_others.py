#!/usr/bin/env python3
"""
Other Models Training Script
Converts notebook 4_training-others.ipynb to Python script
"""

import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import warnings
import os
import gc

# 设置环境变量以减少TensorFlow日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # 只显示错误信息
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 设置使用特定的GPU (可以根据需要修改)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from functions.training import train_eval_ae, train_eval_lstmae


def check_gpu():
    """Check GPU availability"""
    warnings.filterwarnings('ignore')

    print("Version of Tensorflow: ", tf.__version__)
    print("Cuda Availability: ", tf.test.is_built_with_cuda())
    print("GPU Availability: ", tf.config.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def clear_memory():
    """Clear memory to prevent OOM errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_flat_autoencoder():
    """Train Flat Autoencoder (Nolle et al., 2018, Nguyen et al., 2019)"""
    print("Training Flat Autoencoder...")

    contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    datasets = {
        'BPIC 2017': {
            'base_path': 'process_tables/bpi17',
            'attr_dims': [2, 41, 4, 2, 1, 3, 1, 14, 1, 1, 1, 145, 1, 2]
        },
        'DS2': {
            'base_path': 'process_tables/ds2',
            'attr_dims': [17, 21, 1, 20, 1]
        }
    }

    all_results = []

    for dataset_name, config in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}\n")

        for cont_rate in contamination_rates:
            print(f"\nContamination Rate: {cont_rate * 100:.0f}%")
            tab_path = f"{config['base_path']}/outlier_{cont_rate}prc/"

            if not os.path.exists(tab_path):
                print(f"Path {tab_path} does not exist, skipping...")
                continue

            try:
                # Train AE model
                print("\n" + "=" * 80)
                print(f"Training AE - {dataset_name} - {cont_rate * 100:.0f}%")
                print("=" * 80)

                if dataset_name == 'BPIC 2017':
                    train_eval_ae(tab_path, batch_size=8192, attribute_dims=config['attr_dims'])
                else:  # DS2
                    train_eval_ae(tab_path, batch_size=1, attribute_dims=config['attr_dims'])

                # Load results for comparison table
                path_parts = tab_path.strip('/').split('/')
                dataset_name_short = path_parts[-2] if len(path_parts) > 1 else 'unknown'
                cont_folder = path_parts[-1] if len(path_parts) > 0 else 'unknown'
                filename_prefix = f'{dataset_name_short}_{cont_folder}'

                ae_results = pd.read_csv(f'results/{filename_prefix}_ae_results.csv')
                ae_r10 = pd.read_csv(f'results/{filename_prefix}_ae_recall10_per_type.csv')

                all_results.append({
                    'Dataset': dataset_name,
                    'Contamination': f'{cont_rate * 100:.0f}%',
                    'Model': 'AE',
                    'AUC-ROC': f"{ae_results['AUC ROC'].mean():.4f} ± {ae_results['AUC ROC'].std():.4f}",
                    'AUC-PR': f"{ae_results['AUC Precision-Recall'].mean():.4f} ± {ae_results['AUC Precision-Recall'].std():.4f}",
                    'F1': f"{ae_results['F1'].mean():.4f} ± {ae_results['F1'].std():.4f}",
                    'Recall@10': f"{ae_results['Recall @ 10'].mean():.4f} ± {ae_results['Recall @ 10'].std():.4f}",
                    'R@10-T1': f"{ae_r10['1.0'].mean():.4f} ± {ae_r10['1.0'].std():.4f}" if '1.0' in ae_r10.columns else 'N/A',
                    'R@10-T2': f"{ae_r10['2.0'].mean():.4f} ± {ae_r10['2.0'].std():.4f}" if '2.0' in ae_r10.columns else 'N/A',
                    'R@10-T3': f"{ae_r10['3.0'].mean():.4f} ± {ae_r10['3.0'].std():.4f}" if '3.0' in ae_r10.columns else 'N/A'
                })

                # Clear memory after each training
                clear_memory()

            except Exception as e:
                print(f"Error training AE on {dataset_name} with contamination rate {cont_rate}: {str(e)}")
                print("Skipping this configuration and continuing with next...")
                clear_memory()
                continue

    return all_results


def train_lstm_autoencoder():
    """Train LSTM Autoencoder (Nolle et al., 2022, Nguyen et al., 2019, Lahann et al., 2022)"""
    print("Training LSTM Autoencoder...")

    contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    datasets = {
        'BPIC 2017': {
            'base_path': 'process_tables/bpi17',
            'attr_dims': [2, 41, 4, 2, 1, 3, 1, 14, 1, 1, 1, 145, 1, 2]
        },
        'DS2': {
            'base_path': 'process_tables/ds2',
            'attr_dims': [17, 21, 1, 20, 1]
        }
    }

    all_results = []

    for dataset_name, config in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}\n")

        for cont_rate in contamination_rates:
            print(f"\nContamination Rate: {cont_rate * 100:.0f}%")
            tab_path = f"{config['base_path']}/outlier_{cont_rate}prc/"

            if not os.path.exists(tab_path):
                print(f"Path {tab_path} does not exist, skipping...")
                continue

            try:
                # Train LSTM-AE model
                print("\n" + "=" * 80)
                print(f"Training LSTM-AE - {dataset_name} - {cont_rate * 100:.0f}%")
                print("=" * 80)

                # Adjust batch size based on dataset to prevent OOM
                if dataset_name == 'BPIC 2017':
                    batch_size = 256  # Reduced from 512 for BPIC 2017
                else:  # DS2
                    batch_size = 512

                train_eval_lstmae(tab_path, batch_size=batch_size, attribute_dims=config['attr_dims'])

                # Load results for comparison table
                path_parts = tab_path.strip('/').split('/')
                dataset_name_short = path_parts[-2] if len(path_parts) > 1 else 'unknown'
                cont_folder = path_parts[-1] if len(path_parts) > 0 else 'unknown'
                filename_prefix = f'{dataset_name_short}_{cont_folder}'

                lstmae_results = pd.read_csv(f'results/{filename_prefix}_lstmae_results.csv')
                lstmae_r10 = pd.read_csv(f'results/{filename_prefix}_lstmae_recall10_per_type.csv')

                all_results.append({
                    'Dataset': dataset_name,
                    'Contamination': f'{cont_rate * 100:.0f}%',
                    'Model': 'LSTM-AE',
                    'AUC-ROC': f"{lstmae_results['AUC ROC'].mean():.4f} ± {lstmae_results['AUC ROC'].std():.4f}",
                    'AUC-PR': f"{lstmae_results['AUC Precision-Recall'].mean():.4f} ± {lstmae_results['AUC Precision-Recall'].std():.4f}",
                    'F1': f"{lstmae_results['F1'].mean():.4f} ± {lstmae_results['F1'].std():.4f}",
                    'Recall@10': f"{lstmae_results['Recall @ 10'].mean():.4f} ± {lstmae_results['Recall @ 10'].std():.4f}",
                    'R@10-T1': f"{lstmae_r10['1.0'].mean():.4f} ± {lstmae_r10['1.0'].std():.4f}" if '1.0' in lstmae_r10.columns else 'N/A',
                    'R@10-T2': f"{lstmae_r10['2.0'].mean():.4f} ± {lstmae_r10['2.0'].std():.4f}" if '2.0' in lstmae_r10.columns else 'N/A',
                    'R@10-T3': f"{lstmae_r10['3.0'].mean():.4f} ± {lstmae_r10['3.0'].std():.4f}" if '3.0' in lstmae_r10.columns else 'N/A'
                })

                # Clear memory after each training
                clear_memory()

            except Exception as e:
                print(f"Error training LSTM-AE on {dataset_name} with contamination rate {cont_rate}: {str(e)}")
                print("Skipping this configuration and continuing with next...")
                clear_memory()
                continue

    return all_results


def main():
    print("Starting other models training...")

    # Check GPU
    check_gpu()

    # Train models
    ae_results = train_flat_autoencoder()
    lstmae_results = train_lstm_autoencoder()

    # Combine all results
    all_results = ae_results + lstmae_results

    if all_results:
        # Display final comparison
        print(f"\n\n{'=' * 80}")
        print("FINAL COMPARISON TABLE")
        print(f"{'=' * 80}\n")

        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))

        # Save
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/ae_vs_lstmae_comparison.csv', index=False)
        print(f"\n{'=' * 80}")
        print("Results saved to: results/ae_vs_lstmae_comparison.csv")
        print(f"{'=' * 80}\n")

    print("Other models training completed!")


if __name__ == "__main__":
    main()