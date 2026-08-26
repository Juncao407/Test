#!/usr/bin/env python3
"""Compare GCNAE vs TGCAE under different contamination rates"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import re
import torch
from functions.training import train_eval_gcnae, train_eval_tgcae

# 在设置环境变量后清理GPU内存
torch.cuda.empty_cache()

def main():
    # contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    datasets = {
        'BPIC 2017': {
            'base_path': 'process_graphs/bpi17',
            'attr_dims': [41, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 3, 14, 145]
        },
        'DS2': {
            'base_path': 'process_graphs/ds2',
            'attr_dims': [21, 1, 1, 20, 17]
        }
    }
    
    all_results = []
    
    for dataset_name, config in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        for cont_rate in contamination_rates:
            print(f"\nContamination Rate: {cont_rate*100:.0f}%")
            graph_path = f"{config['base_path']}/outlier_{cont_rate}prc/"
            
            if not os.path.exists(graph_path):
                continue
            
            # Calculate k value based on contamination rate
            k_value = int(cont_rate * 100)
            k_value = max(1, k_value)
            
            # Train GCNAE with detailed metrics
            print("\n" + "="*80)
            print(f"Training GCNAE - {dataset_name} - {cont_rate*100:.0f}%")
            print("="*80)
            # train_eval_gcnae(graph_path, batch_size=0, attribute_dims=config['attr_dims'])

            # Train TGCAE with detailed metrics
            print("\n" + "="*80)
            print(f"Training TGCAE - {dataset_name} - {cont_rate*100:.0f}%")
            print("="*80)
            train_eval_tgcae(graph_path, batch_size=0, attribute_dims=config['attr_dims'])

            # Load results for comparison table
            path_parts = graph_path.strip('/').split('/')
            dataset_name_short = path_parts[-2] if len(path_parts) > 1 else 'unknown'
            cont_folder = path_parts[-1] if len(path_parts) > 0 else 'unknown'
            filename_prefix = f'{dataset_name_short}_{cont_folder}'
            
            gcnae_results = pd.read_csv(f'results/{filename_prefix}_gcnae_results.csv')
            tgcae_results = pd.read_csv(f'results/{filename_prefix}_tgcae_results.csv')
            gcnae_rk = pd.read_csv(f'results/{filename_prefix}_gcnae_recall{k_value}_per_type.csv')
            tgcae_rk = pd.read_csv(f'results/{filename_prefix}_tgcae_recall{k_value}_per_type.csv')
            
            all_results.append({
                'Dataset': dataset_name,
                'Contamination': f'{cont_rate*100:.0f}%',
                'Model': 'GCNAE',
                'AUC-ROC': f"{gcnae_results['AUC ROC'].mean():.4f} ± {gcnae_results['AUC ROC'].std():.4f}",
                'AUC-PR': f"{gcnae_results['AUC Precision-Recall'].mean():.4f} ± {gcnae_results['AUC Precision-Recall'].std():.4f}",
                'F1': f"{gcnae_results['F1'].mean():.4f} ± {gcnae_results['F1'].std():.4f}",
                'Recall@k': f"{gcnae_results['Recall @ k'].mean():.4f} ± {gcnae_results['Recall @ k'].std():.4f}",
                'R@k-T1': f"{gcnae_rk['1.0'].mean():.4f} ± {gcnae_rk['1.0'].std():.4f}" if '1.0' in gcnae_rk.columns else 'N/A',
                'R@k-T2': f"{gcnae_rk['2.0'].mean():.4f} ± {gcnae_rk['2.0'].std():.4f}" if '2.0' in gcnae_rk.columns else 'N/A',
                'R@k-T3': f"{gcnae_rk['3.0'].mean():.4f} ± {gcnae_rk['3.0'].std():.4f}" if '3.0' in gcnae_rk.columns else 'N/A'
            })
            
            all_results.append({
                'Dataset': dataset_name,
                'Contamination': f'{cont_rate*100:.0f}%',
                'Model': 'TGCAE',
                'AUC-ROC': f"{tgcae_results['AUC ROC'].mean():.4f} ± {tgcae_results['AUC ROC'].std():.4f}",
                'AUC-PR': f"{tgcae_results['AUC Precision-Recall'].mean():.4f} ± {tgcae_results['AUC Precision-Recall'].std():.4f}",
                'F1': f"{tgcae_results['F1'].mean():.4f} ± {tgcae_results['F1'].std():.4f}",
                'Recall@k': f"{tgcae_results['Recall @ k'].mean():.4f} ± {tgcae_results['Recall @ k'].std():.4f}",
                'R@k-T1': f"{tgcae_rk['1.0'].mean():.4f} ± {tgcae_rk['1.0'].std():.4f}" if '1.0' in tgcae_rk.columns else 'N/A',
                'R@k-T2': f"{tgcae_rk['2.0'].mean():.4f} ± {tgcae_rk['2.0'].std():.4f}" if '2.0' in tgcae_rk.columns else 'N/A',
                'R@k-T3': f"{tgcae_rk['3.0'].mean():.4f} ± {tgcae_rk['3.0'].std():.4f}" if '3.0' in tgcae_rk.columns else 'N/A'
            })
            
            improvement = (tgcae_results['AUC ROC'].mean() - gcnae_results['AUC ROC'].mean()) / gcnae_results['AUC ROC'].mean() * 100
            print(f"\nTGCAE Improvement: {improvement:+.2f}%")
    
    # Display final comparison
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/gcnae_vs_tgcae_comparison.csv', index=False)
    print(f"\n{'='*80}")
    print("Results saved to: results/gcnae_vs_tgcae_comparison.csv")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()