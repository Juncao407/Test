#!/usr/bin/env python3
"""
GCN Autoencoder Training Script
Converts notebook 3_training-gcnae.ipynb to Python script
"""

import pandas as pd
import numpy as np
import torch
from functions.training import train_eval_gcnae

# Set random seeds for reproducibility
from functions.training import set_random_seeds
set_random_seeds(42)

def main():
    print("Starting GCN Autoencoder training...")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # BPIC 2017 training
    print("Training BPIC 2017 GCN Autoencoder...")
    bpi17_graph_path = 'process_graphs/bpi17/outlier_0.25prc/'
    bpi17_attr_dims = [41, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2, 3, 14, 145]
    
    # train_eval_gcnae(bpi17_graph_path, batch_size=0, attribute_dims=bpi17_attr_dims)
    print("BPIC 2017 GCN Autoencoder training completed!")
    
    # DS2 training
    print("Training DS2 GCN Autoencoder...")
    ds2_graph_path = 'process_graphs/ds2/outlier_0.1prc/'
    ds2_attr_dims = [21, 1, 1, 20, 17]
    
    train_eval_gcnae(ds2_graph_path, batch_size=0, attribute_dims=ds2_attr_dims)
    print("DS2 GCN Autoencoder training completed!")
    
    # # BPIC 2013 training
    # print("Training BPIC 2013 GCN Autoencoder...")
    # # The processed graphs should be in process_graphs/bpi13/ after running process encoding
    # # Since the original outlier files are named with 0.1prc (10%), we'll use a similar structure
    # bpi13_graph_path = 'process_graphs/bpi13/outlier_0.1prc/'  # This path should match the processed file structure
    # # Based on the structure of BPIC13 dataset from the CSV file, we estimate the attribute dimensions
    # # After processing, each sample will have multiple features
    # bpi13_attr_dims = [115]  # Estimated from the number of attribute columns in the dataset
    #
    # train_eval_gcnae(bpi13_graph_path, batch_size=0, attribute_dims=bpi13_attr_dims)
    # print("BPIC 2013 GCN Autoencoder training completed!")


if __name__ == "__main__":
    main()