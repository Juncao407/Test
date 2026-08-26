#!/usr/bin/env python3
"""
Data Preprocessing Script
Converts notebook 0_data-preprocessing.ipynb to Python script
"""

import pandas as pd
import numpy as np
import os
from functions.data_preprocessing import *

def main():
    print("Starting data preprocessing...")
    
    # Create output directory
    directory = 'data-prepro'
    os.makedirs(directory, exist_ok=True)
    
    # BPIC 2017 Preprocessing
    # print("Processing BPIC 2017 dataset...")
    # bpi17 = pd.read_csv('ocel/BPIC17.csv', sep=',')
    #
    # # Drop unneeded columns
    # bpi17 = bpi17.drop(columns=['event_None', 'event_Unnamed: 0', 'event_start_timestamp', 'event_EventID','event_CaseID'])
    #
    # # Apply preprocessing
    # objects = ['application', 'offer']
    # first_cols = ['event_id', 'event_timestamp', 'event_activity'] + objects
    # split = []
    #
    # bpi17 = preprocess_dataframe(bpi17, first_cols, split)
    #
    # # Save preprocessed data
    # filename = directory + '/bpi17_prepro.csv'
    # bpi17.to_csv(filename, index=False)
    # print(f"BPIC 2017 preprocessed data saved. Shape: {bpi17.shape}")
    #
    # # DS2 Preprocessing
    # print("Processing DS2 dataset...")
    # ds2 = pd.read_csv('ocel/DS2.csv', sep=',')
    #
    # # Rename columns exactly as in the ipynb
    # ds2.columns = ds2.columns.str.replace('ocel:', '')
    # ds2.columns = ds2.columns.str.replace('type:', '')
    # ds2 = ds2.rename(columns={'timestamp': 'event_timestamp', 'activity': 'event_activity'})
    #
    # # Apply preprocessing for DS2
    # objects = ['orders', 'packages', 'items']
    # first_cols = ['event_id', 'event_timestamp', 'event_activity'] + objects
    # split = []
    #
    # ds2 = preprocess_dataframe(ds2, first_cols, split)
    #
    # # Save preprocessed DS2 data
    # filename = directory + '/ds2_prepro.csv'
    # ds2.to_csv(filename, index=False)
    # print(f"DS2 preprocessed data saved. Shape: {ds2.shape}")
    
    # BPI 2013 Preprocessing
    print("Processing BPI 2013 dataset...")
    bpi_2013 = pd.read_csv('ocel/bpi_2013.csv', sep=',')
    
    # Rename columns to match the expected format
    bpi_2013 = bpi_2013.rename(columns={
        'Activity': 'event_activity',
        'Complete Timestamp': 'event_timestamp',
        'Case ID': 'case_id'
    })
    
    # Process Resource and Case ID columns to convert 'Value X' and 'Case X' formats to 'value_X' and 'case_X'
    # Process Resource column
    if 'Resource' in bpi_2013.columns:
        bpi_2013['Resource'] = bpi_2013['Resource'].apply(
            lambda x: f"value_{x.split()[1]}" if isinstance(x, str) and x.startswith('Value ') else x
        )
    
    # Process case_id column (originally 'Case ID')
    if 'case_id' in bpi_2013.columns:
        bpi_2013['case_id'] = bpi_2013['case_id'].apply(
            lambda x: f"case_{x.split()[1]}" if isinstance(x, str) and x.startswith('Case ') else x
        )
    
    # Apply preprocessing for BPI 2013
    # For BPI 2013, we'll use case_id as the main object
    objects = ['case_id', 'Resource', 'product']
    first_cols = ['case_id', 'event_timestamp', 'event_activity'] + objects
    split = ['Variant', 'impact', 'lifecycle:transition', 'org:group', 'org:role', 'organization country', 'organization involved', 'resource country']
    
    # Ensure all columns in first_cols and split exist in the dataframe
    existing_first_cols = [col for col in first_cols if col in bpi_2013.columns]
    existing_split_cols = [col for col in split if col in bpi_2013.columns]
    
    bpi_2013 = preprocess_dataframe(bpi_2013, existing_first_cols, existing_split_cols)
    
    # Save preprocessed BPI 2013 data
    filename = directory + '/bpi_2013_prepro.csv'
    bpi_2013.to_csv(filename, index=False)
    print(f"BPI 2013 preprocessed data saved. Shape: {bpi_2013.shape}")
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()