#!/usr/bin/env python3
"""
Outlier Injection Script with Multiple Contamination Rates
Generate datasets with 0.2, 0.3, 0.4, 0.5 contamination rates
"""
import pandas as pd
import numpy as np
import random
import os
import glob
from functions.outlier_injection import *


def main():
    print("Starting outlier injection with multiple contamination rates...")

    # contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    contamination_rates = [0.05]


    for anom_per in contamination_rates:
        print(f"\n{'=' * 60}")
        print(f"Contamination Rate: {anom_per * 100:.0f}%")
        print(f"{'=' * 60}\n")

        # Set random seeds for reproducibility
        random.seed(123)
        np.random.seed(123)

        # BPIC 2017 outlier generation
        print(f"Processing BPIC 2017 with {anom_per * 100:.0f}% contamination...")
        bpi17_path = f'outliers/bpi17/outlier_{anom_per}prc/'

        # Check if files already exist
        if os.path.exists(bpi17_path) and len(glob.glob(f'{bpi17_path}*.csv')) > 0:
            print(f"Files already exist in {bpi17_path}, skipping...")
        else:
            bpi17_data = pd.read_csv('data-prepro/bpi17_prepro.csv')
            bpi17_dict = {
                'path': bpi17_path,
                'name': 'bpi17',
                'obj1': 'application',
                'obj2': 'offer',
                'obj3': None,
                'start': 5,
                'k': 500
            }
            dataset_generation(bpi17_data, bpi17_dict, iterations=5, anom_per=anom_per)
            print(f"  ✓ BPIC 2017 with {anom_per * 100:.0f}% contamination completed!")

        # DS2 outlier generation
        print(f"Processing DS2 with {anom_per * 100:.0f}% contamination...")
        ds2_path = f'outliers/ds2/outlier_{anom_per}prc/'

        # Check if files already exist
        if os.path.exists(ds2_path) and len(glob.glob(f'{ds2_path}*.csv')) > 0:
            print(f"Files already exist in {ds2_path}, skipping...")
        else:
            ds2_data = pd.read_csv('data-prepro/ds2_prepro.csv')
            ds2_dict = {
                'path': ds2_path,
                'name': 'ds2',
                'obj1': 'orders',
                'obj2': 'packages',
                'obj3': 'items',
                'start': 6,
                'k': 500
            }
            dataset_generation(ds2_data, ds2_dict, iterations=5, anom_per=anom_per)
            print(f"  ✓ DS2 with {anom_per * 100:.0f}% contamination completed!")


    print(f"\n{'=' * 60}")
    print("All outlier injection completed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()