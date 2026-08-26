#!/usr/bin/env python3
"""
Process Encoding Script for Multiple Contamination Rates
"""
import pandas as pd
import numpy as np
import os
import glob
from functions.process_encoding import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    print("Starting process encoding for multiple contamination rates...")

    # contamination_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    contamination_rates = [0.05]
    # BPIC 2017 parameters
    bpi17_object_types = ['application', 'offer']
    bpi17_parameters = {
        'obj_names': bpi17_object_types,
        'val_names': [],
        'act_name': 'event_activity',
        'time_name': 'event_timestamp',
        'sep': ','
    }

    # DS2 parameters
    ds2_object_types = ['orders', 'packages', 'items']
    ds2_parameters = {
        'obj_names': ds2_object_types,
        'val_names': [],
        'act_name': 'event_activity',
        'time_name': 'event_timestamp',
        'sep': ','
    }

    for anom_per in contamination_rates:
        print(f"\n{'=' * 60}")
        print(f"Contamination Rate: {anom_per * 100:.0f}%")
        print(f"{'=' * 60}\n")

        # BPIC 2017 encoding
        print(f"Encoding BPIC 2017 with {anom_per * 100:.0f}% contamination...")
        bpi17_in_path = f'outliers/bpi17/outlier_{anom_per}prc/'
        bpi17_graph_path = f'process_graphs/bpi17/outlier_{anom_per}prc/'
        bpi17_tab_path = f'process_tables/bpi17/outlier_{anom_per}prc/'

        # Check if files already exist
        if os.path.exists(bpi17_graph_path) and len(glob.glob(f'{bpi17_graph_path}*.sav')) > 0:
            print(f"  ⚠️  Files already exist in {bpi17_graph_path}, skipping...")
        else:
            trainfiles_generation(
                parameters=bpi17_parameters,
                in_path=bpi17_in_path,
                graph_path=bpi17_graph_path,
                tab_path=bpi17_tab_path
            )
            print(f"  ✓ BPIC 2017 with {anom_per * 100:.0f}% contamination encoded!")

        # DS2 encoding
        print(f"Encoding DS2 with {anom_per * 100:.0f}% contamination...")
        ds2_in_path = f'outliers/ds2/outlier_{anom_per}prc/'
        ds2_graph_path = f'process_graphs/ds2/outlier_{anom_per}prc/'
        ds2_tab_path = f'process_tables/ds2/outlier_{anom_per}prc/'

        # Check if files already exist
        if os.path.exists(ds2_graph_path) and len(glob.glob(f'{ds2_graph_path}*.sav')) > 0:
            print(f"  ⚠️  Files already exist in {ds2_graph_path}, skipping...")
        else:
            trainfiles_generation(
                parameters=ds2_parameters,
                in_path=ds2_in_path,
                graph_path=ds2_graph_path,
                tab_path=ds2_tab_path
            )
            print(f"  ✓ DS2 with {anom_per * 100:.0f}% contamination encoded!")


    print(f"\n{'=' * 60}")
    print("All process encoding completed successfully!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()