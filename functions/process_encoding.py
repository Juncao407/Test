def exec_encoding(feature_storage, dataframe):
    import torch
    from torch_geometric import data
    import pandas as pd
    import numpy as np
    import pm4py
    import ocpa

    dataframe = dataframe.set_index('event_id')
    table = dataframe.copy()

    graphs_list = []

    for g_idx in list(range(0, len(feature_storage.feature_graphs))):
        g = g_idx + 1

        nodes = [int(n.event_id) for n in feature_storage.feature_graphs[g_idx].nodes]
        node_ids = pd.DataFrame(data=nodes, index=nodes).index
        attributes = dataframe.loc[node_ids, :]
        attributes.loc[node_ids, 'elapsed_time'] = [int(n.attributes[('event_elapsed_time', ())]) for n in
                                                    feature_storage.feature_graphs[g_idx].nodes]

        table.loc[node_ids, 'elapsed_time'] = attributes.loc[node_ids, 'elapsed_time']
        table.loc[node_ids, 'exec_id'] = g

        source = [int(e.source) for e in feature_storage.feature_graphs[g_idx].edges]
        target = [int(e.target) for e in feature_storage.feature_graphs[g_idx].edges]
        edges = pd.DataFrame(data={'source': source, 'target': target})
        edges_ids = edges.index

        edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)

        map_dict = {v.item(): i for i, v in enumerate(torch.unique(edge_idx))}
        map_edge = torch.zeros_like(edge_idx)
        for k, v in map_dict.items():
            map_edge[edge_idx == k] = v

        elap_column = attributes.pop('elapsed_time')
        attributes.insert(0, 'elapsed_time', elap_column)

        # Convert DataFrame to numeric, handling non-numeric data
        attributes_numeric = attributes.select_dtypes(include=[np.number])
        # Fill NaN values with 0
        attributes_numeric = attributes_numeric.fillna(0)
        attrs = torch.tensor(attributes_numeric.to_numpy(), dtype=torch.float)

        edge_idx = map_edge.long()

        globals()['graph_%s' % g] = data.Data(x=attrs, edge_index=edge_idx)
        graphs_list.append(globals()['graph_%s' % g])

    y_column = table.pop('y')
    table['y'] = y_column
    exec_column = table.pop('exec_id')
    table.insert(0, 'exec_id', exec_column)
    elap_column = table.pop('elapsed_time')
    table.insert(1, 'elapsed_time', elap_column)

    return graphs_list, table


def trainfiles_generation(parameters, in_path, graph_path, tab_path):
    import warnings
    warnings.simplefilter(action='ignore')

    import pandas as pd
    import numpy as np
    import pm4py
    import ocpa

    import os
    import pickle
    from ocpa.objects.log.importer.csv import factory as ocel_import_factory
    from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
    from tqdm import tqdm

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    if not os.path.exists(tab_path):
        os.makedirs(tab_path)

    for filename in tqdm(os.listdir(in_path)):
        file_path = in_path + filename

        # Read CSV and handle timestamp format issues before passing to ocpa
        df = pd.read_csv(file_path)
        
        # Fix timestamp format issue for all datasets first
        time_col = parameters['time_name']
        # Convert to datetime and then to string in the format that ocpa expects
        df[time_col] = pd.to_datetime(df[time_col], format='mixed')
        # Convert back to string without microseconds
        df[time_col] = df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Process all object columns to ensure proper format for OCPA
        # Handle 'Value X' patterns and ensure all values are in list format
        for col in df.columns:
            if col != time_col and col not in parameters['obj_names'] and df[col].dtype == 'object':  # Only process string/object columns, excluding time column and object columns
                # Replace 'Value X' patterns with just the numeric value X
                df[col] = df[col].apply(lambda x: x.split(' ')[1] if isinstance(x, str) and 'Value ' in str(x) else x)
                # Convert to numeric, setting errors='coerce' to handle non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Convert all values to proper list format for OCPA compatibility
                df[col] = df[col].apply(lambda x: '[]' if pd.isna(x) else 
                                       str([int(x)]) if isinstance(x, (int, float)) and x == int(x) and not pd.isna(x) else 
                                       str([x]) if isinstance(x, (int, float)) and not pd.isna(x) else 
                                       str([x]) if isinstance(x, str) else 
                                       str([x]))
                # Ensure column is of object type to prevent pandas from changing the format
                df[col] = df[col].astype('object')
        
        # Also handle columns that are in obj_names (like product, resource, case_id.1, etc.) to ensure they are in proper list format
        for col in parameters['obj_names']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str([str(x)]) if pd.notna(x) else '[]')
                df[col] = df[col].astype('object')
        
        import csv
        # Save the fixed CSV to a temporary file with all non-numeric values quoted
        temp_file_path = file_path + '_temp.csv'
        df.to_csv(temp_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        try:
            ocel_log = ocel_import_factory.apply(file_path=temp_file_path, parameters=parameters)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        # Load eventlog and create process executions graphs
        ocel_activities = list(set(ocel_log.log.log["event_activity"].tolist()))
        feature_set = [(predictive_monitoring.EVENT_ELAPSED_TIME, ())]

        ocel_feature_storage = predictive_monitoring.apply(ocel_log, feature_set, [])

        features = df.drop(columns=parameters['obj_names'])
        # Drop columns that are no longer needed, but only if they exist
        columns_to_drop = ['old_event_id', 'event_activity', 'event_timestamp']
        existing_columns_to_drop = [col for col in columns_to_drop if col in features.columns]
        features = features.drop(columns=existing_columns_to_drop)

        glist, tab = exec_encoding(ocel_feature_storage, features)

        graph_name = graph_path + 'graph_' + filename.replace('csv', 'sav')
        pickle.dump(glist, open(graph_name, 'wb'))

        tab_name = tab_path + 'table_' + filename
        tab.to_csv(tab_name, index=True, index_label='event_id')