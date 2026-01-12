# -*- coding: utf-8 -*-
"""
Energy Anomaly Detection Dataset Loader for TAnoGAN
Adapted from nab_dataset.py to work with energy-anomaly-detection competition data
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing

class EnergyDataset(Dataset):
    def __init__(self, data_settings):
        """
        Args:
            data_settings (object): settings for loading data and preprocessing
        """
        self.train = data_settings.train
        self.window_length = data_settings.window_length
        
        df_x, df_y = self.read_data(
            data_file=data_settings.data_file,
            group_by_building=data_settings.group_by_building
        )
        
        # Select and standardize data
        df_x = df_x[['meter_reading']]
        df_x = self.normalize(df_x)
        df_x.columns = ['value']
        
        # Important parameters
        if data_settings.train:
            self.stride = 1
        else:
            self.stride = self.window_length
        
        self.n_feature = len(df_x.columns)
        
        # x, y data
        x = df_x
        y = df_y
        
        # Adapt the datasets for the sequence data shape
        x, y = self.unroll(x, y)
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y])).float()
        
        self.data_len = x.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # Create sequences 
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            un_data.append(data.iloc[idx:idx+seq_len].values)
            un_labels.append(labels.iloc[idx:idx+seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    def read_data(self, data_file=None, group_by_building=False):
        """
        Read energy anomaly detection data from CSV file
        
        Args:
            data_file: Path to CSV file (train.csv or test.csv)
            group_by_building: If True, process each building separately and concatenate
                             If False, process all data together sorted by timestamp
        """
        df = pd.read_csv(data_file)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle missing values in meter_reading
        # Forward fill within each building, then backward fill
        if group_by_building:
            df = df.sort_values(['building_id', 'timestamp'])
            df['meter_reading'] = df.groupby('building_id')['meter_reading'].ffill()
            df['meter_reading'] = df.groupby('building_id')['meter_reading'].bfill()
        else:
            df = df.sort_values('timestamp')
            df['meter_reading'] = df['meter_reading'].ffill()
            df['meter_reading'] = df['meter_reading'].bfill()
        
        # If still missing, fill with 0
        df['meter_reading'] = df['meter_reading'].fillna(0)
        
        # Sort by timestamp for sequence processing
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract features and labels
        if 'anomaly' in df.columns:
            # Training data has anomaly labels
            df_x = df[['timestamp', 'meter_reading']].copy()
            df_y = df[['anomaly']].copy()
        else:
            # Test data doesn't have anomaly labels (set to 0 for now)
            df_x = df[['timestamp', 'meter_reading']].copy()
            df_y = pd.DataFrame(np.zeros(len(df)), columns=['anomaly'])
        
        # Remove timestamp column (not needed for model)
        df_x = df_x[['meter_reading']]
        
        return df_x, df_y
    
    def normalize(self, df_x=None):
        """Normalize data using StandardScaler"""
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(df_x)
        df_x = pd.DataFrame(np_scaled)
        return df_x


# Settings for data loader
class EnergyDataSettings:
    def __init__(self):
        self.data_file = 'data/train.csv'  # Path to train.csv or test.csv
        self.train = True  # True for training, False for testing
        self.window_length = 60  # Sequence length
        self.group_by_building = False  # If True, process each building separately
