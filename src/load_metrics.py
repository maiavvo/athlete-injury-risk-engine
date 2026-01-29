"""
Calculate training load metrics (ACWR, monotony, load spikes).
Based on Gabbett 2016 and Hulin 2016 research.
"""

import pandas as pd
import numpy as np
import os
import sys

# Get the project root directory
if __name__ == "__main__":
    # Running directly
    DATA_DIR = '../data'
else:
    # Running from gunicorn (from root)
    DATA_DIR = 'data'


class LoadMetricsCalculator:
    
    def __init__(self):
        self.acute_window = 7
        self.chronic_window = 28
    
    def calculate_rolling_loads(self, athlete_data):
        """Calculate acute and chronic training loads."""
        
        df = athlete_data.sort_values('date').copy()
        
        # 7-day rolling sum
        df['acute_load'] = df['session_load'].rolling(
            window=self.acute_window, 
            min_periods=self.acute_window
        ).sum()
        
        # 28-day rolling average (scaled to match acute timeframe)
        df['chronic_load'] = df['session_load'].rolling(
            window=self.chronic_window,
            min_periods=self.chronic_window
        ).mean() * self.acute_window
        
        # ACWR calculation
        df['acwr'] = np.where(
            df['chronic_load'] > 0,
            df['acute_load'] / df['chronic_load'],
            np.nan
        )
        
        return df
    
    def calculate_weekly_metrics(self, athlete_data):
        """Calculate training monotony and strain."""
        
        df = athlete_data.copy()
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        
        weekly_stats = df.groupby(['athlete_id', 'year', 'week']).agg({
            'session_load': ['sum', 'mean', 'std']
        }).reset_index()
        
        weekly_stats.columns = ['athlete_id', 'year', 'week', 
                                'weekly_total_load', 'weekly_mean_load', 
                                'weekly_std_load']
        
        # Monotony = mean / std
        weekly_stats['training_monotony'] = np.where(
            weekly_stats['weekly_std_load'] > 0,
            weekly_stats['weekly_mean_load'] / weekly_stats['weekly_std_load'],
            0
        )
        
        weekly_stats['training_strain'] = (
            weekly_stats['weekly_total_load'] * 
            weekly_stats['training_monotony']
        )
        
        df = df.merge(
            weekly_stats[['athlete_id', 'year', 'week', 
                         'training_monotony', 'training_strain']],
            on=['athlete_id', 'year', 'week'],
            how='left'
        )
        
        return df
    
    def detect_load_spikes(self, athlete_data, threshold=0.15):
        """Flag sudden increases in training load."""
        
        df = athlete_data.copy()
        df['previous_acute_load'] = df['acute_load'].shift(7)
        
        df['load_change_pct'] = np.where(
            df['previous_acute_load'] > 0,
            (df['acute_load'] - df['previous_acute_load']) / df['previous_acute_load'],
            0
        )
        
        df['load_spike'] = df['load_change_pct'] > threshold
        df['spike_magnitude'] = np.where(df['load_spike'], df['load_change_pct'], 0)
        
        return df
    
    def calculate_all_metrics(self, athlete_data):
        """Run all calculations."""
        
        df = athlete_data.copy()
        df = self.calculate_rolling_loads(df)
        df = self.calculate_weekly_metrics(df)
        df = self.detect_load_spikes(df)
        
        return df


if __name__ == "__main__":
    print("Loading data...")
    data = pd.read_csv('data/sample_athlete_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    calculator = LoadMetricsCalculator()
    processed_data = []
    
    for athlete_id in data['athlete_id'].unique():
        athlete_data = data[data['athlete_id'] == athlete_id].copy()
        athlete_metrics = calculator.calculate_all_metrics(athlete_data)
        processed_data.append(athlete_metrics)
    
    all_metrics = pd.concat(processed_data, ignore_index=True)
    all_metrics.to_csv('data/athlete_metrics.csv', index=False)
    
    print(f"Processed {data['athlete_id'].nunique()} athletes")
    print(f"Records with valid ACWR: {all_metrics['acwr'].notna().sum()}")
    print("Saved to data/athlete_metrics.csv")
