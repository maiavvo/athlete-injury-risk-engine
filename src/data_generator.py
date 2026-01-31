"""
data_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class AthleteDataGenerator:
    """
    Generates synthetic training data for different athlete profiles.
    """
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_athlete_data(self, athlete_id, days=90, archetype='conservative'):
        """Generate training data for a single athlete."""
        
        params = self._get_archetype_parameters(archetype)
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Initialize data storage
        data = []
        
        for i, date in enumerate(dates):
            # Determine if athlete trains this day
            if np.random.random() < params['training_frequency']:
                
                duration = np.random.normal(
                    params['avg_duration'], 
                    params['duration_std']
                )
                duration = max(30, min(180, duration))
                
                intensity = np.random.normal(
                    params['avg_intensity'], 
                    params['intensity_std']
                )
                
                day_of_week = date.weekday()
                week_number = i // 7
                
                # Recovery week every 4 weeks
                if week_number % 4 == 3:
                    intensity *= 0.75
                    duration *= 0.80
                elif archetype == 'aggressive':
                    if np.random.random() < 0.15:
                        intensity *= 1.4
                        duration *= 1.3
                
                if day_of_week in [5, 6]:
                    if archetype in ['conservative', 'optimal']:
                        intensity *= 0.85
                    else:
                        intensity *= np.random.uniform(0.75, 1.1)
                elif day_of_week in [2, 3]:
                    intensity *= 1.20
                
                intensity = max(1, min(10, intensity))
                
                session_load = duration * intensity
                
                soreness = np.random.normal(
                    params['avg_soreness'], 
                    params['soreness_std']
                )
                
                if session_load > params['avg_duration'] * params['avg_intensity'] * 1.3:
                    soreness += np.random.uniform(1, 2)
                
                soreness = max(0, min(10, soreness))
                
            else:
                # Rest day
                duration = 0
                intensity = 0
                session_load = 0
                soreness = np.random.uniform(0, 2)
            
            data.append({
                'athlete_id': athlete_id,
                'date': date,
                'duration_minutes': round(duration, 1),
                'intensity_rpe': round(intensity, 1),
                'session_load': round(session_load, 1),
                'soreness': round(soreness, 1),
                'has_prior_injury': params['has_prior_injury']
            })
        
        return pd.DataFrame(data)
    
    def _get_archetype_parameters(self, archetype):
        """Define training parameters for each athlete archetype."""
        
        archetypes = {
            'conservative': {
                'training_frequency': 0.80,
                'avg_duration': 70,
                'duration_std': 20,
                'avg_intensity': 6.0,
                'intensity_std': 1.5,
                'avg_soreness': 3.5,
                'soreness_std': 2.0,
                'has_prior_injury': False
            },
            'aggressive': {
                'training_frequency': 0.90,
                'avg_duration': 100,
                'duration_std': 35,
                'avg_intensity': 8.0,
                'intensity_std': 2.0,
                'avg_soreness': 6.0,
                'soreness_std': 2.5,
                'has_prior_injury': False
            },
            'injury_prone': {
                'training_frequency': 0.75,
                'avg_duration': 75,
                'duration_std': 25,
                'avg_intensity': 6.5,
                'intensity_std': 2.0,
                'avg_soreness': 6.5,
                'soreness_std': 2.5,
                'has_prior_injury': True
            },
            'optimal': {
                'training_frequency': 0.82,
                'avg_duration': 80,
                'duration_std': 18,
                'avg_intensity': 7.0,
                'intensity_std': 1.3,
                'avg_soreness': 4.0,
                'soreness_std': 1.8,
                'has_prior_injury': False
            },
            'custom': {
                'training_frequency': 0.80,
                'avg_duration': 75,
                'duration_std': 20,
                'avg_intensity': 6.5,
                'intensity_std': 1.5,
                'avg_soreness': 4.0,
                'soreness_std': 2.0,
                'has_prior_injury': False
            }
        }
        
        if archetype not in archetypes:
            raise ValueError(f"Unknown archetype: {archetype}")
        
        return archetypes[archetype]
    
    def generate_cohort(self, num_athletes=20, days=90):
        """Generate data for multiple athletes with mixed archetypes."""
        
        all_data = []
        archetypes = ['conservative', 'aggressive', 'injury_prone', 'optimal']
        
        for i in range(num_athletes):
            archetype = archetypes[i % len(archetypes)]
            athlete_id = f"ATH_{i+1:03d}_{archetype[:3].upper()}"
            
            athlete_data = self.generate_athlete_data(
                athlete_id=athlete_id,
                days=days,
                archetype=archetype
            )
            
            all_data.append(athlete_data)
        
        return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    print("=" * 60)
    print("DATA GENERATOR TEST")
    print("=" * 60)
    
    generator = AthleteDataGenerator(seed=42)
    
    # Test each archetype
    for archetype in ['conservative', 'aggressive', 'injury_prone', 'optimal', 'custom']:
        print(f"\n{archetype.upper()}:")
        data = generator.generate_athlete_data(f"TEST_{archetype}", days=30, archetype=archetype)
        print(f"  Sessions: {(data['session_load'] > 0).sum()}")
        print(f"  Avg load: {data[data['session_load'] > 0]['session_load'].mean():.1f}")
    
    print("\nAll archetypes working!")
