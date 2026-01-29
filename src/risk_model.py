"""
Multi-factor injury risk scoring model.
Combines ACWR, load spikes, soreness, and injury history.
"""

import pandas as pd
import numpy as np


class InjuryRiskScorer:
    
    def __init__(self):
        self.weights = {
            'acwr_risk': 0.35,
            'load_spike': 0.25,
            'soreness': 0.20,
            'injury_history': 0.20
        }
        
        self.acwr_thresholds = {
            'safe_low': 0.85,
            'safe_high': 1.25,
            'moderate': 1.40,
            'high': 1.40
        }
        
        self.spike_threshold = 0.10
    
    def score_acwr_risk(self, acwr):
        """Convert ACWR to risk score (0-100)."""
        
        if pd.isna(acwr):
            return 0
        
        if acwr < self.acwr_thresholds['safe_low']:
            detraining_amount = (self.acwr_thresholds['safe_low'] - acwr) / 0.15
            return 25 + (detraining_amount * 15)
        
        elif acwr <= self.acwr_thresholds['safe_high']:
            position = (acwr - self.acwr_thresholds['safe_low']) / (self.acwr_thresholds['safe_high'] - self.acwr_thresholds['safe_low'])
            return position * 10
        
        elif acwr <= self.acwr_thresholds['moderate']:
            range_size = self.acwr_thresholds['moderate'] - self.acwr_thresholds['safe_high']
            position = (acwr - self.acwr_thresholds['safe_high']) / range_size
            return 10 + (position * 40)
        
        else:
            excess = acwr - self.acwr_thresholds['high']
            score = 50 + (excess * 150)
            return min(score, 100)
    
    def score_load_spike(self, load_change_pct, spike_magnitude):
        """Score risk from load increases."""
        
        if pd.isna(load_change_pct):
            return 0
        
        if load_change_pct <= self.spike_threshold:
            return 0
        
        excess = load_change_pct - self.spike_threshold
        score = (excess / 0.10) * 40
        return min(score, 100)
    
    def score_soreness(self, soreness_value):
        """Convert soreness to risk score."""
        
        if pd.isna(soreness_value):
            return 0
        
        if soreness_value <= 2:
            return (soreness_value / 2) * 15
        elif soreness_value <= 5:
            position = (soreness_value - 2) / 3
            return 15 + (position * 30)
        else:
            position = (soreness_value - 5) / 5
            return 45 + (position * 55)
    
    def score_injury_history(self, has_prior_injury):
        """Score from injury history."""
        return 80 if has_prior_injury else 0
    
    def calculate_risk_score(self, athlete_row):
        """Calculate overall risk score."""
        
        acwr_score = self.score_acwr_risk(athlete_row.get('acwr', np.nan))
        spike_score = self.score_load_spike(
            athlete_row.get('load_change_pct', np.nan),
            athlete_row.get('spike_magnitude', np.nan)
        )
        soreness_score = self.score_soreness(athlete_row.get('soreness', np.nan))
        injury_history_score = self.score_injury_history(
            athlete_row.get('has_prior_injury', False)
        )
        
        weighted_scores = {
            'acwr': acwr_score * self.weights['acwr_risk'],
            'spike': spike_score * self.weights['load_spike'],
            'soreness': soreness_score * self.weights['soreness'],
            'history': injury_history_score * self.weights['injury_history']
        }
        
        total_score = sum(weighted_scores.values())
        
        if total_score < 25:
            category = 'Low'
        elif total_score < 50:
            category = 'Moderate'
        else:
            category = 'High'
        
        recommendations = self._generate_recommendations(
            athlete_row, acwr_score, spike_score, soreness_score, 
            injury_history_score, category
        )
        
        return {
            'risk_score': round(total_score, 1),
            'risk_category': category,
            'factor_scores': {
                'acwr_score': round(acwr_score, 1),
                'spike_score': round(spike_score, 1),
                'soreness_score': round(soreness_score, 1),
                'history_score': round(injury_history_score, 1)
            },
            'weighted_contributions': {k: round(v, 1) for k, v in weighted_scores.items()},
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, row, acwr_score, spike_score, 
                                  soreness_score, history_score, category):
        """Generate actionable recommendations."""
        
        recommendations = []
        acwr = row.get('acwr', np.nan)
        
        if not pd.isna(acwr):
            if acwr > 1.4:
                reduction = int((acwr - 1.25) / acwr * 100)
                recommendations.append(
                    f"⚠️ ACWR elevated at {acwr:.2f} - Reduce load by {reduction}%"
                )
            elif acwr > 1.25:
                recommendations.append(
                    f"⚠️ ACWR of {acwr:.2f} approaching caution zone"
                )
            elif acwr < 0.85:
                recommendations.append(
                    f"⚠️ ACWR of {acwr:.2f} suggests detraining - Increase load 8-12%"
                )
        
        if spike_score > 20:
            recommendations.append(
                "⚠️ Significant load spike - Maintain current volume 1-2 weeks"
            )
        
        soreness = row.get('soreness', 0)
        if soreness > 5:
            recommendations.append(
                f"⚠️ Elevated soreness ({soreness:.1f}/10) - Prioritize recovery"
            )
        elif soreness > 3:
            recommendations.append(
                f"ℹ️ Moderate soreness ({soreness:.1f}/10) - Monitor closely"
            )
        
        if history_score > 0:
            recommendations.append(
                "⚠️ Prior injury history - Maintain conservative progression"
            )
        
        if category == 'High':
            recommendations.append(
                "🚨 HIGH RISK - Consider medical consultation"
            )
        elif category == 'Moderate':
            recommendations.append(
                "⚠️ MODERATE RISK - Avoid load spikes until risk normalizes"
            )
        else:
            recommendations.append(
                "✅ LOW RISK - Continue with standard monitoring"
            )
        
        return recommendations
    
    def score_all_athletes(self, metrics_data):
        """Calculate risk for all athletes."""
        
        df = metrics_data.copy()
        df['risk_score'] = 0.0
        df['risk_category'] = 'Unknown'
        df['acwr_component'] = 0.0
        df['spike_component'] = 0.0
        df['soreness_component'] = 0.0
        df['history_component'] = 0.0
        
        for idx, row in df.iterrows():
            risk_assessment = self.calculate_risk_score(row)
            df.at[idx, 'risk_score'] = risk_assessment['risk_score']
            df.at[idx, 'risk_category'] = risk_assessment['risk_category']
            df.at[idx, 'acwr_component'] = risk_assessment['weighted_contributions']['acwr']
            df.at[idx, 'spike_component'] = risk_assessment['weighted_contributions']['spike']
            df.at[idx, 'soreness_component'] = risk_assessment['weighted_contributions']['soreness']
            df.at[idx, 'history_component'] = risk_assessment['weighted_contributions']['history']
        
        return df


if __name__ == "__main__":
    print("Loading metrics data...")
    metrics_data = pd.read_csv('../data/athlete_metrics.csv')
    metrics_data['date'] = pd.to_datetime(metrics_data['date'])
    
    scorer = InjuryRiskScorer()
    risk_data = scorer.score_all_athletes(metrics_data)
    
    print(f"\nCohort Summary:")
    print(f"  High risk days: {(risk_data['risk_category'] == 'High').sum()}")
    print(f"  Moderate risk days: {(risk_data['risk_category'] == 'Moderate').sum()}")
    print(f"  Low risk days: {(risk_data['risk_category'] == 'Low').sum()}")
    print(f"  Avg risk score: {risk_data['risk_score'].mean():.1f}")
    
    risk_data.to_csv('../data/athlete_risk_scores.csv', index=False)
    print("\nSaved to ../data/athlete_risk_scores.csv")
