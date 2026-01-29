"""
visualization.py

Creates professional, interactive visualizations for injury risk analysis.

Visualizations:
1. Risk Timeline - Shows how risk evolves over time for an athlete
2. ACWR Trend Chart - Training load ratios with risk zones
3. Risk Factor Breakdown - Stacked bar showing what drives risk
4. Cohort Risk Heatmap - Calendar view of risk across all athletes
5. Load vs Risk Scatter - Relationship between training load and injury risk

Uses Plotly for interactive charts that can be embedded in reports.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import sys

# Get the project root directory
if __name__ == "__main__":
    # Running directly
    DATA_DIR = '../data'
else:
    # Running from gunicorn (from root)
    DATA_DIR = 'data'


class RiskVisualizer:
    """
    Creates publication-quality visualizations for sports medicine analytics.
    """
    
    def __init__(self):
        """Initialize with consistent color scheme."""
        self.colors = {
            'high_risk': '#DC3545',      # Red
            'moderate_risk': '#FFC107',  # Amber
            'low_risk': '#28A745',       # Green
            'acute': '#007BFF',          # Blue
            'chronic': '#6C757D',        # Gray
            'soreness': '#E83E8C'        # Pink
        }
    
    def plot_risk_timeline(self, athlete_data, athlete_id):
        """
        Create multi-axis timeline showing risk evolution.
        
        Shows:
        - Daily training load (bars)
        - Acute load (line)
        - Chronic load (line)
        - ACWR with risk zones (line + shaded regions)
        - Risk score (background color)
        
        Args:
            athlete_data (pd.DataFrame): Single athlete's data with all metrics
            athlete_id (str): Athlete identifier for title
        
        Returns:
            plotly.graph_objects.Figure
        """
        
        # Filter to rows with valid ACWR
        data = athlete_data[athlete_data['acwr'].notna()].copy()
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=(
                f'{athlete_id} - Training Load & ACWR',
                'Injury Risk Score'
            ),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}]]
        )
        
        # Top plot: Training loads and ACWR
        # Add session load bars
        fig.add_trace(
            go.Bar(
                x=data['date'],
                y=data['session_load'],
                name='Session Load',
                marker_color='lightblue',
                opacity=0.5
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add acute load line
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['acute_load'],
                name='Acute Load (7d)',
                line=dict(color=self.colors['acute'], width=2),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add chronic load line
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['chronic_load'],
                name='Chronic Load (28d)',
                line=dict(color=self.colors['chronic'], width=2, dash='dash'),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add ACWR on secondary axis
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['acwr'],
                name='ACWR',
                line=dict(color='purple', width=3),
                mode='lines+markers'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Add ACWR risk zones (horizontal lines on secondary axis)
        fig.add_hline(
            y=1.5, line_dash="dash", line_color=self.colors['high_risk'],
            annotation_text="High Risk (1.5)",
            row=1, col=1, secondary_y=True
        )
        fig.add_hline(
            y=1.3, line_dash="dot", line_color=self.colors['moderate_risk'],
            annotation_text="Caution (1.3)",
            row=1, col=1, secondary_y=True
        )
        fig.add_hline(
            y=0.8, line_dash="dot", line_color=self.colors['moderate_risk'],
            annotation_text="Detraining (0.8)",
            row=1, col=1, secondary_y=True
        )
        
        # Bottom plot: Risk score with color-coded background
        # Create colored areas based on risk category
        for category, color in [('High', self.colors['high_risk']), 
                                ('Moderate', self.colors['moderate_risk']), 
                                ('Low', self.colors['low_risk'])]:
            category_data = data[data['risk_category'] == category]
            if len(category_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=category_data['date'],
                        y=category_data['risk_score'],
                        name=f'{category} Risk',
                        line=dict(color=color, width=0),
                        fill='tozeroy',
                        fillcolor=color,
                        opacity=0.6,
                        mode='none'
                    ),
                    row=2, col=1
                )
        
        # Add risk score line on top
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['risk_score'],
                name='Risk Score',
                line=dict(color='black', width=2),
                mode='lines',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Training Load", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="ACWR", row=1, col=1, secondary_y=True, range=[0, 2.5])
        fig.update_yaxes(title_text="Risk Score (0-100)", row=2, col=1, range=[0, 100])
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Injury Risk Analysis: {athlete_id}",
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_factor_breakdown(self, athlete_data, athlete_id):
        """
        Stacked bar chart showing risk factor contributions over time.
        
        Args:
            athlete_data (pd.DataFrame): Single athlete's data
            athlete_id (str): Athlete identifier
        
        Returns:
            plotly.graph_objects.Figure
        """
        
        data = athlete_data[athlete_data['risk_score'] > 0].copy()
        data = data.sort_values('date')
        
        fig = go.Figure()
        
        # Add each risk component as a stacked bar
        fig.add_trace(go.Bar(
            x=data['date'],
            y=data['acwr_component'],
            name='ACWR Risk',
            marker_color='#8B4513'
        ))
        
        fig.add_trace(go.Bar(
            x=data['date'],
            y=data['spike_component'],
            name='Load Spike',
            marker_color='#FF6347'
        ))
        
        fig.add_trace(go.Bar(
            x=data['date'],
            y=data['soreness_component'],
            name='Soreness',
            marker_color='#FFD700'
        ))
        
        fig.add_trace(go.Bar(
            x=data['date'],
            y=data['history_component'],
            name='Injury History',
            marker_color='#4B0082'
        ))
        
        fig.update_layout(
            barmode='stack',
            title=f'{athlete_id} - Risk Factor Contributions',
            xaxis_title='Date',
            yaxis_title='Risk Score Contribution',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_cohort_heatmap(self, risk_data):
        """
        Calendar heatmap showing risk across all athletes.
        Only shows training days with valid risk calculations.
        
        Args:
            risk_data (pd.DataFrame): All athletes' risk data
        
        Returns:
            plotly.graph_objects.Figure
        """
        
        # Filter to only rows with valid risk scores (training days only)
        valid_data = risk_data[(risk_data['acwr'].notna()) & (risk_data['risk_score'] > 0)].copy()
        
        # Pivot data for heatmap
        heatmap_data = valid_data.pivot_table(
            values='risk_score',
            index='athlete_id',
            columns='date',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[
                [0, self.colors['low_risk']],
                [0.25, self.colors['low_risk']],
                [0.5, self.colors['moderate_risk']],
                [0.75, self.colors['high_risk']],
                [1, self.colors['high_risk']]
            ],
            colorbar=dict(title="Risk Score"),
            hovertemplate='Athlete: %{y}<br>Date: %{x}<br>Risk: %{z:.1f}<extra></extra>',
            zmin=0,
            zmax=100
        ))
        
        fig.update_layout(
            title='Cohort Injury Risk Heatmap - Training Days Only<br><sub>White/blank = rest days or insufficient data for ACWR</sub>',
            xaxis_title='Date',
            yaxis_title='Athlete ID',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def plot_load_vs_risk_scatter(self, risk_data):
        """
        Scatter plot showing relationship between ACWR and risk score.
        
        Args:
            risk_data (pd.DataFrame): All athletes' data
        
        Returns:
            plotly.graph_objects.Figure
        """
        
        data = risk_data[risk_data['acwr'].notna()].copy()
        
        fig = px.scatter(
            data,
            x='acwr',
            y='risk_score',
            color='risk_category',
            color_discrete_map={
                'Low': self.colors['low_risk'],
                'Moderate': self.colors['moderate_risk'],
                'High': self.colors['high_risk']
            },
            hover_data=['athlete_id', 'date', 'soreness'],
            title='ACWR vs Risk Score (All Athletes)',
            labels={'acwr': 'ACWR', 'risk_score': 'Risk Score'}
        )
        
        # Add vertical lines for ACWR thresholds
        fig.add_vline(x=0.8, line_dash="dash", line_color="gray", 
                      annotation_text="Detraining")
        fig.add_vline(x=1.3, line_dash="dash", line_color="orange",
                      annotation_text="Caution")
        fig.add_vline(x=1.5, line_dash="dash", line_color="red",
                      annotation_text="High Risk")
        
        fig.update_layout(
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_athlete_report(self, athlete_data, athlete_id, output_path=None):
        """
        Generate comprehensive visual report for a single athlete.
        
        Args:
            athlete_data (pd.DataFrame): Single athlete's data
            athlete_id (str): Athlete identifier
            output_path (str): Optional path to save HTML report
        
        Returns:
            None (displays/saves figures)
        """
        
        print(f"\n📊 Generating report for {athlete_id}...")
        
        # Create all visualizations
        timeline_fig = self.plot_risk_timeline(athlete_data, athlete_id)
        breakdown_fig = self.plot_factor_breakdown(athlete_data, athlete_id)
        
        # Display or save
        if output_path:
            timeline_fig.write_html(f"{output_path}/{athlete_id}_timeline.html")
            breakdown_fig.write_html(f"{output_path}/{athlete_id}_breakdown.html")
            print(f"   ✅ Saved visualizations to {output_path}")
        else:
            timeline_fig.show()
            breakdown_fig.show()


# Example usage and testing
if __name__ == "__main__":
    """
    Generate visualizations from risk-scored data.
    """
    
    print("=" * 70)
    print("SPORTS MEDICINE ANALYTICS - VISUALIZATION DASHBOARD")
    print("=" * 70)
    print()
    
    # Load risk data
    print("📂 Loading risk-scored data...")
    risk_data = pd.read_csv('data/athlete_risk_scores.csv')
    risk_data['date'] = pd.to_datetime(risk_data['date'])
    print(f"✅ Loaded {len(risk_data)} records for {risk_data['athlete_id'].nunique()} athletes")
    print()
    
    # Initialize visualizer
    visualizer = RiskVisualizer()
    
    # Create visualizations for first few athletes
    print("🎨 Creating visualizations...")
    print("-" * 70)
    
    athletes_to_plot = risk_data['athlete_id'].unique()[:4]
    
    for athlete_id in athletes_to_plot:
        athlete_data = risk_data[risk_data['athlete_id'] == athlete_id]
        
        # Generate timeline
        print(f"\n📈 {athlete_id} - Risk Timeline")
        timeline_fig = visualizer.plot_risk_timeline(athlete_data, athlete_id)
        timeline_fig.write_html(f'data/{athlete_id}_timeline.html')
        print(f"   ✅ Saved to data/{athlete_id}_timeline.html")
        
        # Generate factor breakdown
        print(f"📊 {athlete_id} - Factor Breakdown")
        breakdown_fig = visualizer.plot_factor_breakdown(athlete_data, athlete_id)
        breakdown_fig.write_html(f'data/{athlete_id}_breakdown.html')
        print(f"   ✅ Saved to data/{athlete_id}_breakdown.html")
    
    # Create cohort-wide visualizations
    print("\n📊 Creating cohort-wide visualizations...")
    
    # Heatmap
    print("   🔥 Cohort risk heatmap")
    heatmap_fig = visualizer.plot_cohort_heatmap(risk_data)
    heatmap_fig.write_html('data/cohort_heatmap.html')
    print("   ✅ Saved to data/cohort_heatmap.html")
    
    # Scatter plot
    print("   📉 ACWR vs Risk scatter")
    scatter_fig = visualizer.plot_load_vs_risk_scatter(risk_data)
    scatter_fig.write_html('data/acwr_vs_risk_scatter.html')
    print("   ✅ Saved to data/acwr_vs_risk_scatter.html")
    
    print("\n" + "-" * 70)
    print("\n✨ Visualization complete!")
    print("\n📁 Open the HTML files in your browser to view interactive charts:")
    print("   - Individual athlete timelines")
    print("   - Risk factor breakdowns")
    print("   - Cohort heatmap")
    print("   - ACWR vs Risk analysis")
    print("\n💡 These charts are fully interactive - hover, zoom, and pan!")
