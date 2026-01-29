"""
Interactive dashboard for athlete monitoring.
Real-time risk assessment with team overview and comparison features.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Get the project root directory
if __name__ == "__main__":
    # Running directly
    DATA_DIR = '../data'
else:
    # Running from gunicorn (from root)
    DATA_DIR = 'data'

COLORS = {
    'safe': '#51CF66',
    'caution': '#FFD43B',
    'warning': '#FF922B',
    'danger': '#FF6B6B',
    'acute': '#4ECDC4',
    'chronic': '#6C757D',
    'background': '#F8F9FA',
    'text': '#212529'
}

print("Loading data...")
risk_data = pd.read_csv('data/athlete_risk_scores.csv')
risk_data['date'] = pd.to_datetime(risk_data['date'])

min_date = risk_data['date'].min()
max_date = risk_data['date'].max()
date_range_days = (max_date - min_date).days

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Athlete Injury Risk Engine"
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("💪 Athlete Injury Risk Engine 💪", 
                style={'textAlign': 'center', 'color': COLORS['text'], 'marginBottom': 10}),
        html.P("Algorithm powered injury risk prediction and training load management system",
               style={'textAlign': 'center', 'color': COLORS['text'], 'marginBottom': 20})
    ]),
    
    dcc.Tabs(id='main-tabs', value='individual', children=[
        dcc.Tab(label='🎱 Individual Athlete', value='individual'),
        dcc.Tab(label='🏀 Team Overview', value='team'),
        dcc.Tab(label='🎥 Compare Athletes', value='compare')
    ]),
    
    html.Div(id='tab-content', style={'marginTop': 20})
    
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'})


def create_gauge(value, title, range_max, thresholds):
    """Create gauge chart for KPI display."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, range_max], 'tickwidth': 1},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': thresholds['steps'],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds.get('danger_threshold', range_max)
            }
        }
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=COLORS['background'],
        font={'size': 12}
    )
    
    return fig


def get_athlete_current_status(athlete_data):
    """Get current metrics for an athlete."""
    
    valid_data = athlete_data[athlete_data['acwr'].notna()]
    if len(valid_data) == 0:
        return None
    
    recent = valid_data.tail(7)
    latest = valid_data.iloc[-1]
    
    return {
        'athlete_id': athlete_data['athlete_id'].iloc[0],
        'avg_acwr': recent['acwr'].mean(),
        'current_risk': latest['risk_score'],
        'risk_category': latest['risk_category'],
        'avg_load': recent['acute_load'].mean(),
        'avg_soreness': recent['soreness'].mean(),
        'high_risk_days': (valid_data['risk_category'] == 'High').sum(),
        'archetype': athlete_data['athlete_id'].iloc[0].split('_')[-1]
    }


def create_timeline_figure(athlete_data, athlete_id, date_filter=None):
    """Create main timeline chart."""
    
    valid_timeline = athlete_data[athlete_data['acwr'].notna()].copy()
    
    if date_filter:
        start_date, end_date = date_filter
        valid_timeline = valid_timeline[
            (valid_timeline['date'] >= start_date) & 
            (valid_timeline['date'] <= end_date)
        ]
    
    if len(valid_timeline) == 0:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=(f'{athlete_id} - Training Load & ACWR', 'Risk Score Timeline'),
        vertical_spacing=0.25,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Top chart: Loads and ACWR
    fig.add_trace(
        go.Bar(x=valid_timeline['date'], y=valid_timeline['session_load'],
               name='Session Load', marker_color=COLORS['acute'], opacity=0.4),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=valid_timeline['date'], y=valid_timeline['acute_load'],
                   name='Acute Load (7d)', line=dict(color=COLORS['acute'], width=2.5)),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=valid_timeline['date'], y=valid_timeline['chronic_load'],
                   name='Chronic Load (28d)', line=dict(color=COLORS['chronic'], width=2.5, dash='dash')),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=valid_timeline['date'], y=valid_timeline['acwr'],
                   name='ACWR', line=dict(color='purple', width=3),
                   mode='lines+markers', marker=dict(size=4)),
        row=1, col=1, secondary_y=True
    )
    
    # ACWR zones
    fig.add_hrect(y0=0.85, y1=1.25, fillcolor=COLORS['safe'], opacity=0.1,
                  row=1, col=1, secondary_y=True)
    fig.add_hrect(y0=1.4, y1=2.0, fillcolor=COLORS['danger'], opacity=0.15,
                  row=1, col=1, secondary_y=True)
    
    fig.add_hline(y=1.4, line_dash="dash", line_color=COLORS['danger'], line_width=2,
                  row=1, col=1, secondary_y=True)
    
    # Bottom chart: Risk score
    for category, color in [('High', COLORS['danger']), ('Moderate', COLORS['caution']), ('Low', COLORS['safe'])]:
        cat_data = valid_timeline[valid_timeline['risk_category'] == category]
        if len(cat_data) > 0:
            fig.add_trace(
                go.Scatter(x=cat_data['date'], y=cat_data['risk_score'],
                           name=f'{category} Risk', fill='tozeroy',
                           fillcolor=color, line=dict(color=color, width=0),
                           opacity=0.6, mode='none'),
                row=2, col=1
            )
    
    fig.add_trace(
        go.Scatter(x=valid_timeline['date'], y=valid_timeline['risk_score'],
                   name='Risk Score', line=dict(color='black', width=2.5),
                   showlegend=False),
        row=2, col=1
    )
    
    fig.update_yaxes(title_text="Training Load", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ACWR", row=1, col=1, secondary_y=True, range=[0, 2.2])
    fig.update_yaxes(title_text="Risk Score", row=2, col=1, range=[0, 100])
    
    fig.update_xaxes(
        row=1, col=1,
        tickformat='%b %d',
        dtick=86400000*7,
        tickangle=0
    )
    
    fig.update_xaxes(
        title_text="Date",
        row=2, col=1,
        tickformat='%b %d',
        dtick=86400000*7,
        tickangle=0
    )
    
    fig.update_layout(
        height=700,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=70, t=100, b=80),
        font=dict(size=12)
    )
    
    return fig


@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'individual':
        return render_individual_tab()
    elif tab == 'team':
        return render_team_tab()
    elif tab == 'compare':
        return render_compare_tab()


def render_individual_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Select Athlete:", style={'fontSize': 16, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='athlete-selector',
                    options=[{'label': aid, 'value': aid} for aid in sorted(risk_data['athlete_id'].unique())],
                    value=risk_data['athlete_id'].iloc[0],
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 30}),
            
            html.Div([
                html.Label("Date Range:", style={'fontSize': 16, 'fontWeight': 'bold', 'marginRight': 10}),
                dcc.RangeSlider(
                    id='date-range-slider',
                    min=0,
                    max=date_range_days,
                    value=[0, date_range_days],
                    marks={i: (min_date + timedelta(days=i)).strftime('%b %d') 
                           for i in range(0, date_range_days+1, 20)},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={'display': 'inline-block', 'width': '500px', 'verticalAlign': 'top'}),
            
            html.Button('💼 Export Report', id='export-btn', n_clicks=0,
                       style={'marginLeft': 20, 'padding': '10px 20px', 'fontSize': 14,
                              'backgroundColor': COLORS['acute'], 'color': 'white',
                              'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'marginLeft': 20, 'marginBottom': 20}),
        
        html.Div(id='export-status', style={'marginLeft': 20, 'marginTop': 10, 'color': COLORS['safe']}),
        html.Div(id='kpi-gauges'),
        html.Div(id='quick-stats', style={'marginBottom': 20, 'marginLeft': 20, 'marginRight': 20}),
        dcc.Graph(id='main-timeline'),
        html.Div(id='recommendations-panel', style={'margin': '20px'})
    ])


def render_team_tab():
    return html.Div([
        html.H2("🏀 Team Risk Overview", style={'marginLeft': 20, 'color': COLORS['text']}),
        html.P("Current status for all athletes based on last 7 days",
               style={'marginLeft': 20, 'color': COLORS['text'], 'marginBottom': 30}),
        html.Div(id='team-grid')
    ])


def render_compare_tab():
    return html.Div([
        html.H2("🎥 Compare Athletes", style={'marginLeft': 20, 'color': COLORS['text']}),
        html.Div([
            html.Div([
                html.Label("Athlete 1:", style={'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='compare-athlete-1',
                    options=[{'label': aid, 'value': aid} for aid in sorted(risk_data['athlete_id'].unique())],
                    value=risk_data['athlete_id'].iloc[0],
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 50}),
            
            html.Div([
                html.Label("Athlete 2:", style={'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Dropdown(
                    id='compare-athlete-2',
                    options=[{'label': aid, 'value': aid} for aid in sorted(risk_data['athlete_id'].unique())],
                    value=risk_data['athlete_id'].iloc[1] if len(risk_data['athlete_id'].unique()) > 1 else risk_data['athlete_id'].iloc[0],
                    style={'width': '250px'}
                )
            ], style={'display': 'inline-block'})
        ], style={'marginLeft': 20, 'marginBottom': 30}),
        html.Div(id='comparison-content')
    ])


@app.callback(
    Output('team-grid', 'children'),
    Input('main-tabs', 'value')
)
def update_team_grid(tab):
    if tab != 'team':
        return html.Div()
    
    team_status = []
    for athlete_id in risk_data['athlete_id'].unique():
        athlete_data = risk_data[risk_data['athlete_id'] == athlete_id]
        status = get_athlete_current_status(athlete_data)
        if status:
            team_status.append(status)
    
    team_status.sort(key=lambda x: x['current_risk'], reverse=True)
    
    cards = []
    for status in team_status:
        if status['risk_category'] == 'High':
            border_color = COLORS['danger']
            bg_color = '#FFE8E8'
        elif status['risk_category'] == 'Moderate':
            border_color = COLORS['caution']
            bg_color = '#FFF8E1'
        else:
            border_color = COLORS['safe']
            bg_color = '#E8F5E9'
        
        card = html.Div([
            html.H4(status['athlete_id'], style={'margin': '0 0 10px 0', 'color': COLORS['text']}),
            html.Div([
                html.Div([
                    html.Strong("Risk: "),
                    html.Span(f"{status['current_risk']:.0f}/100", 
                             style={'color': border_color, 'fontSize': 18, 'fontWeight': 'bold'})
                ]),
                html.Div([html.Strong("ACWR: "), f"{status['avg_acwr']:.2f}"]),
                html.Div([html.Strong("Load: "), f"{status['avg_load']:.0f}"]),
                html.Div([html.Strong("Soreness: "), f"{status['avg_soreness']:.1f}/10"]),
                html.Div([html.Strong("Type: "), status['archetype']], 
                        style={'fontSize': 11, 'color': COLORS['chronic'], 'marginTop': 5})
            ])
        ], style={
            'width': '220px',
            'display': 'inline-block',
            'margin': '10px',
            'padding': '15px',
            'backgroundColor': bg_color,
            'borderLeft': f'5px solid {border_color}',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'verticalAlign': 'top'
        })
        
        cards.append(card)
    
    return html.Div(cards, style={'marginLeft': 10})


@app.callback(
    [Output('comparison-content', 'children')],
    [Input('compare-athlete-1', 'value'),
     Input('compare-athlete-2', 'value')]
)
def update_comparison(athlete1, athlete2):
    if not athlete1 or not athlete2:
        return [html.Div()]
    
    data1 = risk_data[risk_data['athlete_id'] == athlete1]
    data2 = risk_data[risk_data['athlete_id'] == athlete2]
    
    status1 = get_athlete_current_status(data1)
    status2 = get_athlete_current_status(data2)
    
    if not status1 or not status2:
        return [html.Div("Insufficient data")]
    
    comparison = html.Div([
        html.Div([
            html.Div([
                html.H3(athlete1, style={'textAlign': 'center', 'color': COLORS['text']}),
                html.Div([
                    html.Div([html.Strong("Risk Score:"), f" {status1['current_risk']:.0f}/100"]),
                    html.Div([html.Strong("Category:"), f" {status1['risk_category']}"]),
                    html.Div([html.Strong("Avg ACWR:"), f" {status1['avg_acwr']:.2f}"]),
                    html.Div([html.Strong("Avg Load:"), f" {status1['avg_load']:.0f}"]),
                    html.Div([html.Strong("Soreness:"), f" {status1['avg_soreness']:.1f}/10"]),
                    html.Div([html.Strong("High Risk Days:"), f" {status1['high_risk_days']}"]),
                ], style={'lineHeight': 2})
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px',
                     'backgroundColor': 'white', 'borderRadius': '8px', 'verticalAlign': 'top'}),
            
            html.Div("VS", style={'width': '8%', 'display': 'inline-block', 'textAlign': 'center',
                                 'fontSize': 24, 'fontWeight': 'bold', 'color': COLORS['chronic'],
                                 'verticalAlign': 'middle'}),
            
            html.Div([
                html.H3(athlete2, style={'textAlign': 'center', 'color': COLORS['text']}),
                html.Div([
                    html.Div([html.Strong("Risk Score:"), f" {status2['current_risk']:.0f}/100"]),
                    html.Div([html.Strong("Category:"), f" {status2['risk_category']}"]),
                    html.Div([html.Strong("Avg ACWR:"), f" {status2['avg_acwr']:.2f}"]),
                    html.Div([html.Strong("Avg Load:"), f" {status2['avg_load']:.0f}"]),
                    html.Div([html.Strong("Soreness:"), f" {status2['avg_soreness']:.1f}/10"]),
                    html.Div([html.Strong("High Risk Days:"), f" {status2['high_risk_days']}"]),
                ], style={'lineHeight': 2})
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px',
                     'backgroundColor': 'white', 'borderRadius': '8px', 'verticalAlign': 'top'})
        ], style={'marginLeft': 20, 'marginRight': 20, 'marginBottom': 30}),
        
        html.Div([
            dcc.Graph(figure=create_timeline_figure(data1, athlete1), 
                     style={'width': '49%', 'display': 'inline-block'}),
            dcc.Graph(figure=create_timeline_figure(data2, athlete2), 
                     style={'width': '49%', 'display': 'inline-block'})
        ])
    ])
    
    return [comparison]


@app.callback(
    [Output('kpi-gauges', 'children'),
     Output('quick-stats', 'children'),
     Output('main-timeline', 'figure'),
     Output('recommendations-panel', 'children')],
    [Input('athlete-selector', 'value'),
     Input('date-range-slider', 'value')]
)
def update_individual_dashboard(athlete_id, date_range):
    if not athlete_id or not date_range:
        return html.Div("Select athlete"), html.Div(""), go.Figure(), html.Div("")
    
    athlete_data = risk_data[risk_data['athlete_id'] == athlete_id].copy()
    
    start_date = min_date + timedelta(days=date_range[0])
    end_date = min_date + timedelta(days=date_range[1])
    filtered_data = athlete_data[
        (athlete_data['date'] >= start_date) & 
        (athlete_data['date'] <= end_date)
    ]
    
    recent_valid = filtered_data[filtered_data['acwr'].notna()].tail(7)
    
    if len(recent_valid) == 0:
        return html.Div("Insufficient data"), html.Div(""), go.Figure(), html.Div("")
    
    latest = recent_valid.iloc[-1]
    avg_acwr = recent_valid['acwr'].mean()
    avg_weekly_load = recent_valid['acute_load'].mean()
    avg_soreness = recent_valid['soreness'].mean()
    current_risk = latest['risk_score']
    
    # Create gauges
    acwr_gauge = create_gauge(
        value=avg_acwr,
        title="ACWR (7-day avg)",
        range_max=2.0,
        thresholds={
            'danger_threshold': 1.4,
            'steps': [
                {'range': [0, 0.85], 'color': COLORS['warning']},
                {'range': [0.85, 1.25], 'color': COLORS['safe']},
                {'range': [1.25, 1.4], 'color': COLORS['caution']},
                {'range': [1.4, 2.0], 'color': COLORS['danger']}
            ]
        }
    )
    
    risk_gauge = create_gauge(
        value=current_risk,
        title="Risk Score",
        range_max=100,
        thresholds={
            'danger_threshold': 50,
            'steps': [
                {'range': [0, 25], 'color': COLORS['safe']},
                {'range': [25, 50], 'color': COLORS['caution']},
                {'range': [50, 75], 'color': COLORS['warning']},
                {'range': [75, 100], 'color': COLORS['danger']}
            ]
        }
    )
    
    load_gauge = create_gauge(
        value=avg_weekly_load,
        title="Weekly Load",
        range_max=max(athlete_data['acute_load'].max() * 1.2, 3000),
        thresholds={
            'danger_threshold': athlete_data['acute_load'].quantile(0.9),
            'steps': [
                {'range': [0, athlete_data['acute_load'].quantile(0.3)], 'color': COLORS['caution']},
                {'range': [athlete_data['acute_load'].quantile(0.3), 
                          athlete_data['acute_load'].quantile(0.8)], 'color': COLORS['safe']},
                {'range': [athlete_data['acute_load'].quantile(0.8), 
                          max(athlete_data['acute_load'].max() * 1.2, 3000)], 'color': COLORS['danger']}
            ]
        }
    )
    
    soreness_gauge = create_gauge(
        value=avg_soreness,
        title="Soreness (7-day)",
        range_max=10,
        thresholds={
            'danger_threshold': 6,
            'steps': [
                {'range': [0, 3], 'color': COLORS['safe']},
                {'range': [3, 5], 'color': COLORS['caution']},
                {'range': [5, 7], 'color': COLORS['warning']},
                {'range': [7, 10], 'color': COLORS['danger']}
            ]
        }
    )
    
    kpi_row = html.Div([
        html.Div([dcc.Graph(figure=acwr_gauge)], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=risk_gauge)], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=load_gauge)], style={'width': '25%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=soreness_gauge)], style={'width': '25%', 'display': 'inline-block'})
    ])
    
    # Quick stats
    valid_data = filtered_data[filtered_data['acwr'].notna()]
    high_risk_days = (valid_data['risk_category'] == 'High').sum()
    moderate_risk_days = (valid_data['risk_category'] == 'Moderate').sum()
    load_spikes = valid_data['load_spike'].sum()
    max_acwr = valid_data['acwr'].max()
    
    stats_style = {
        'display': 'inline-block',
        'width': '24.5%',
        'padding': '20px 15px',
        'margin': '0',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'textAlign': 'center',
        'verticalAlign': 'top'
    }
    
    quick_stats = html.Div([
        html.Div([
            html.H3(f"{high_risk_days}", style={'color': COLORS['danger'], 'margin': '0 0 5px 0', 'fontSize': '32px'}),
            html.P("High Risk Days", style={'margin': 0, 'fontSize': '13px', 'color': COLORS['text']})
        ], style=stats_style),
        
        html.Div([
            html.H3(f"{moderate_risk_days}", style={'color': COLORS['caution'], 'margin': '0 0 5px 0', 'fontSize': '32px'}),
            html.P("Moderate Risk Days", style={'margin': 0, 'fontSize': '13px', 'color': COLORS['text']})
        ], style=stats_style),
        
        html.Div([
            html.H3(f"{load_spikes}", style={'color': COLORS['warning'], 'margin': '0 0 5px 0', 'fontSize': '32px'}),
            html.P("Load Spikes", style={'margin': 0, 'fontSize': '13px', 'color': COLORS['text']})
        ], style=stats_style),
        
        html.Div([
            html.H3(f"{max_acwr:.2f}", style={'color': COLORS['danger'] if max_acwr > 1.4 else COLORS['text'], 'margin': '0 0 5px 0', 'fontSize': '32px'}),
            html.P("Peak ACWR", style={'margin': 0, 'fontSize': '13px', 'color': COLORS['text']})
        ], style=stats_style)
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '0.5%'})
    
    timeline_fig = create_timeline_figure(filtered_data, athlete_id, (start_date, end_date))
    
    # Recommendations
    recent_risk = valid_data[valid_data['risk_score'] > 25].tail(1)
    
    if len(recent_risk) > 0:
        risk_row = recent_risk.iloc[0]
        recommendations = []
        
        if risk_row['acwr'] > 1.4:
            recommendations.append(f"⚠️ ACWR elevated at {risk_row['acwr']:.2f} - Reduce load by {int((risk_row['acwr']-1.25)/risk_row['acwr']*100)}%")
        if risk_row['load_spike']:
            recommendations.append("⚠️ Recent load spike - Maintain current volume 1-2 weeks")
        if risk_row['soreness'] > 5:
            recommendations.append(f"⚠️ Elevated soreness ({risk_row['soreness']:.1f}/10) - Prioritize recovery")
        if risk_row['has_prior_injury']:
            recommendations.append("🤓 Prior injury history - Continue conservative progression")
        if risk_row['risk_category'] == 'High':
            recommendations.append("🚨 HIGH RISK - Consider medical consultation")
        
        rec_items = [html.Li(rec, style={'marginBottom': 8}) for rec in recommendations]
        
        rec_panel = html.Div([
            html.H3("💪 Current Recommendations", style={'color': COLORS['text'], 'marginBottom': 15}),
            html.Ul(rec_items)
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'borderLeft': f'4px solid {COLORS["warning"]}'
        })
    else:
        rec_panel = html.Div([
            html.H3("✅ Status: Low Risk", style={'color': COLORS['safe']}),
            html.P("No immediate concerns. Continue with standard monitoring.")
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '8px',
            'borderLeft': f'4px solid {COLORS["safe"]}'
        })
    
    return kpi_row, quick_stats, timeline_fig, rec_panel


@app.callback(
    Output('export-status', 'children'),
    Input('export-btn', 'n_clicks'),
    State('athlete-selector', 'value')
)
def export_report(n_clicks, athlete_id):
    if n_clicks == 0:
        return ""
    
    athlete_data = risk_data[risk_data['athlete_id'] == athlete_id]
    status = get_athlete_current_status(athlete_data)
    
    if status:
        report_text = f"""
ATHLETE RISK REPORT - {athlete_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

CURRENT STATUS:
- Risk Score: {status['current_risk']:.0f}/100 ({status['risk_category']})
- Avg ACWR (7-day): {status['avg_acwr']:.2f}
- Avg Weekly Load: {status['avg_load']:.0f}
- Avg Soreness: {status['avg_soreness']:.1f}/10
- High Risk Days: {status['high_risk_days']}
        """
        
        with open(f"data/{athlete_id}_report.txt", 'w') as f:
            f.write(report_text)
        
        return f"✅ Report exported: data/{athlete_id}_report.txt"
    
    return "❌ Error"


if __name__ == '__main__':
    print("\n" + "="*70)
    print("💪 ATHLETE INJURY RISK ENGINE 💪")
    print("="*70)
    print("\n✅ Starting dashboard...")
    print("📊 Open: http://localhost:1229")
    print("⚠️  Press Ctrl+C to stop\n")
    
    app.run(debug=True, port=1229)
