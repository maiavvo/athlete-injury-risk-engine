# 💪 Athlete Injury Risk Engine 💪
A dashboard that predicts injury risk for athletes by analyzing their training load patterns.
## What It Does
Monitors athlete training data and flags when someone's at high risk of injury. Combines several metrics that sports medicine researchers have validated (ACWR, load spikes, soreness levels) into one risk score that's easy to understand.
## Features
- **Individual athlete monitoring** - Select an athlete, see their current status
- **Traffic light gauges** - Green/yellow/red indicators for quick decisions
- **Team overview** - Check entire roster at once
- **Comparison view** - Compare two athletes side by side
- **Specific recommendations** - Not just "high risk" but "reduce load by 20%"
The system tracks:
- Training duration and intensity (RPE scale)
- Weekly load patterns (acute vs chronic)
- Self-reported soreness
- Previous injury history
Then calculates an ACWR (Acute-to-Chronic Workload Ratio) - basically comparing recent training load to what the athlete's body is adapted to. Research shows ACWR > 1.4 means significantly higher injury risk.
The risk score combines:
- ACWR (35% weight) - biggest predictor
- Load spikes (25%) - sudden increases are dangerous
- Soreness (20%) - athlete's own feedback
- Injury history (20%) - prior injury increases risk
## Tech Stack
- Python, Pandas, NumPy for data processing
- Plotly Dash for the interactive dashboard
- Deployed on Render
## Live Demo
[View Dashboard](https://athlete-injury-risk-engine.onrender.com)**
*Note: Initial load may take 40-60 seconds as the free-tier server spins up. Please be patient!*
## Running It Locally
```bash
pip install -r requirements.txt
python src/data_generator.py  # Creates sample data
python src/load_metrics.py     # Calculates ACWR and other metrics
python src/risk_model.py       # Scores risk
python src/dashboard.py        # Launches dashboard at localhost:1229
```
## Sample Results
Testing on 20 synthetic athletes over 120 days:
- Identified 5 high-risk days requiring immediate attention
- Flagged 54 moderate-risk days for monitoring
- Caught one athlete at 73.4/100 risk with ACWR of 1.69 (way over safe threshold)
- System recommended specific load reduction percentage
## Limitations
Uses synthetic data, so not validated with actual injury outcomes. Would need real athlete data and injury records to tune the model properly. Risk thresholds might need adjustment based on sport type and athlete level.
## Future Ideas
- Integration with wearable devices (GPS trackers, heart rate monitors)
- Machine learning model trained on real injury data
- Mobile app version
- Automated alerts/notifications

Built as a portfolio project for fun.
