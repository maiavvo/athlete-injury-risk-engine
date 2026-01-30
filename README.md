# 💪 Athlete Injury Risk Prevention Dashboard
A sports medicine analytics system that predicts athlete injury risk using evidence-based training load metrics.
## What It Does
Monitors athlete training data and flags when someone's at high risk of injury. Combines several metrics validated by sports medicine research (ACWR, load spikes, soreness levels) into one 0-100 risk score with specific, actionable recommendations.
## Features
**Individual Athlete Monitoring**
- Select any athlete to view their current risk status
- Traffic-light KPI gauges (green/yellow/red) for quick decision-making
- Specific recommendations like "ACWR elevated at 1.74, reduce load by 28%"
**Team Management**
- Team overview displaying entire roster sorted by risk level
- Add new athletes with custom training parameters
- Remove athletes from the roster
**Analysis Tools**
- Comparison view for side-by-side athlete analysis
- Date filtering to focus on specific training periods
- Real-time data updates across all views
## How It Works
The system tracks:
- Training duration and intensity (RPE scale 1-10)
- Weekly load patterns (acute vs. chronic)
- Self-reported soreness levels
- Previous injury history
Then calculates **ACWR (Acute-to-Chronic Workload Ratio)** - comparing recent training load to what the athlete's body is adapted to. Research shows ACWR > 1.4 significantly increases injury risk.
**Risk Model (0-100 scale):**
- ACWR: 35% weight (primary predictor)
- Load spikes: 25% (sudden increases are dangerous)
- Soreness: 20% (athlete feedback)
- Injury history: 20% (prior injury increases risk)
## Tech Stack
- Python, Pandas, NumPy for data processing
- Plotly Dash for interactive dashboard
- Deployed on Render
## Live Demo
**[View Dashboard](https://athlete-injury-risk-engine.onrender.com)**
## **Initial load may take 30-50 seconds as the free-tier server spins up**

## Running Locally
```bash
pip install -r requirements.txt
python src/data_generator.py  # Creates sample data
python src/load_metrics.py    # Calculates ACWR and metrics
python src/risk_model.py      # Generates risk scores
python src/dashboard.py       # Launches dashboard at localhost:1229
```

## Sample Results
Testing on 20 simulated athletes over 120 days:
- Identified athletes at 69.5/100 and 73.4/100 risk requiring immediate intervention
- Detected ACWR spikes above 1.4 threshold (peak of 1.74)
- Flagged 30 high-risk days and 52 moderate-risk days across the cohort
- Generated specific load reduction recommendations (e.g., "reduce by 28%")

## Athlete Archetypes (Test Data)
The system simulates four realistic training patterns:
**Conservative (CON)** - Stable training, 5-6 days/week, moderate intensity (RPE ~6.0), low soreness (~3.5/10), no injury history
**Aggressive (AGG)** - Variable loads, 6-7 days/week, high intensity (RPE ~8.0), elevated soreness (~6.0/10), no injury history
**Injury-Prone (INJ)** - Cautious approach, 5 days/week, moderate-high intensity (RPE ~6.5), high soreness (~6.5/10), **prior injury = TRUE**
**Optimal (OPT)** - Evidence-based periodization, 5-6 days/week, controlled intensity (RPE ~7.0), moderate soreness (~4.0/10), no injury history
The injury-prone group consistently scores highest for risk, validating the model's ability to identify athletes requiring closer attention.
## Limitations
- The system uses synthetic data that is not validated with actual injury outcomes, so it would require real athlete data and injury records to tune the model properly.
- The risk thresholds may need adjustment based on sport type and competition level and is NOT a replacement for coaching expertise or athlete self-awareness.

## Future Enhancements
- Integration with wearable devices (GPS trackers, heart rate monitors)
- Machine learning model trained on real injury data
- Automated alerts and notifications, export reports for medical staff
---
Built as a portfolio project exploring how to translate sports medicine research into practical tools.
