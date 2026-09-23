# DIRISA SDC 2026 — Turnout Forecasting

Predicting expected voter turnout per ward/municipality for the 4 November 2026
Local Government Elections, using historical turnout patterns from the 2016 and
2021 LGE results.

**Team qualification submission — due 8:00 AM, 28 September 2026.**

## Problem statement

Can historical registration and turnout patterns (2016, 2021) predict expected
voter turnout per ward or municipality for the 2026 Local Government Elections?

**Why it matters:** turnout forecasts help the IEC and parties plan logistics
and outreach, and let journalists and civil society flag wards at risk of low
participation before election day.

## Team

| Member | Focus area |
|---|---|
| _name_ | Data acquisition & cleaning |
| _name_ | Feature engineering & EDA |
| _name_ | Modeling & deployment |

## Datasets

All sourced from the IEC (no scraping required):

- 2016 LGE results — https://results.elections.org.za/home/downloads/npe-results
- 2021 LGE results — https://results.elections.org.za/home/downloads/me-results
- Party registration statistics — https://www.elections.org.za/pw/StatsData/Political-Parties-Statistics
- (Optional context) Current voter registration statistics — https://www.elections.org.za/pw/StatsData/Voter-Registration-Statistics

## Project structure

```
data/
  raw/          # untouched downloads
  interim/      # cleaned but not yet joined
  processed/    # final ward-level feature table — the shared handoff artifact
notebooks/      # numbered pipeline, run in order
src/            # reusable functions imported by notebooks and the app
app/            # Streamlit deployment
slides/         # presentation deck
video/          # recorded presentation (or a link to it, if too large for git)
```

## Pipeline

1. `01_data_collection` — pull raw files, confirm ward ID / column consistency between 2016 and 2021
2. `02_cleaning_and_merge` — join 2016 + 2021 on ward, output `data/processed/ward_panel.csv`
3. `03_eda` — turnout distributions, trends, sanity checks
4. `04_feature_engineering` — turnout, turnout delta, registration growth rate, spoiled ballot ratio
5. `05_modeling` — train on 2016 features → 2021 turnout; validate this is a genuine
   out-of-sample check, not an in-sample fit; then apply the validated approach to
   forecast 2026
6. `06_evaluation` — metrics, error analysis, limitations discussion

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Known limitations (fill in as you go)

- Turnout depends on unobserved factors the model can't see (weather, local
  issues, candidate scandals, protests).
- Municipal demarcation may have shifted ward boundaries between 2016 and 2021 —
  document how any join mismatches were handled.
- _add more as you discover them_

## Submission checklist

- [ ] Defined problem statement + why it matters
- [ ] Working, well-commented, reproducible notebook/codebase (full pipeline)
- [ ] Trained model addressing the problem statement
- [ ] Model deployed in a usable form (app/dashboard/API/interactive notebook)
- [ ] Honest discussion of data limitations
- [ ] 15-minute video, every team member presenting
- [ ] Presentation slides with references
- [ ] Uploaded to NextCloud and shared with bsebusho@csir.co.za by 8:00 AM, 28 Sep 2026
