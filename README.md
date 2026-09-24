# DIRISA SDC 2026 — Turnout Forecasting

Predicting expected voter turnout per ward/municipality for the 4 November 2026
Local Government Elections, using historical turnout patterns from the 2011,
2016, and 2021 LGE results.

**Team qualification submission — due 8:00 AM, 28 September 2026.**

## Problem statement

Can historical registration and turnout patterns predict expected voter
turnout per ward or municipality for the 2026 Local Government Elections?

**Why it matters:** turnout forecasts help the IEC and parties plan logistics
and outreach, and let journalists and civil society flag wards at risk of low
participation before election day.

## Team

| Member       | Focus area |
|--------------|---|
| Karabo Nkomo | Data acquisition & cleaning |
| _name_       | Feature engineering & EDA |
| _name_       | Modeling & deployment |

## Datasets

All sourced from the IEC (no scraping required):

- 2011 LGE results — https://results.elections.org.za/home/downloads/me-results (archived
  municipal election results; 2011 files are UTF-16 encoded, unlike 2016/2021)
- 2016 LGE results — https://results.elections.org.za/home/downloads/me-results
- 2021 LGE results — https://results.elections.org.za/home/downloads/me-results
- Party registration statistics — https://www.elections.org.za/pw/StatsData/Political-Parties-Statistics
- (Optional context) Current voter registration statistics — https://www.elections.org.za/pw/StatsData/Voter-Registration-Statistics

## Project structure

```
data/
  raw/          # untouched downloads
  interim/      # cleaned but not yet joined
  processed/    # final ward-level feature tables — the shared handoff artifacts
notebooks/      # numbered pipeline, run in order
src/            # reusable functions imported by notebooks and the app —
                # import shared logic (e.g. to_ward_level) rather than
                # redefining it inline in a notebook
app/            # Streamlit deployment
slides/         # presentation deck
video/          # recorded presentation (or a link to it, if too large for git)
```

## Pipeline

1. `01_data_collection` — pull raw files, confirm ward ID / column consistency between 2016 and 2021
2. `02_cleaning_and_merge` — join 2016 + 2021 on ward, output `data/processed/ward_panel.csv`
3. `02b_add_2011` — extend the panel with 2011 results (different encoding —
   see limitations), output `data/processed/ward_panel_3yr.csv` with the
   corrected, non-leaky delta features (see Feature usage below)
4. `03_eda` — turnout distributions, trends, sanity checks
5. `04_feature_engineering` — turnout, registration growth, spoiled ballot ratio
6. `05_modeling` — train on features known as of 2016 → predict 2021 turnout,
   as a genuine out-of-sample check, not an in-sample fit; then apply the
   validated approach to forecast 2026 from 2021-known features
7. `06_evaluation` — metrics, error analysis, limitations discussion

## Feature usage — read before adding features to the model

With only two election years, a feature like `Turnout_2021 - Turnout_2016`
requires already knowing `Turnout_2021` — the value being predicted. That's
direct label leakage: the model would learn to do subtraction, not to
forecast, and the trick stops working the moment you try to predict 2026
(which hasn't happened yet). Adding 2011 fixes this: a delta between two years
that are **both already in the past relative to the prediction target** is
safe, because the same shape of feature will genuinely exist at deployment
time.

- **Safe model features (from `ward_panel_3yr.csv`):**
  `TurnoutDelta_2011_2016` = `Turnout_2016 - Turnout_2011`,
  `RegistrationGrowth_2011_2016` = `RegisteredVoters_2016 / RegisteredVoters_2011 - 1`,
  plus any single-election snapshot feature (`Turnout_2016`, `RegisteredVoters_2016`,
  spoiled ballot ratio, etc.)
- **EDA / storytelling only, never a model feature:**
  `Turnout_2021 - Turnout_2016` and any delta that touches the current
  prediction target — useful for describing what happened, not for predicting it.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Known limitations (fill in as you go)

- Turnout depends on unobserved factors the model can't see (weather, local
  issues, candidate scandals, protests).
- Municipal demarcation shifted ward boundaries between election cycles —
  ward matching was done on exact `Ward` code + `Province` (not `Municipality`,
  since names/labels changed even when the ward itself didn't). Overlap was
  ~99%/97% between 2016 and 2021, and ~90%/88% between 2011 and 2016 (lower,
  consistent with the municipal amalgamation process ahead of the 2016 LGE).
  Non-matching wards were excluded from the modelling panel.
- Turnout is computed from the **Ward ballot only** (not PR or DC 40%), for
  consistency across metro and local municipality types — summing across all
  ballot types would inflate and inconsistently inflate vote totals.
- A small number of 2016 wards were excluded after investigation: 4 Limpopo
  wards (LIM345, an unresolved demarcation dispute that disrupted 2016 voting)
  and 1 North West ward (Matlosana, an apparent source-data error where
  recorded votes exceeded registered voters at one station). No exclusions
  were required for 2011.
- 2011 source files required UTF-16 decoding, unlike the UTF-8 2016/2021 files.
- See "Feature usage" above for which derived features are safe to model with.
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