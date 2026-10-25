# Metabolic link between type 2 diabetes and Alzheimer's disease

A dynamic mathematical model that predicts an individual's neurodegenerative risk
from routine clinical data and identifies the intervention points that most reduce it.

Biomathematics 374, Stellenbosch University — Serena Taylor Parkins & Katharine Jones.

## What the model does

Three stages combined into one pipeline:

1. **Network A** — a neural-network classifier trained on real clinical data and
   validated against diabetes labels (the empirical anchor). Features are chosen with
   a three-way train/validation/test split, the output is a softmax probability, and
   the network is compared against a logistic-regression baseline with bootstrap
   confidence intervals.
2. **The ODE** — a six-compartment system of differential equations that turns a
   patient's metabolic burden into a 50-year brain trajectory (insulin signalling,
   GSK-3β, amyloid, tau, neuroinflammation, neuronal density), with two feedback
   loops. Solved with `ode45`.
3. **Network B** — a surrogate network that emulates the ODE so interventions can be
   searched quickly.

## Files

| File | What it is |
|---|---|
| `biomath.m` | **The whole project in one file.** Run this. |
| `diabetes_model_input.csv` | The cleaned cohort (5,922 patients). |

## How to run

Open MATLAB, put both files in the **same folder**, make it the Current Folder, then:

```matlab
biomath
```

It runs seven parts in order: (1) Network A + baseline, (2) the ODE and four
profiles, (3) numerical analysis — RK4 convergence, tolerance study, stiffness check
and Jacobian eigenvalues, (3B) intervention synergy and timing, (4) Network B, (5) an
end-to-end real-patient demo, (6) saving results, and (7) the live typed-in-patient
demo. Edit the `demoPatient` block at the top of the file to change the demo patient.

**Requires:** MATLAB with the Statistics and Machine Learning Toolbox
(`fitcnet`, `fitrnet`, `fitclinear`, `perfcurve`, `cvpartition`, `lhsdesign`).
`ode45` and `ode15s` are base MATLAB.

## What a run produces

- **`final_run.txt`** — every line printed to the console, saved automatically (via `diary`).
- **`projectResults.mat`** — the trained models, AUCs and confidence intervals, ODE
  results, numerical-analysis stats and intervention results.
- **Figures** — the ODE trajectories, the intervention bar chart, and the live-demo plots.

## Data and code availability

The dataset is a de-identified community diabetes-screening cohort, cleaned from the
original file to the variables used here. All code to reproduce the results is in this
repository.

## Note on the model

ODE state variables are dimensionless; the rate constants are in per-year units
(`dAP` and `kW` are dimensionless multipliers). The parameters are literature-informed,
not formally fitted — so the ordering and relative size of the results are meaningful,
but absolute values are not yet. The cohort contains no dementia outcome, so the
diabetes→Alzheimer's link is encoded from the literature as differential equations
rather than learned from data.

## Generative AI use declaration

Generative AI (Claude) was used to help draft and comment the code and structure the
write-up. All results were verified, and the mathematics and modelling choices are the
authors' own understanding.
