# Quantum games for robot teams

Code and recorded numerical data for the six-stage delivery experiment in
Section 5 of *Quantum games for robot teams* and its supplementary material.
The experiment uses prescribed stage durations and local quantum measurement
outcomes. It is a numerical model, not a flight-dynamics or hardware experiment.

## Contents

- `scripts/delivery_stage_experiments.py`: exact dynamic programming, parameter
  sweeps and sampled-mission validation for the current experiment.
- `scripts/drone_delivery.py`: local measurements, Born outcome distributions
  and task-success predicates imported by the current experiment.
- `scripts/plot_stage_experiments.py` and `scripts/figure_style.py`: regenerate
  the inventory and memory-ageing figures.
- `delivery_results/stage_experiments.json`: original recorded results.
- `delivery_results/README.md`: detailed model, output schema and validation notes.
- `requirements.txt`: the two direct third-party dependencies.
- `validation_report.json`: the recorded isolated reproduction check dated
  27 October 2026, including source hashes and numerical comparisons.
- `manifest.sha256`: hashes of the repository files as supplied.

## Environment

The recorded verification used Python 3.9.13, NumPy 1.25.2 and Matplotlib 3.8.0
on Windows. This identifies the verification environment; it does not assert
the environment that originally generated the data. Use a compatible Python
version, preferably the verified version, when installing these pinned library
versions. The requirements file pins direct dependencies only, not all transitive
dependencies. Other Python/library combinations have not been verified here.

## Reproduce

Download or clone the repository and open a terminal at its root. Keep the
`scripts/` and `delivery_results/` directories beside each other.

```sh
python -m pip install -r requirements.txt
python -B scripts/delivery_stage_experiments.py --missions 10000
python -B scripts/plot_stage_experiments.py
```

The experiment command recomputes the exact curves and runs 13 sampled
configurations, each with 10,000 missions, using seed 20261014. It overwrites
`delivery_results/stage_experiments.json`; preserve the supplied data or run in
a separate checkout if you want to retain the original file. The plotting
command reads that JSON and creates or overwrites these files at the repository
root:

- `fig_delivery_inventory.pdf` and `fig_delivery_inventory.png`
- `fig_delivery_ageing.pdf` and `fig_delivery_ageing.png`

To regenerate only the figures from the supplied results, run just the plotting
command after installing the dependencies.

Do not run `scripts/drone_delivery.py` directly to reproduce Section 5. Its
standalone entry point belongs to an earlier timing model; the current experiment
imports only its task and measurement functions. The data README is retained
verbatim from the manuscript sources for provenance. Its references to a
"manuscript source directory" mean this repository root for the commands above;
its archived-campaign section describes files in the broader manuscript project,
some of which are deliberately not included here.

## Recorded verification

The supplied validation report records a full isolated execution of both
commands, comparison of 802 numeric fields against the original data, and checks
of the two figures. That run found no numeric differences. The supplied dataset
contains 45 inventory records, 17 memory-sweep records and 13 sampled
configurations. The maximum sampled discrepancy was approximately 1.08 standard
errors. These checks concern reproducibility of the prescribed-duration model,
not real-flight feasibility or hardware performance.

The four scripts and two data/documentation files are byte-for-byte copies of
the manuscript sources, matching the hashes in the recorded validation report.
The report describes that earlier verification run; it is not a claim that a
new full simulation was run when preparing this repository.

The SHA-256 manifest can be checked before a reproduction run. Generated figures
are not distributed, and their PDF hashes may change between environments or
builds because of metadata. Reproduction overwrites the JSON, so its manifest
entry describes the original supplied file.
