# Delivery mission numerical experiments

## Current Section 5 experiments

Run from the manuscript source directory with Python and NumPy:

    python scripts/delivery_stage_experiments.py --missions 10000

This writes `delivery_results/stage_experiments.json`. Regenerate the two current
figures with Python and Matplotlib:

    python scripts/plot_stage_experiments.py

This writes `fig_delivery_inventory.pdf/png` and `fig_delivery_ageing.pdf/png`.
The current experiment is a
stochastic six-stage mission with prescribed durations. It reuses the task
measurements, complete Born outcome distributions, and landing/route success
predicates from `scripts/drone_delivery.py`; it does not reuse that script's
physical trajectory clock or claim that those trajectories fit the new durations.

Each independently drawn stage is Task A, Task B, or idle with probabilities
0.5, 0.3, and 0.2. Successful and idle stages last 10 s. An unsuccessful Task A
stage lasts 50 s and an unsuccessful Task B stage lasts 80 s, including the
nominal 10 s. The associated failure overheads are therefore 40 s and 70 s.
Local observations are independent and uniform within each task, and no radio
message arrives in time to affect a decision. Stage type and past success/failure
are available to the common mission planner; current private observations are
used only in the robots' local task strategies.

Fresh pairs have visibility 0.95 and age zero. All stored pairs have a common age,
which increases by the realised stage duration. Visibility is
`0.95 * exp(-2*age/tm)`; `tm = null` denotes no memory decay. A classical strategy
uses zero pairs; Task A's CHSH strategy uses one; Task B's six-correlation
strategy uses one, while its square strategy uses two. Selected pairs are
consumed regardless of success. Task A's active menu is classical or CHSH, and
Task B's active menu is classical, six-correlation, or square.

The exact dynamic program minimises expected total completion time, including
the fixed 60 s nominal mission duration. The policies and comparison settings are:

- **Age-aware allocation:** plans and evaluates with actual decaying visibility.
- **Ignores ageing:** plans with constant initial visibility but is evaluated
  using the actual decay and outcome-dependent elapsed time.
- **Six initially, no replacement:** uses age-aware allocation with six pairs
  available at departure, and no later supply.
- **Free replacement:** starts with three pairs and uses age-aware allocation; immediately before stage four
  it discards all residual stock and supplies three fresh pairs at no time cost.

The inventory experiment varies initial stock from zero to fourteen pairs for no
decay and memory constants 2400 s and 600 s. At most twelve pairs can be consumed
in six stages, so the extended sweep includes two points beyond this bound.
The exact curves plateau from twelve pairs for no decay and 2400 s, and from
eight pairs for 600 s. The memory experiment varies the memory constant from
30 s to 30000 s. Its main comparison is three initial pairs plus three fresh
pairs before stage four versus six initial pairs without replacement: both
receive six pairs in total, but availability, age and discarded stock differ.
Separate diagnostics retain the original three-pair policies without replacement.
The current memory figure plots total expected completion time and the
104 s classical benchmark. The exact
classical baseline is 104 s. Its expected successful coordination-task count is
3.85; idle stages are excluded from this count.

## Output schema

- `parameters`: all durations, priors, strategy costs, memory law, replacement
  convention, random seed, and sample metric order.
- `inventory`: records with `tm`, `inventory`, `completion_time_s`,
  `successful_coordination_tasks`, and `pairs_consumed`.
- `memory_sweep`: records with `tm` and, for each of `age_aware`,
  `ignores_ageing`, `refill`, and `initial_six`, fields ending in `_s`, `_savings_s`,
  `_successful_tasks`, and `_pairs_consumed`. Savings are `104 - completion time`.
  `parameters.memory_sweep_initial_inventory` records the corresponding initial
  inventories; the figure compares `refill_s` with `initial_six_s`.
- `validation`: independent enumeration checks of success formulas, the exact
  classical baseline, monotonicity/evaluation checks, and maximum Monte Carlo
  discrepancy in standard-error units.
- `samples`: 10000-mission checks of the classical policy, the three-pair quantum
  policies, and the six-pair policy at memory constants 30 s, 600 s, and 2400 s. Each record contains its
  exact tuple, sampled mean, standard error, and signed discrepancy divided by
  that standard error. Tuple order is total completion time, successful
  coordination-task count, and pairs consumed.

## Verification

The script exhaustively checks all 16 landing action cases and 576 route
assignment cases. For every strategy, it enumerates the complete Born outputs
at visibility values 0, 0.3, 0.6, 0.95, and 1 and matches the analytical success
formula to numerical precision. These checks concern task success, with no
physical trajectory timing assertions.

The inventory curves are nonincreasing, zero inventory recovers the classical
baseline, and optimal policy values agree with independent exact policy
evaluation. The age-aware policy is no worse than the policy that ignores ageing
at every memory sweep point. Sampled missions draw full local outputs and apply
the task predicates, rather than drawing success directly from the closed-form
success probability. Seed 20261014 supplies independent task, input, and outcome
streams; each mission index uses common draws across policies. Across the current
campaign, all three sampled metrics agree with the exact values within 1.08
standard errors. This quantifies sampling uncertainty, not uncertainty in the
assumed durations or task probabilities. Replacement is not uniformly better
than six initial pairs: it saves 1.18 s at a 2400 s memory constant and 3.90 s
at 600 s, but is 0.17 s slower at 30000 s. The validation statement that
replacement is no worse refers specifically to the three-initial-pair diagnostic.

## Archived earlier campaign

`results.json`, `scripts/drone_delivery.py`, and `scripts/plot_drone_delivery.py`
preserve the earlier campaign. That campaign used 60 s nominal stages, 20 s input
release, and additional recovery delays of 45 s for Task A and 12 s for Task B,
together with a prescribed trajectory model. Its numerical output and old plots
do not represent the current Section 5 duration assumptions. It can be reproduced
with the two original scripts, but it should not be used to regenerate the
current stage-duration results.
