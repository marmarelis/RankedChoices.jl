# RankedChoices.jl
### Gleaning Insights from Ordinal Preferences

You've gone through the hard work of collecting a survey with ranked choices. You appreciate the beauty of this format, by which participants relay their preferences without getting caught up in allocating abstract units of utility. What next? It would be a shame to throw away any of the population dynamics conveyed through these rankings.

This package was developed with the single aim of capturing the bevy of high-order relations between cohort preferences. It provides a flexible Bayesian model, implements two samplers, and exposes an arsenal of procedures to interpret resulting posteriors. Key features include mixtures of full-rank multivariate Gaussian cohorts, and the ability for one model to accommodate rankings between distinct sets of items.

Idealistically, our elections should run on this kind of algorithm. See this [blog post](https://myrl.marmarel.is/posts/ranked-choice-voting/) that elaborates further on the subject.


### Installation
`pkg> add RankedChoices`

(press `]` in the REPL to invoke the package manager on your current project)

## Interface
Here I detail how to parse rankings, handle the Gibbs sampler, and interpret the results. For instance,
```julia
# (n_participants x ranking_size)
rankings = [
  "B" "C" "D";
  "A" "D" "C";
  "B" "C" "E";
  "A" "B" "" ; # denote missing entries with empty strings
  "D" "A" "B";
  "A" ""  "" ] # e.g...

# there are other ways to extract IssueVote structures
votes, candidates = parse_matrix(rankings)

n_candidates = length(candidates) # 5, per above
n_cohorts = 3
n_trials = 40_000

prior = make_impartial_prior(n_candidates, n_cohorts,
  precision_scale=1e1, precision_dof=1e2,
  mean_scale=1e0, dirichlet_weight=1e1)

simulation = HamiltonianSim(
  n_trajectories=1, collision_limit=128)

result = simulate(prior, simulation, votes, n_trials,
  n_burnin=10_000, seed=1337)

```

### Structures

The main data type to carry your observations is the `IssueVote`, which contains an array `choices` of `RankedChoice` immutable objects. They correspond to a participant's ranking of `1:n_candidates` items. All `RankedChoice`s of each `IssueVote` are static integer vectors of common length `R`, the maximum ranking size. A model actually consumes `MultiIssueVote` objects, which wrap an `IssueVote` tuple.

### Parsing

You produce `IssueVote(choices, n_candidates)` objects by manual construction or through...
* `parse_matrix(votes::Matrix{String})`
* the more versatile `parse_matrix(votes, candidates)` where you list off the candidate strings in your desired order, and unrecognized strings are tossed into the second return value.
* `read_csv(filename, candidates)`
* `read_xlsx(filename, candidates)`



### Simulators

Two simulation tactics are exposed: `RejectionSim` and `HamiltonianSim`. The original and simpler implementation, `RejectionSim(n_sample_attempts::Int)`, converges in fewer iterations because its samples are independent. It is, however, drastically slower in handling large rankings. Please ensure that `n_sample_attempts` is large enough that the resultant `n_failures / (n_voters * n_iterations)` is close to zero.

The snazzy `HamiltonianSim(n_trajectories::Int, collision_limit::Int)` is much more complex, but converges quickly even on large rankings. You can set parameter `n_trajectories := 1` if you don't mind consecutive samples being correlated. Additionally, `collision_limit` can be set to a relatively high number like `128`. It determines how many times a ball can bounce between the walls of a linearly constrained Gaussian before we give up early.


More details on the core methods coming (very) soon.
