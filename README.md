# RankedChoices.jl
### Gleaning Insights from Ordinal Preferences

You've gone through the hard work of collecting a survey with ranked choices. You appreciate the beauty of this format, by which participants relay their preferences without getting caught up in allocating abstract units of utility. What next? It would be a shame to throw away any of the population dynamics conveyed through these rankings.

This package was developed with the single aim of capturing the bevy of high-order relations between cohort preferences. It provides a flexible Bayesian model, implements two samplers, and exposes an arsenal of procedures to interpret resulting posteriors. Key features include mixtures of full-rank multivariate Gaussian cohorts, and the ability for one model to accommodate rankings between distinct sets of items.

Idealistically, our elections should run on this kind of algorithm. See this [blog post](https://myrl.marmarel.is/posts/ranked-choice-voting/) that elaborates further on the subject.


### Installation
`pkg> add RankedChoices`

(once the package is approved)

## Interface

Two simulation tactics are exposed: `RejectionSim` and `HamiltonianSim`. The original and simpler implementation, `RejectionSim(n_sample_attempts::Int)`, converges in fewer iterations because its samples are independent. It is, however, drastically slower in handling large rankings. Please ensure that `n_sample_attempts` is large enough that the resultant `n_failures / (n_voters * n_iterations)` is close to zero.

The snazzy `HamiltonianSim(n_trajectories::Int, collision_limit::Int)` is much more complex, but converges quickly even on large rankings. You can set parameter `n_trajectories := 1` if you don't mind consecutive samples being correlated. Additionally, `collision_limit` can be set to a relatively high number like `128`. It determines how many times a ball can bounce between the walls of a linearly constrained Gaussian before we give up early.


More details on the core methods coming (very) soon.
