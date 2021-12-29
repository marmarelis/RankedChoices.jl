# RankedChoices.jl
### Gleaning Insights from Ordinal Preferences

You've gone through the hard work of collecting a survey with ranked choices. You appreciate the beauty of this format, by which participants relay their preferences without getting caught up in allocating abstract units of utility. What next? It would be a shame to throw away any of the population dynamics conveyed through these rankings.

This package was developed with the single aim of capturing the bevy of high-order relations between cohort preferences. It provides a flexible Bayesian model, implements two samplers, and exposes various procedures to interpret resulting posteriors. Key features include mixtures of full-rank multivariate Gaussian cohorts, and the ability for one model to accommodate rankings between distinct sets of items.

Idealistically, our elections should run on this kind of algorithm. See this [blog post](https://myrl.marmarel.is/posts/ranked-choice-voting/) that elaborates further on the subject.
