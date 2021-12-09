##  Copyright 2021 Myrl Marmarelis
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

using StatsBase, Statistics
using StaticArrays
using PDMats
using Distributions
using LinearAlgebra
using ProgressMeter: Progress, next!

import Base.@kwdef

# we shall also use the prior to draw an initial estimate/guess
# parameters {N,M} exist solely to inform the methods downstream
@kwdef struct Prior{ N, M, T }
  precision_scale::AbstractPDMat{T}
  precision_dof::T
  mean_loc::AbstractVector{T}
  mean_scale::T
  dirichlet_weights::AbstractVector{T}
  n_sample_attempts::Int
end

function make_impartial_prior(N::Int, M::Int;
    precision_scale::T, precision_dof::T, mean_scale::T,
    dirichlet_weight::T, n_sample_attempts::Int) where T
  Prior{N,M,T}(
    precision_scale = PDMat(fill(precision_scale, N) |> Diagonal |> Matrix),
    mean_loc = fill(zero(T), N),
    dirichlet_weights = fill(dirichlet_weight, M);
    precision_dof, mean_scale, n_sample_attempts)
end

function simulate(prior::Prior{N,M,T}, choices::Vector{RankedChoice{R}},
    n_rounds::Int; n_burnin::Int = 0, indifference::Bool = false,
    verbose::Bool = false, gc_interval::Int = 0)::NamedTuple where {N,M,R,T}
  n_total_rounds = n_rounds + n_burnin
  initial_cohorts = sample_mixture_posteriors(VoterRealization{N,M,T}[];
    prior.precision_scale, prior.precision_dof, prior.mean_loc, prior.mean_scale)
  initial_shares = fill(one(T)/M, M)
  mixture = VoterMixture{N,M,T}(initial_cohorts, initial_shares)
  n_voters = length(choices)
  voters = zeros(VoterRealization{N,M,T}, n_voters)
  mixtures = VoterMixture{N,M,T}[]
  n_failures = 0
  progress = Progress(n_total_rounds, enabled=verbose)
  for trial in 1:n_total_rounds
    n_local_failures = sample_utilities!(
      voters, choices, mixture, prior.n_sample_attempts, indifference)
    if trial > n_burnin
      n_failures += n_local_failures # don't count them otherwise
    end
    shares = sample_mixture_shares(voters, prior.dirichlet_weights)
    cohorts = sample_mixture_posteriors(voters; prior.precision_scale,
      prior.precision_dof, prior.mean_loc, prior.mean_scale)
    mixture = VoterMixture{N,M,T}(cohorts, shares)
    push!(mixtures, mixture)
    report = () -> [(:means, mean(mixture))]
    next!(progress, showvalues=report)
    if gc_interval > 0 && trial % gc_interval == 0
      GC.gc(true) # full round to reorganize memory and avoid continuous swapping from fragmented objects in virtual memory?
    end
  end
  sampled_mixtures = @view mixtures[(1+n_burnin):end]
  (; mixtures=sampled_mixtures, n_failures )
end
