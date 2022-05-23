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
using Random
using ProgressMeter: Progress, next!

import Base.@kwdef

abstract type Simulation end

# more of a tactic than a strategy
@kwdef struct RejectionSim <: Simulation
  n_sample_attempts :: Int
end

@kwdef struct HamiltonianSim <: Simulation
  n_trajectories :: Int
  collision_limit :: Int
end

# we shall also use the prior to draw an initial estimate/guess
# parameters {N,M} exist solely to inform the methods downstream
# for multiple issues: support a block-diagonal prior?
@kwdef struct Prior{ N, M, T }
  precision_scale :: AbstractPDMat{T}
  precision_dof :: T
  mean_loc :: AbstractVector{T}
  mean_scale :: T
  dirichlet_weights :: AbstractVector{T}
end

function make_impartial_prior(N::Int, M::Int;
    precision_scale::T, precision_dof::T, mean_scale::T,
    dirichlet_weight::T) where T
  Prior{N,M,T}(
    precision_scale = PDMat(fill(precision_scale, N) |> Diagonal |> Matrix),
    mean_loc = fill(zero(T), N),
    dirichlet_weights = fill(dirichlet_weight, M);
    precision_dof, mean_scale)
end

function simulate_utilities(simulation::RejectionSim, status, voters, vote, mixture, indifference)
  n_local_failures = rejection_sample_utilities!(
    voters, vote, mixture, simulation.n_sample_attempts, indifference)
  n_failures = ( status === nothing ?
    n_local_failures : (status.n_failures + n_local_failures) )
  (; n_failures )
end

# should be more than reasonable for Julia to infer fully concrete types here
function simulate_utilities(simulation::HamiltonianSim, status, voters, vote, mixture, indifference)
  for (voter_index, voter) in enumerate(voters)
    @assert( obeys_ranking(voter.utility, voter_index, vote, indifference), # disable for high-performance sessions?
      "ranking violated: $voter, $voter_index" )
  end
  local_incompleteness = hamiltonian_sample_utilities!(
    voters, vote, mixture, simulation.n_trajectories,
    simulation.collision_limit, indifference)
  incompleteness = ( status === nothing ?
    local_incompleteness : (status.incompleteness + local_incompleteness) )
  (; incompleteness )
end

# we do marginal (univariate) moments, not covariance and higher-order relations..
# note that multivariate normal does not carry any information above second order,
# but the mixture and the predictive posterior could (and almost always do).
# what if I included rolling quantiles for each voter's utility?
function simulate(prior::Prior{N,M,T}, simulation::Simulation,
    vote::Union{MultiIssueVote, IssueVote}, n_rounds::Int; seed::Int,
    n_utility_moments::Int = 1, n_burnin::Int = 0, indifference::Bool = false,
    verbose::Bool = false, gc_interval::Int = 0, do_coincident::Bool = true
    )::NamedTuple where {N,M,T}
  L = N * N
  Random.seed!(seed)
  if vote isa IssueVote
    vote = (vote,) :: MultiIssueVote
  end # sanitize.(vote.choices, vote.n_candidates) would have been fabulous syntax
  vote = map(vote) do issue
    map(choice -> sanitize(choice, issue.n_candidates), issue)
  end
  n_total_rounds = n_rounds + n_burnin
  initial_cohorts = sample_mixture_posteriors(VoterRealization{N,M,T}[];
    prior.precision_scale, prior.precision_dof, prior.mean_loc, prior.mean_scale)
  initial_shares = rand(prior.dirichlet_weights|>Dirichlet) #fill(one(T)/M, M)
  mixture = VoterMixture{N,M,T,L}(initial_cohorts, initial_shares)
  n_voters = length(vote[1].choices)
  voters = map(1:n_voters) do voter
    VoterRealization{N,M,T}(
      rand(initial_shares|>Categorical), zero(Utility{N,T}) )
  end # the above needs to be adjusted to actually properly fit the rankings,
  # so that we're not resting on all the walls (if initialized at zero, and
  # in the Hamiltonian case) and possibly passing right through them...
  #rejection_sample_utilities!(
  #  voters, vote, mixture, Int(1e10), indifference) # make sure no failure occurs?
  voters = map(1:n_voters) do voter
    VoterRealization{N,M,T}(
      rand(initial_shares|>Categorical),
      shoehorn_ranking(@SVector(randn(T, N)),
          voter, vote, indifference) )
  end
  moments = [ zero(Utility{N,T})
    for m in 1:n_utility_moments, v in 1:n_voters ]
  coincident = ( do_coincident ?
    zeros(Int, n_voters, n_voters) : nothing )
  mixtures = VoterMixture{N,M,T}[]
  sim_status = nothing
  progress = Progress(n_total_rounds, enabled=verbose)
  for trial in 1:n_total_rounds
    sim_status = simulate_utilities(simulation, sim_status,
      voters, vote, mixture, indifference)
    if trial <= n_burnin
      status = nothing # don't count them if still in burnin stage
    else
      for v in 1:n_voters
        for m in 1:n_utility_moments
          moments[m, v] += voters[v].utility .^ m
        end
      end
      if do_coincident
        count_coincidences!(coincident, voters)
      end
    end
    shares = sample_mixture_shares(voters, prior.dirichlet_weights)
    cohorts = sample_mixture_posteriors(voters; prior.precision_scale,
      prior.precision_dof, prior.mean_loc, prior.mean_scale)
    mixture = VoterMixture{N,M,T,L}(cohorts, shares)
    push!(mixtures, mixture)
    report = () -> [(:means, mean(mixture))]
    next!(progress, showvalues=report)
    if gc_interval > 0 && trial % gc_interval == 0
      GC.gc(true) # full round to reorganize memory and avoid continuous swapping from fragmented objects in virtual memory?
    end
  end
  sampled_mixtures = @view mixtures[(1+n_burnin):end]
  moments ./= n_rounds
  (; mixtures=sampled_mixtures, coincident, moments, sim_status... )
end
