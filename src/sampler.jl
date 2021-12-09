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

using StatsBase
using StaticArrays
using PDMats
using Distributions
using LinearAlgebra
using Random: rand!

function sample_mixture_shares(voters::Vector{VoterRealization{N,M,T}},
    dirichlet_weights::AbstractVector{T})::SVector{M,T} where {N,M,T}
  @assert length(dirichlet_weights) == M
  count_map = countmap(voter.membership for voter in voters)
  counts = SVector{M,T}(get(count_map, cohort, 0) for cohort in 1:M)
  posterior_weights = counts + dirichlet_weights
  Dirichlet(posterior_weights) |> rand
end

# if we were to add weights to voters in order to virtually increase the
# sample size without expanding the spread, like for elections with more
# powerful individuals than others (ahem, shareholder meetings) then we
# would make weighted estimates of the covariance and means here.
# everything else can mostly stay the same?
# care more about the few allocations (in number of candidates) happening here?
function sample_mixture_posteriors(voters::Vector{VoterRealization{N,M,T}};
    precision_scale::AbstractPDMat{T}, precision_dof::T, mean_loc::AbstractVector{T},
    mean_scale::T)::SVector{M,VoterCohort{T}} where {N,M,T}
  SVector{M}( begin
    cohort_voters = Iterators.filter(voters) do voter
      voter.membership == cohort
    end
    cohort_utilities = (voter.utility for voter in cohort_voters)
    n_samples = sum(_ -> 1, cohort_utilities, init=0)
    utility_covariance = zero(SMatrix{N,N,T,N*N})
    utility_means = SVector{N,T}(mean_loc)
    #error("what to do with an empty cohort?")
    #make it draw from the prior, basically
    if n_samples > 0
      utility_covariance = mean(cohort_utilities) do utility
        utility * utility'
      end
      utility_means = mean(cohort_utilities)
    end
    precision_prior_scale = precision_scale + (
      utility_covariance + mean_scale/(mean_scale+n_samples)
        * (utility_means-mean_loc) * (utility_means-mean_loc)' ) / 2
    precision_prior_dof = precision_dof + n_samples/2
    precision_dist = Wishart(precision_prior_dof,
      Symmetric(precision_prior_scale) |> cholesky)
    precision_sample = rand(precision_dist)
    mean_means = (mean_loc*mean_scale + n_samples*utility_means) / (n_samples+mean_scale)
    covariance_sample = inv(precision_sample)
    mean_covariance = covariance_sample / (n_samples+mean_scale)
    mean_dist = MvNormal(mean_means, Symmetric(mean_covariance)) # Hermitian simply to counter numerical instability post-inversion
    mean_sample = rand(mean_dist)
    MvNormal(mean_sample, Symmetric(covariance_sample))
  end for cohort in 1:M )
end

# allow zeros to demarcate absent choices (abstained). first element MUST be nonzero.
# question. say there are five candidates and I fill out
#   3, _, _, _, 2.
# does that actually imply that #2 yields lower utility than the remaining unnamed candidates?
# this special case does not lead to any actionable generalization; it falls apart immediately
# say if R<N or I fill in the fourth entry rather than the fifth.
function obeys_ranking(utility::Utility{N,T}, # AbstractVector for ranking?
    ranking::RankedChoice{R}, indif::Bool)::Bool where {N,R,T}
  @assert ranking[1] > 0
  # don't take the time to check that all elements are within [1,N]
  seen = @MVector fill(false, N)
  last_index = 1
  @inbounds for rank_index in 2:R # pairwise comparisons down the line
    if ranking[rank_index] == 0
      continue
    end
    first = ranking[last_index]
    second = ranking[rank_index]
    #ref_index = min(rank_index-1, R)
    if utility[second] > utility[first]
      return false
    end
    seen[first] = seen[second] = true # logically separate these housekeeping lines from the above
    last_index = rank_index
  end
  # valid so far
  if indif
    return true
  end
  last_ranking = ranking[last_index]
  last_ranked = utility[last_ranking]
  @inbounds for candidate in 1:N
    if seen[candidate]
      continue
    end
    if utility[candidate] > last_ranked
      return false
    end
  end
  return true
end # TODO use LoopVectorization.jl here?

using Base.Threads

# straight into informed sampling. rejection sampler with variable runtime. sample_ordered_gaussian_mixture()
function sample_utilities!(voters::Vector{VoterRealization{N,M,T}},
    choices::Vector{RankedChoice{R}}, mixture::VoterMixture{N,M,T},
    n_sample_attempts::Int, indif::Bool=false)::Int where {N,M,R,T}
  n_voters = length(voters)
  @assert length(choices) == n_voters
  @assert isapprox(sum(mixture.shares), 1)
  categorical = Categorical(mixture.shares) # is it wiser to store `shares` as a `Categorical`?
  n_threads = nthreads()
  sample = zeros(T, N, n_threads) # allocate an array to act as an intermediary in which to store results before assigning to a Vector of SVectors
  utility = zeros(Utility{N,T}, n_threads)
  n_failures = zeros(Int, n_threads)
  @threads for voter_index in 1:n_voters
    thread_id = threadid()
    this_sample = @view sample[:, thread_id]
    cohort_index = 0
    n_failures[thread_id] += 1 # this shall be deemed a failure until proven otherwise
    @inbounds for attempt in 1:n_sample_attempts # beyond this we simply give up and accept a suboptimal fate
      cohort_index = rand(categorical)
      cohort = mixture.cohorts[cohort_index]
      rand!(cohort, this_sample) # per the docs, the global RNG is thread-safe as of Julia 1.3 (I've been following the development since at least sophomore year!)
      ranking = choices[voter_index]
      utility[thread_id] = Utility{N,T}(this_sample)
      if obeys_ranking(utility[thread_id], ranking, indif) # && break
        n_failures[thread_id] -= 1
        break
      end # idea: settle on the one with the FEWEST violations? also try to Gibbs sampler on constrained marginals...
    end
    realization = VoterRealization{N,M,T}(cohort_index, utility[thread_id])
    voters[voter_index] = realization
  end
  sum(n_failures)
end

draw_cohort(mixture::VoterMixture) =
  mixture.cohorts[ Categorical(mixture.shares) |> rand ]

function draw_marginal(mixture::VoterMixture{N,M,T}, candidate::Int)::T where {N,M,T}
  @assert 1 <= candidate <= N
  cohort = draw_cohort(mixture)
  means = mean(cohort)
  covariances = cov(cohort)
  location = means[candidate]
  variance = covariances[candidate, candidate]
  sqrt(variance)*randn() + location
end
