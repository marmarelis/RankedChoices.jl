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
using StaticArrays # another package to watch is TupleVectors
using PDMats
using Distributions
using LinearAlgebra
using Random: rand!

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


function sample_mixture_shares(voters::Vector{VoterRealization{N,M,T}},
    dirichlet_weights::AbstractVector{T}, voter_weights::Vector{T}=T[]
    )::SVector{M,T} where {N,M,T}
  n_voters = length(voters)
  @assert length(voter_weights) in (0, n_voters)
  @assert length(dirichlet_weights) == M
  counts = @MVector zeros(T, M)
  for (voter_index, voter) in enumerate(voters)
    cohort = voter.membership
    weight = T(1)
    if length(voter_weights) > 0
      weight = voter_weights[voter_index]
    end
    counts[cohort] += weight
  end
  #count_map = countmap(voter.membership for voter in voters)
  #counts = SVector{M,T}(get(count_map, cohort, 0) for cohort in 1:M)
  posterior_weights = counts + dirichlet_weights
  Dirichlet(posterior_weights) |> rand
end

function weighted_mean(f::Function, iterand, weights)
  sum(zip(iterand, weights)) do (value, weight)
    weight * f(value)
  end / sum(weights)
end

# eventually, the best way to improve performance would be to avoid the gradual
# memory fragmentation due to many small matrix allocations. the "purest" approach
# would be to get rid of the crutch that is our dependency on `MvNormal` algotegher.
# retrofit the library procedures entirely in terms of static matrices.
# first step: relieve most of the memory burden/footprint via the persistent objects.
function sample_mixture_posteriors(voters::Vector{VoterRealization{N,M,T}};
    precision_scale::AbstractPDMat{T}, precision_dof::T, mean_loc::AbstractVector{T},
    mean_scale::T, voter_weights::Vector{T}=T[])::SVector{M} where {N,M,T} # VC<:VoterCohort{N,T}
  n_voters = length(voters) # weight for recency (online prediction) or inverse propensity scoring to adjust for bias
  @assert length(voter_weights) in (0, n_voters)
  SVector{M}( begin
    cohort_voter_indices = Iterators.filter(1:n_voters) do voter_index
      voters[voter_index].membership == cohort
    end
    cohort_voters = (voters[voter_index] for voter_index in cohort_voter_indices)
    cohort_weights = Iterators.map(cohort_voter_indices) do voter_index
      length(voter_weights) > 0 ? voter_weights[voter_index] : T(1)
    end
    cohort_utilities = (voter.utility for voter in cohort_voters)
    n_samples = sum(cohort_weights, init=T(0))
    utility_covariance = zero(SMatrix{N,N,T,N*N})
    utility_means = SVector{N,T}(mean_loc)
    #error("what to do with an empty cohort?")
    #make it draw from the prior, basically
    if n_samples > 0
      utility_covariance = weighted_mean(
          cohort_utilities, cohort_weights) do utility
        utility * utility'
      end
      utility_means = weighted_mean(identity,
        cohort_utilities, cohort_weights)
    end
    # to be proper, I would actually invert `precision_scale`
    # see arXiv:2109.07384 for how we use `precision_dof` (the `alpha` therein)
    precision_prior_scale = precision_dof*precision_scale + n_samples*(
      utility_covariance + mean_scale/(mean_scale+n_samples)
        * (utility_means-mean_loc) * (utility_means-mean_loc)' )
    # alongside that Bayesreg.pdf, see https://en.wikipedia.org/wiki/Normal-Wishart_distribution
    # it's rather difficult to find reliable sources on this..
    precision_prior_dof = (precision_dof + N + 1) + n_samples
    precision_dist = Wishart(precision_prior_dof,
      inv(precision_prior_scale) |> Symmetric |> cholesky) # not the most resourceful sequence of operations?
    precision_sample = rand(precision_dist) .|> T
    covariance_sample = inv(precision_sample)
    mean_means = (mean_loc*mean_scale + n_samples*utility_means) / (n_samples+mean_scale)
    mean_covariance = covariance_sample / (n_samples+mean_scale)
    mean_dist = MvNormal(mean_means, Symmetric(mean_covariance)) # Hermitian simply to counter numerical instability post-inversion
    mean_sample = rand(mean_dist) |> SVector{N,T}
    covariance_decomp = cholesky(covariance_sample |> SMatrix{N,N,T,N*N} |> Symmetric)
    MvNormal(mean_sample, PDMat(covariance_decomp))
  end for cohort in 1:M )
end

# allow zeros to demarcate absent choices (abstained). first element MUST be nonzero.
# question. say there are five candidates and I fill out
#   3, _, _, _, 2.
# does that actually imply that #2 yields lower utility than the remaining unnamed candidates?
# this special case does not lead to any actionable generalization; it falls apart immediately
# say if R<N or I fill in the fourth entry rather than the fifth.
function obeys_ranking(utility::Utility{N,T}, # AbstractVector for ranking?
    ranking::RankedChoice{R}, indif::Bool,
    interval::UnitRange{Int}=1:N)::Bool where {N,R,T}
  @assert ranking[1] > 0
  # don't take the time to check that all elements are within [1,N]
  seen = @MVector fill(false, N)
  # passing in a view from the get-go is costlier because the size is dynamic
  offset = interval.start - 1
  last_index = 1
  seen[ranking[1] + offset] = true # first item is always seen
  @inbounds for rank_index in 2:R # pairwise comparisons down the line
    if ranking[rank_index] == 0
      continue
    end
    first = ranking[last_index] + offset
    second = ranking[rank_index] + offset
    #ref_index = min(rank_index-1, R)
    if utility[second] >= utility[first]
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
  @inbounds for candidate in interval
    if seen[candidate]
      continue
    end
    if utility[candidate] >= last_ranked
      return false
    end
  end
  return true
end # TODO use LoopVectorization.jl here?

function obeys_ranking(utility::Utility{N,T},
    voter_index::Int, vote::MultiIssueVote, indif::Bool)::Bool where {N,T}
  offset = 0
  all(vote) do issue
    interval = (offset+1) : (offset+issue.n_candidates)
    offset += issue.n_candidates
    obeys_ranking(utility, issue.choices[voter_index], indif, interval)
  end
end

function check_issues(vote::MultiIssueVote, n_voters::Int, N::Int)
  @assert all( length(issue.choices) == n_voters for issue in vote )
  @assert sum( issue.n_candidates for issue in vote ) == N
end

using Base.Threads

# straight into informed sampling. rejection sampler with variable runtime. sample_ordered_gaussian_mixture()
function rejection_sample_utilities!(voters::Vector{VoterRealization{N,M,T}},
    vote::MultiIssueVote, mixture::VoterMixture{N,M,T},
    n_sample_attempts::Int, indif::Bool=false)::Int where {N,M,T}
  n_voters = length(voters)
  n_issues = length(vote)
  @assert isapprox(sum(mixture.shares), 1)
  check_issues(vote, n_voters, N)
  categorical = Categorical(mixture.shares) # is it wiser to store `shares` as a `Categorical`?
  ## the following is what I would do if I were to Gibbs-sample from each
  ## realization `sub_interval` at a time, conditioning on the rest
  #outside_interval = vcat(
  #  1:(sub_interval.start-1), (sub_interval.stop+1):N )
  #conditional_mean_scales = [
  #  ( covariance[sub_interval, outside_interval]
  #    * inv(covariance)[outside_interval, outside_interval] )
  #  for covariance in cov.(mixture.cohorts) ]
  #conditional_covariances = [
  #  invcov(cohort)[sub_interval, sub_interval] |> inv
  #  for cohort in mixture.cohorts ]
  #centered_conditional_cohorts = [
  #  VoterCohort{T}(zeros(T, length(sub_interval)), covariance)
  #  for covariance in conditional_covariances ]
  n_threads = nthreads()
  sample = zeros(T, N, n_threads) # allocate an array to act as an intermediary in which to store results before assigning to a Vector of SVectors
  #utility = zeros(Utility{N,T}, n_threads) why did I have this? be careful when removing it (ref: when you see a fence with an unknown purpose...)
  n_failures = zeros(Int, n_threads)
  @threads :static for voter_index in 1:n_voters
    thread_id = threadid()
    this_sample = @view sample[:, thread_id]
    cohort_index = 0
    n_failures[thread_id] += 1 # this shall be deemed a failure until proven otherwise
    utility = zero(Utility{N,T})
    @inbounds for attempt in 1:n_sample_attempts # beyond this we simply give up and accept a suboptimal fate
      cohort_index = rand(categorical)
      cohort = mixture.cohorts[cohort_index]
      rand!(cohort, this_sample) # per the docs, the global RNG is thread-safe as of Julia 1.3 (I've been following the development since at least sophomore year!)
      utility = Utility{N,T}(this_sample)
      offset = 0
      failed = false
      for (issue, interval) in iterate_issues(vote) # better praxis would be to wrap this into a smaller function..
        ranking = issue.choices[voter_index]
        satisfaction = obeys_ranking(utility, ranking, indif, interval)
        if !satisfaction # && break
          failed = true
          break
        end # idea: settle on the one with the FEWEST violations? also try to Gibbs sampler on constrained marginals...
      end
      if !failed
        n_failures[thread_id] -= 1
        break
      end
    end
    realization = VoterRealization{N,M,T}(cohort_index, utility)
    voters[voter_index] = realization
  end
  sum(n_failures)
end


# the "complete data" log-likelihood for a single voter
function voter_log_likelihood(voter::VoterRealization{N,M,T},
    mixture::VoterMixture{N,M,T})::T where {N,M,T}
  cohort = mixture.cohorts[voter.membership]
  loglikelihood(cohort, voter.utility)
end


function iterate_issues(vote::MultiIssueVote)
  offset = 0
  Iterators.map(vote) do issue
    interval = (offset+1):(offset+issue.n_candidates)
    offset += issue.n_candidates # non-overlapping
    (issue, interval)
  end
end


function center_voter_utility(vote::MultiIssueVote, utility::Utility{N,T}
    )::SVector{N,T} where {N,T}
  normalized = @MVector zeros(T, N)
  for (issue, interval) in iterate_issues(vote)
    slice = @view utility[interval]
    center, scatter = mean(slice), std(slice)
    for index in interval
      normalized[index] = (utility[index] - center) / scatter
    end
  end
  normalized
end

function center_cohort(cohort::VoterCohort{N,T,L}) where {N,T,L}
  center = mean(cohort) |> mean
  scatter = cov(cohort) |> eigvals |> maximum
  VoterCohort{N,T,L}(
    (mean(cohort) .- center) ./ sqrt(scatter),
    cov(cohort) ./ scatter )
end