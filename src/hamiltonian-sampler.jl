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


# the following is a timeline of my original research..

# see "The Soft Multivariate Truncated Normal Distribution", circa 2018.
# a Gibbs sampler for the linearly-constrained/truncated multivariate normal,
# without the approximated cutoff, could probably be figured out as well.

# for implementational simplicity, and probably because our dimensionality is
# not high enough to warrant the above, we go with the procedure described in
# "Efficient Sampling Methods for Truncated Multivariate Normal and Student-t
# Distributions Subject to Linear Inequality Constraints" (2015), wherein the
# straightforward Gibbs approach is validated theoretically and fleshed out.

# "Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians" (2014)
# with a Gibbs step in between Hamiltonian iterations to possibly switch
# between mixture components. All components are truncated the same way, and
# the partition function takes a value over the entire mixture---not per
# component; hence some may have much lower total mass than others. Can I
# therefore form a categorical likelihood over normalized component masses,
# and sample the resultant Dirichlet-multinomial (n=1) posterior? I think so.
# See "Gibbs sampling for fitting finite and infinite Gaussian mixture models"
# from 2013. Though it is not directly useful.

using StatsFuns

# p(membership|utility) = p(membership) * p(utility|membership) / (sum over possible memberships)
# used for sampling the mixture, not any parameter posterior.
# this sampler could have trouble mixing if cohort spreads are not wide enough (via large precision_scale and dof?)
function sample_realization_membership(utility::Utility{N,T},
    mixture::VoterMixture{N,M,T})::Int where {N,M,T}
  log_prior = log.(mixture.shares)
  log_likelihood = SVector{M}( logpdf(cohort, utility)
    for cohort in mixture.cohorts )
  energy = log_prior + log_likelihood
  log_probability = energy .- logsumexp(energy) # for numerical stability?
  exp.(log_probability) |> Categorical |> rand
end

# for walls, g is positive but basically zero (for strict inequality constraints,)
# and F is populated with {-1, 0, 1}. I might want a tiny `threshold` somewhere in here.
# further, we must add Sum(f_j^i mu_i) to g_j to account for the centered version. paper was unclear here.
# `walls` could possibly have fewer columns `R`, but we simply pad with zeros
# `sample` is an allocated staging area for the velocity vector.
# fully deterministic..
function simulate_particle_trajectory!(sample::AbstractVector{T},
    utility::Utility{N,T}, mean_utility::SVector{N,T}, inv_cov::SMatrix{N,N,T,L}, # `inv_cov` for this mixture component
    cohort::VoterCohort{N,T,L}, walls::SMatrix{N,N,Int,L}, remaining_time::T,
    collision_limit::Int, small_gap::T=T(1e-5))::Tuple{Utility{N,T}, T} where {N,T,L}
  rand!(cohort, sample) # ^ needs to be like 1e-2 for Float32, although said type appears to be slower for some reason... (casting at runtime?)
  velocity = SVector{N,T}(sample) .- mean_utility
  position = utility
  leftover_time = T(Inf)
  for t in 1:collision_limit
    momentum::SVector = inv_cov * velocity
    alpha = velocity
    beta = position - mean_utility
    #no_collision = iszero.(walls * alpha) .& iszero.(walls * beta) # Eq. 2.24 in the paper, simply check for any nonzero per j
    condition = sqrt.( (walls * alpha).^2 + (walls * beta).^2 )
    shift = walls * mean_utility
    collision_time = T(Inf)
    collision_wall = @SVector zeros(T, N)
    for wall in 1:N
      if condition[wall] < abs(shift[wall]) #no_collision[wall]
        continue
      end
      coefficients = walls[wall, :] # should be an SVector
      tangent = -(coefficients' * alpha) / (coefficients' * beta)
      phase = atan(tangent)
      # cos(t + phase) = 0 for t = pi/2 - phase + n*pi. find smallest n for t>0
      # cos(t + phase) = -shift/condition for t = acos(-shift/condition) - phase + n*pi
      target = -shift[wall] / condition[wall]
      time = acos(target) - phase - 2pi # MAKE SURE we don't start too large, by subtracting an extra 2pi
      while time <= 0 # todo a better closed-form solution?
        time += pi
      end
      if time < collision_time
        collision_time = time
        collision_wall = coefficients
      end
    end
    @assert collision_time > 0
    collision_time *= 1 - small_gap # leave an air gap so that we are for sure not overstepping the boundary
    eval_time = min(remaining_time, collision_time)
    position = mean_utility + alpha*sin(eval_time) + beta*cos(eval_time)
    next_velocity = alpha*cos(eval_time) - beta*sin(eval_time)
    leftover_time = remaining_time - collision_time
    if leftover_time <= 0
      break
    end
    reflection = (
      (collision_wall' * next_velocity) / sum(abs2, collision_wall) )
    reflected_velocity = next_velocity - 2reflection * collision_wall
    # Julia cares not for tail-call optimization, so I unwound into a loop.
    # alas, the recursion was much more elegant.
    velocity = reflected_velocity
    remaining_time = leftover_time
  end
  position, max(leftover_time, 0)
end

# actually, we have an `n_candidates-1` maximum number of active walls per issue,
# so they will never occupy the entire `N` columns. but the impact should be marginal.
function build_walls(::Val{N}, voter_index::Int, vote::MultiIssueVote,
    indifference::Bool)::SMatrix{N,N,Int} where N
  construction = @MMatrix zeros(Int, N, N) # transposed for memory efficiency
  seen = @MVector fill(false, N)
  wall = 1
  offset = 0
  for issue in vote
    ranking = issue.choices[voter_index]
    n_ranked = length(ranking)
    @assert ranking[1] > 0
    last_ranking = ranking[1]
    seen[last_ranking + offset] = true # in case the below iterand skips entirely
    for constraint in 2:n_ranked
      if ranking[constraint] == 0
        continue
      end
      greater = last_ranking
      lesser = ranking[constraint]
      construction[greater+offset, wall] = 1
      construction[lesser+offset, wall] = -1
      seen[greater+offset] = seen[lesser+offset] = true
      last_ranking = lesser
      wall += 1
    end
    if !indifference
      for candidate in 1:issue.n_candidates
        if seen[candidate+offset]
          continue
        end
        construction[last_ranking+offset, wall] = 1
        construction[candidate+offset, wall] = -1
        wall += 1
      end
    end
    offset += issue.n_candidates
  end
  convert(SVector, construction')
end

# use `voters` as MCMC initializations for the Hamiltonian part of the overall,
# hierarchical, Gibbs sampler for the posterior.
# initialize either with a call to `sample_utilities!`, or by zeroing utilities
# across the board. the latter works because we aren't doing strict inequalities. (yet)
# walls do not couple different issues on an absolute scale, so it is important to
# restrain the means by having a large `mean_scale`.
function hamiltonian_sample_utilities!(voters::Vector{VoterRealization{N,M,T}},
    vote::MultiIssueVote, mixture::VoterMixture{N,M,T},
    n_trajectories::Int, collision_limit::Int, indif::Bool=false) where {N,M,T}
  trajectory_length = pi/2 |> T
  n_voters = length(voters)
  n_issues = length(vote)
  @assert isapprox(sum(mixture.shares), 1)
  check_issues(vote, n_voters, N)
  # `inv(cohort.Σ)` does preserve PDMat structure
  inv_covs = [ invcov(cohort) |> SMatrix{N,N,T} for cohort in mixture.cohorts ] # all this comprehension *could* be an @SVector
  n_threads = nthreads()
  sample = zeros(T, N, n_threads) # intermediary for initial velocity
  incompletenesses = zeros(T, n_threads)
  @threads for voter_index in 1:n_voters
    walls = build_walls(Val(N), voter_index, vote, indif)
    thread_id = threadid()
    this_sample = @view sample[:, thread_id]
    voter = voters[voter_index]
    utility = voter.utility
    membership = voter.membership
    for trajectory in 1:n_trajectories
      cohort = mixture.cohorts[membership]
      cohort_mean = mean(cohort) |> SVector{N}
      utility, leftover_time = simulate_particle_trajectory!(this_sample,
        utility, cohort_mean, inv_covs[membership], cohort, walls,
        trajectory_length, collision_limit) # as prescribed by the paper
      membership = sample_realization_membership(utility, mixture)
      incompleteness = leftover_time / trajectory_length # aggregate and average these..
      incompletenesses[thread_id] += incompleteness
    end
    realization = VoterRealization{N,M,T}(membership, utility)
    voters[voter_index] = realization
  end
  incompleteness = sum(incompletenesses) / (n_trajectories * n_voters)
  incompleteness
end

using Random: shuffle!

# is this guaranteed to converge? it should, methinks..
function shoehorn_ranking(utility::Utility{N,T},
    voter_index::Int, vote::MultiIssueVote, indif::Bool)::Utility{N,T} where {N,T}
  malleable = convert(MVector, utility)
  walls = build_walls(Val(N), voter_index, vote, indif)
  search_order = 1:N |> collect
  satisfied = false
  while !satisfied
    satisfied = true
    shuffle!(search_order)
    for wall_index in search_order
      wall = walls[wall_index, :]
      if wall' * malleable < 0 # pair is wrong
        pair = findall( wall .!= 0 )
        @assert length(pair) == 2
        temp = malleable[pair[1]]
        malleable[pair[1]] = malleable[pair[2]]
        malleable[pair[2]] = temp
        satisfied = false
      end
    end
  end
  convert(SVector, malleable)
end
