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

# The prior tends to constrain the range and scale of the inferred distributions,
# but they can still float around during the Gibbs sampling.
# I think that tighter decisions can be made by looking for metrics that are
# translation- and scale- invariant across all the means of the mixture, at least.
# Metrics that are functions of the covariances can be examined later. It is also
# not entirely necessary to do this.

using Statistics
using Distributions

Statistics.mean(mixture::VoterMixture{N,M,T}) where {N,M,T} =
  sum(zip(mixture.cohorts, mixture.shares)) do params
    cohort, share = params
    means = convert(SVector{N,T}, mean(cohort))
    means .* share
  end

# https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
Statistics.cov(mixture::VoterMixture{N,M,T}) where {N,M,T} =
  let mixture_means = mean(mixture)
    sum(zip(mixture.cohorts, mixture.shares)) do params
      cohort, share = params
      means = convert(SVector{N,T}, mean(cohort))
      covariance = convert(SMatrix{N,N,T}, cov(cohort))
      displacement = means - mixture_means # how much off-center
      cohort_cov = covariance + displacement * displacement'
      cohort_cov .* share
    end
  end

# I know that this works with two blocks of variables, at least
# we can simply average this over monte carlo posteriors but NOT mixture components!
function mutual_information(cohort::VoterCohort{T},
    blocks::AbstractVector{Int}...)::T where T
  @assert length(blocks) > 1
  covariance = cov(cohort)
  numerator = sum(blocks) do block
    @view(covariance[block, block]) |> logdet
  end
  coverage = vcat(blocks...)
  @views denominator = covariance[coverage, coverage] |> logdet #logdetcov(cohort)
  0.5(numerator - denominator)
end

# set a low quantile to find the "safest" candidate, i.e. most moderately agreeable, least polarizing
function estimate_quantiles(mixtures::AbstractVector{VoterMixture{N,M,T}},
    portion::Union{Vector{T},T}, n_draws::Y)::AbstractVector{Utility{N,T}} where {N,M,T,Y<:Real}
  n_quantiles = length(portion)
  quantiles = zeros(T, N, n_quantiles) # align for reinterpretation, even though quantile! will have to skip around
  for candidate in 1:N
    sample = T[]
    remainder = zero(Y)
    for mixture in mixtures
      remainder += n_draws
      n_round_draws = floor(Int, remainder)
      remainder -= convert(Y, n_round_draws)
      for draw in 1:n_round_draws
        utility = draw_marginal(mixture, candidate)
        push!(sample, utility) # does it even help to pre-allocate?
      end
    end
    candidate_quantiles = @view quantiles[candidate, :]
    quantile!(candidate_quantiles, sample, portion)
  end
  reinterpret(Utility{N,T}, vec(quantiles))
end

# these functions are designed to be mapped across a mixture sample as a pipeline

function normalize_utility(utilities::Utility{N,T}, norm_power::T=T(2))::Utility{N,T} where {N,T}
  centered = utilities .- mean(utilities)
  norm = mean(u -> abs(u) ^ norm_power, utilities) ^ (1/norm_power)
  centered ./ norm
end

function estimate_quantiles_old(mixture::VoterMixture{N,M,T}, quantile::T)::Utility{N,T} where {N,M,T}
  # obtain the marginals. covariance matrices are magical in that the diagonal tells you all you need
  # to know about individual variables (following the positive-definite construction)
  overall_quantiles = @SVector zeros(T, N)
  for (share, cohort) in zip(mixture.shares, mixture.cohorts)
    quantiles = marginalize_quantiles(Val(N), cohort, quantile)
    overall_quantiles += quantiles * share
  end
  overall_quantiles
end

function marginalize_quantiles(::Val{N}, cohort::VoterCohort{T}, quantile::T)::Utility{N,T} where {N,T}
  means = mean(cohort) # hope these don't allocate new arrays...
  covariance = cov(cohort)
  SVector{N}( begin # @SVector [...] is more precarious, syntactically
    stdev = covariance[i, i] |> sqrt
    normal_quantile(means[i], stdev, quantile)
  end for i in 1:N )
end

using SpecialFunctions: erfinv

function normal_quantile(mu::T, sigma::T, quantile::T)::T where T <: Real
  mu + sigma*sqrt(2)*erfinv(2quantile - 1)
end
