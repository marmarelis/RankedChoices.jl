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

# `N` is the number of candidates, summed up over all issues (should there be more than one issue).
# `M` is the number of mixtures. `V` is the number of voters.
# `R` is the size of the ranking (i.e. number of choices.)

## below, the dangling type parameter `T` would carry over implicitly
##const VoterCohort = MvNormal{PDMat, }
## but partial parametrization is not supported. brackets after `where` is compulsory at `const` expressions.
#const VoterCohort{ N, T, L } =
#  MvNormal{T, PDMat{T, SMatrix{N, N, T, L}}, SVector{N, T}} # N::Int
#
## if `matrix` is an SMatrix, no new allocations (should) occur
#Base.convert(::Type{PDMat{T,SMatrix{N,N,T,L}}}, matrix::AbstractMatrix{T}) where {N,T,L} =
#  PDMat{T,SMatrix{N,N,T,L}}(N, matrix, cholesky(matrix)) # assert L == N*N?
# these things graduated to the level of a massive pain...

const VoterCohort{T} = MvNormal{T, PDMat{T, Matrix{T}}, Vector{T}}
# ^ allow structures and functions that operate thereupon to fully specialize
# careful introspection revealed that this enabled massive performance gains
# by stabilizing the types in critical loops

# nice-to-have: enable cohorts to assign values to more than one set of candidates, i.e. question.
#   instead of sampling for each separately, it would be valuable to force the cohorts to correspond
#   across the two or more questions. alternatively, find post-hoc couplings between independent samples
#   my past attempt was to sample realizations for each issue separately, conditioning on other issues' realizations.
#   how this precisely squares with the cross-issue covariance structure is tenuous.
#   I have a hunch that it extends, and abides fully with my Gibbs scheme.
struct VoterMixture{ N, M, T } # all locally and statically allocated
  cohorts :: SVector{M, VoterCohort{T}}
  shares :: SVector{M, T} # must sum to unit. the composition hereof
end

# ranked choices come in vectors that need not be as long as the total number of candidates.
# in that case, the constraints regarding the unincluded (~not quite excluded~) candidates are
# that their specific utility is less than that of the included ones.
# ordered list of candidate indicies, descending in utility
const RankedChoice{ R } = SVector{ R, Int } #where { R, T <: Real } rather than RankedChoice{ R, T }
# better semantics than the implicity of a matrix?
# strongly typed and explicitly-declared type parameters

struct IssueVote{ R }
  choices :: Vector{RankedChoice{R}}
  n_candidates :: Int # all issue `n_candidates` sum to the cohorts' big `N`
end

# encode in such a way that the entire nested type-structure
# (e.g. each array's heterogeneous `R`) becomes concrete at runtime
const MultiIssueVote = Tuple{Vararg{IssueVote}}

const Utility{ N, T } = SVector{ N, T }

const Membership = Int # if only I could easily parametrize the integer to live within a certain range, and define operations and promotion rules with it

struct VoterRealization{ N, M, T }
  membership :: Membership # Membership{M} ?
  utility :: Utility{N,T}
end

Base.convert(::Type{VoterRealization{N,M,T}},
    params::Tuple{Int, AbstractVector{T}}) where {N,M,T} = # to allow easy array instantiations
  VoterRealization{N,M,T}(params[1], Utility{N,T}(params[2]))

Base.zero(::Type{VoterRealization{N,M,T}}) where {N,M,T} =
  VoterRealization{N,M,T}(0, zero(Utility{N,T}))

function validate_choices(rankings::AbstractVector{RankedChoice{R}}, N::Int)::Bool where R
  all( all( (ranking .>= 1) .& (ranking .<= N) ) for ranking in rankings )
end

## predictive posterior of these voter realizations
# for issue A and B:
# KL(joint(A,B) || single(A) x single(B)) is a sort of mutual information?
# consider this quantity more deeply.
# different from just taking one joint and splitting it because we *may*
# seek to study coupled vs. uncoupled
