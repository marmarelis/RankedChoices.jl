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

# `N` is the number of candidates. `M` is the number of mixtures.
# `V` is the number of voters. `R` is the size of the ranking (i.e. number of choices.)

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
