##  Copyright 2022 Myrl Marmarelis
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

function scramble_latent_utilities(choice::RankedChoice{R},
    ::Val{N}, spread::T)::RankedChoice{R} where {N,R,T}
  # assumes candidates go from 1:n_candidates
  # throw somewhere in the unit interval? or evenly?
  ceiling = T(1)
  draw_utility = () -> Uniform(T(0), ceiling) |> rand |> T
  utilities = @MVector fill(T(-1), N)
  for rank_index in 1:R
    candidate = choice[rank_index]
    if candidate == 0
      continue
    end
    utility = draw_utility()
    utilities[candidate] = utility
    ceiling = utility
  end
  for leftover in 1:N
    if utilities[leftover] >= 0
      continue
    end
    utility = draw_utility()
    utilities[candidate] = utility
  end
  @show utilities
  # now, finally, we add noise. since candidates further down the line
  # are closer together in the latent space, perturbations are a bigger deal for them.
  for candidate in 1:N
    utilities[candidate] += spread * randn(T)
  end
  ranking = @MVector zeros(Int, N)
  sortperm!(ranking, utilities, order=Base.Reverse)
  ranking[SVector{R}(1:R)] # index by a static vector to keep static
end
