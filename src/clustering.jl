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

using LinearAlgebra

# resultant number of clusters may differ from the cohort parametrization in the likelihood
# this feels more salient than a relabeling algorithm using anchor examples... what if the anchors are weird "outliers"?
function count_coincidences!(coincident::AbstractMatrix{Int}, # can be sparse for huge voter bases?
    voters::Vector{VoterRealization{N,M,T}})::Nothing where {N,M,T}
  n_voters = length(voters)
  for first_index in 1:n_voters
    first_cohort = voters[first_index].membership # don't thrash the cache by indexing this in the inner loop
    for second_index in 1:n_voters
      second_cohort = voters[second_index].membership
      if first_cohort == second_cohort
        coincident[second_index, first_index] += 1
      end
    end
  end
end
