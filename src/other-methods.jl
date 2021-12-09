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

# must keep the `R` specification to enforce consistent sizes across vector entries
function count_simple_plurality(choices::Vector{RankedChoice{R}})::Vector{Int} where R
  first_counts = countmap(choice[1] for choice in choices)
  count_keys = keys(first_counts)
  candidate_range = 1 : maximum(count_keys) # disregard zero
  [first_counts[index] for index in candidate_range]
end

# todo some actual Condorcet (using all preference info, rather than first) methods. beats(..., ...) would be a cool method name

# knock one out at a time; avoid the complexity of the shortcut to knock out as many from the bottom that add up to less than the next up
function find_instant_runoff_winner(choices::Vector{RankedChoice{R}})::Int where R
  banned = Set{Int}()
  tally = Dict{Int, Int}()
  while length(tally) != 1
    current_votes = Iterators.map(choices) do choice
      for candidate in choice
        if candidate in banned
          continue
        end
        if candidate == 0
          continue
        end
        return candidate
      end
      return 0
    end
    tally = countmap(current_votes)
    delete!(tally, 0)
    min_count, min_candidate = findmin(tally)
    push!(banned, min_candidate)
  end
  tally |> keys |> first
end


function find_condorcet_winner(choices::AbstractVector{RankedChoice{R}},
    allow_ties::Bool = false)::Int where R
  N = maximum(reinterpret(Int, choices))
  differences = matrix_of_differences(N, choices)
  for possibility in 1:N
    winner = true
    for alternative in 1:N
      if alternative == possibility
        continue
      end
      operator = allow_ties ? (<) : (<=)
      if operator(differences[possibility, alternative], 0)
        winner = false
        break
      end
    end
    winner && return possibility # ever care about the chance for more than one
  end
end

# here zero signifies a tie, with indiscernible votes having no influence
matrix_of_differences(N::Int, choices::AbstractVector{RankedChoice{R}}) where R = [ # Val{N} ?
  sum(choices) do choice
    winner = find_pairwise_winner(choice, row, col)
    (winner == row ? +1 : (winner == col ? -1 : 0))
  end
for row in 1:N, col in 1:N ]

function find_pairwise_winner(choice::RankedChoice,
    first_candidate::Int, second_candidate::Int)::Int
  if first_candidate == second_candidate
    return 0
  end
  # for our current purposes, it suffices to find the candidate that appears first in the ranked preferences
  for chosen_candidate in choice
    if chosen_candidate == first_candidate # could use short-circuit shorthand notation
      return first_candidate
    elseif chosen_candidate == second_candidate
      return second_candidate
    end
  end
  return 0 # zero signifies indiscernible with the information available
end
