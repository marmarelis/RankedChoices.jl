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

#using Automa -- for a fast state-machine compiler

# remove duplicate entries, favoring earlier ones
function sanitize(choice::RankedChoice{R},
    n_candidates::Int)::RankedChoice{R} where R
  counts = @MVector zeros(Int, n_candidates)
  updated = MVector(choice)
  for r in 1:R
    c = choice[r]
    if c == 0
      continue
    end
    if counts[c] >= 1
      updated[r] = 0
    end
    counts[c] += 1
  end
  convert(SVector, updated)
end

function Base.vcat(issues::IssueVote{R}...)::IssueVote{R} where R
  choices = reduce(vcat, (issue.choices for issue in issues))
  n_candidates = issues[1].n_candidates
  @assert all( n_candidates == issue.n_candidates for issue in issues )
  IssueVote(choices, n_candidates)
end

function Base.filter(f::Function, issue::IssueVote{R})::IssueVote{R} where R
  IssueVote(filter(f, issue.choices), issue.n_candidates)
end

# generated functions to automatically wrap and unwrap?
function Base.map(f::Function, issue::IssueVote{R})::IssueVote where R
  IssueVote(map(f, issue.choices), issue.n_candidates)
end

function read_prm_file(filename, N, R)
  rankings = RankedChoice{R}[]
  for line in eachline(filename)
    header_content = split(line, "1) ")
    if length(header_content) != 2
      println("Line malformed: $line.")
      continue
    end
    choices = split(header_content[2], ',')
    equal_choices = filter(choices) do choice
      split(choice, '=') |> length > 1
    end
    # no support for tie votes...
    if length(equal_choices) > 0
      println("Can't set candidates equal...")
      continue
    end
    filtered_choices = filter(choices) do choice
      length(choice) > 0 && !startswith(choice, "WI") # write-ins
    end
    candidates = map(filtered_choices) do choice
      candidate = 0
      try
        candidate_string =
          split(split(choice, 'C')[end], '[')[begin]
        candidate = parse(Int, candidate_string)
      catch except
        error("Couldn't parse `$choice`.")
      end
      if candidate < 1 || candidate > N
        error("Candidate number out of range: $candidate.")
      end
      candidate
    end
    n_choices = length(candidates)
    n_choices == 0 && continue
    @assert n_choices <= R
    padded_candidates = vcat(candidates, zeros(Int, R-n_choices))
    push!(rankings, RankedChoice{R}(padded_candidates))
  end
  IssueVote(rankings, N)
end

using CSV, DataFrames

function read_csv(filename::String, candidates::AbstractVector{String})
  matrix = CSV.read(filename, DataFrame, header=false) |> Matrix
  parse_matrix(convert.(String, matrix), candidates)
end

using XLSX

function read_xlsx(filename::String, candidates::AbstractVector{String};
    sheet_index=1, columns::AbstractVector{Int}, first_row=2)
  sheet = XLSX.readxlsx(filename)
  matrix = sheet[sheet_index][:][first_row:end, columns]
  converted_matrix = map(matrix) do v
    v isa Missing ? "N/A" : convert(String, v)
  end # convert.(String, matrix) is such convenient notation
  parse_matrix(converted_matrix, candidates)
end

function parse_matrix(matrix::AbstractMatrix{S},
    candidates::AbstractVector{S}) where S <: AbstractString
  N = length(candidates)
  V, R = size(matrix)
  waste = Set{String}()
  sliced_votes = map(1:R) do rank
    slice = @view matrix[:, rank]
    slice_votes, slice_waste = quantize(slice, candidates)
    union!(waste, slice_waste)
    slice_votes
  end
  votes = zip(sliced_votes...) .|> RankedChoice{R}
  filtered_votes = filter(votes) do vote
    vote[1] != 0 # first entry can't be zero!! todo: ensure all other entries are zero too? if not, we could shift forward...
  end
  IssueVote(filtered_votes, N), waste
end

function quantize(items::AbstractVector{S}, keys::AbstractVector{S}) where S
  waste = Set{S}()
  quanta = map(items) do item
    for (index, key) in enumerate(keys)
      if item == key
        return index
      end
    end
    push!(waste, item)
    return 0
  end
  quanta, waste
end

# empty strings count as nothing
function parse_matrix(matrix::AbstractMatrix{S}) where S <: AbstractString
  candidates = filter(!isempty, unique(matrix)) # `!` operator creates a closure on the function automagically!
  sort!(candidates)
  votes, waste = parse_matrix(matrix, candidates)
  # (candidates, parse_matrix(matrix, candidates)...)
  votes, candidates
end
