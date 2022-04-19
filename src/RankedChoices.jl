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

module RankedChoices

include("structures.jl")
include("sampler.jl")
include("hamiltonian-sampler.jl")
include("clustering.jl")
include("simulator.jl")
include("analysis.jl")
include("other-methods.jl")
include("parser.jl")
include("scrambler.jl")

export VoterRealization, VoterCohort, VoterMixture, Membership, Utility, RankedChoice
export IssueVote, MultiIssueVote
export validate_choices, sample_mixture_shares, sample_mixture_posterior
export obeys_ranking, shoehorn_ranking
export rejection_sample_utilities!, hamiltonian_sample_utilities!
export Prior, make_impartial_prior, simulate, HamiltonianSim, RejectionSim
export normalize_utility, estimate_quantiles, mutual_information
export count_simple_plurality, find_instant_runoff_winner, find_condorcet_winner
export read_prm_file, read_csv, read_xlsx, parse_matrix, sanitize
export scramble_latent_utilities, sample_possible_rankings

end # module
