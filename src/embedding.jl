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


## embedding stuff taken from my IntrinsicGeometry.jl package

function reduce_to_principal_components(data::AbstractMatrix{<:Real}, n_dims::Int)
  eigval, eigvec = data' |> cov |> Hermitian |> eigen
  means = mean(data, dims=2)
  n_dims = min(n_dims, size(data, 1))
  projection = eigvec[:, (end-n_dims+1):end]'
  reduced = projection * (data .- means)
  eigenvalues = eigval[(end-n_dims+1):end]
  (; reduced, projection, means, eigenvalues)
end

function compute_potential_distances(transitions::AbstractMatrix{T},
    gamma::T = T(1))::Matrix{T} where T
  n_points = size(transitions, 1)
  @assert size(transitions, 1) == size(transitions, 2)
  if gamma == 1
    f = (p::T) -> log(max(p, floatmin(T)))
  elseif gamma == -1
    f = (p::T) -> p
  else
    f = (p::T) -> 2/(1-gamma) * p^((1-gamma)/2)
  end
  distances = zeros(T, n_points, n_points)
  log_transitions = f.(transitions)
  for k in 1:n_points
    Threads.@threads for i in 1:n_points
      for j in 1:(i-1)
        difference = log_transitions[i, k] - log_transitions[j, k]
        distances[j, i] += difference ^ 2
      end
    end
  end
  for i in 1:n_points, j in 1:(i-1)
    distances[i, j] = distances[j, i]
  end
  distances .|> sqrt
end

using PyCall, Conda # maybe this shouldn't be necessary to import?

function perform_classic_mds(distances::AbstractMatrix, n_manifold_dims::Int)
  squared = distances .^ 2
  semi_centered = distances .- mean(distances, dims=1)
  centered = semi_centered .- mean(semi_centered, dims=2)
  embedding = reduce_to_principal_components(centered, n_manifold_dims)
  embedding.reduced
end

function perform_metric_mds(distances::AbstractMatrix{T},
    n_manifold_dims::Int)::AbstractMatrix{T} where T
  classic_result = perform_classic_mds(distances, n_manifold_dims)
  skl_manifold = pyimport_conda("sklearn.manifold", "scikit-learn")
  embedding, stress, n_iter = skl_manifold.smacof(distances, metric=true,
    n_components=n_manifold_dims, return_n_iter=true, n_init=1, init=classic_result')
  convert.(T, embedding)
end

function fit_line(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real
  x_bar, y_bar = mean(x), mean(y)
  numerator = sum((x .- x_bar) .* (y .- y_bar))
  denominator = sum((x .- x_bar).^2)
  slope = numerator / denominator # how to represent an infinnite slope? don't: filter for NaNs later
  intercept = y_bar - slope*x_bar
  fit = intercept .+ slope.*x
  error = sum((fit .- y).^2)
  (; slope, intercept, error)
end

# very simple method, which I believe is the same one harnessed by and for PHATE
function find_knee(x::AbstractVector{T}, y::AbstractVector{T})::T where T <: Real
  @assert length(x) == length(y)
  n_points = length(x)
  errors = zeros(T, n_points)
  for (i, pivot) in enumerate(x)
    # do not exclude the pivot for symmetry. we need to sum over the same set of items each time for commensurable error tallies
    left_mask = (x .<= pivot)
    right_mask = .!left_mask
    left_fit = @views fit_line(x[left_mask], y[left_mask])
    right_fit = @views fit_line(x[right_mask], y[right_mask])
    total_error = left_fit.error + right_fit.error
    @debug pivot left_fit right_fit
    errors[i] = total_error
  end
  masked_errors = ifelse.(isfinite.(errors), errors, Inf)
  # possibly disallow lines of length 1 and 2 as there are 2 degrees of freedom
  best_index = argmin(masked_errors[2:end-1]) + 1
  x[best_index]
end

function estimate_von_neumann_entropy(eigval::AbstractVector{T})::T where T <: Real
  positive_eigval = eigval[eigval .> 0]
  normalizer = sum(positive_eigval)
  probabilities = positive_eigval / normalizer
  pointwise_entropies = -probabilities .* log.(probabilities)
  sum(pointwise_entropies)
end

function optimize_von_neumann_entropy(eigval::Vector{T},
    powers::AbstractRange{T})::T where T # <:Real is implied
  n_powers = length(powers)
  x, y = zeros(T, n_powers), zeros(T, n_powers)
  for (i, t) in enumerate(powers)
    step_eigval = eigval.^t
    entropy = estimate_von_neumann_entropy(step_eigval)
    x[i] = t
    y[i] = entropy
  end
  knee = find_knee(x, y)
  #convert(Int, knee) # no InexactErrors here
end

function decompose_diffusion(diffusion::Matrix{T})::Tuple{Vector{T}, Matrix{T}, Matrix{T}} where T
  scales = sum(diffusion, dims=2).^0.5 # symmetric version
  transitions = diffusion ./ (scales .* scales')
  eigval, eigvec = transitions |> Hermitian |> eigen
  eigval, eigvec, scales
end

function embed_diffusion(diffusion::Matrix{T}, n_manifold_dims::Int;
    n_steps::T = T(-1), step_powers::AbstractRange{T} = 1:T(50)
    )::Matrix{T} where T
  eigval, eigvec, scales = decompose_diffusion(diffusion)
  transition = diffusion ./ (scales.^2)
  if n_steps <= 0
    n_steps = optimize_von_neumann_entropy(eigval, step_powers)
    @info "Taking $n_steps diffusion steps."
  end
  distances = compute_potential_distances(
    transition ^ n_steps |> real)
  perform_metric_mds(distances, n_manifold_dims)
end
