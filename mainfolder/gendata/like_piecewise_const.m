function [ samples ] = like_piecewise_const( probs, n )
%LIKE_PIECEWISE_CONST Generate n samples from a piecewise constant distribution 
%defined on an interval [0, n]. The height of the bin [i-1, i] is specified by 
%probs(i). Require sum(probs) = 1.
%

assert(all(probs) >= 0, 'probs entries must be >= 0');
assert(all(probs) <= 1, 'probs entries must be <= 1');
assert( abs(sum(probs) - 1) <= 1e-8, 'sum of probs must be 1.' );

%num_bins = length(probs);
%bins = randsample(1:num_bins, n, true, probs);

% sampling bin indices
bins = discrete_rnd(probs(:), 1, n);
samples = rand(1, n) + bins - 1;

end

