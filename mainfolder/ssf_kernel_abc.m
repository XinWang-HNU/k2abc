function [ results, op ] = ssf_kernel_abc( Obs, op )
%SSF_KERNEL_ABC A function for sufficient statistic free kernel ABC.
%   - Will try many epsilon's (soft rejection threshold) in the candidate list.
%
% Input:
%   - Obs : D x M where D is dimension and M is the number of observations
%   - op : a struct specifying options. See the code for possible options.
%

% random seed. 1 by default.
op.seed = myProcessOptions(op, 'seed', 1);
oldRng = rng();
rng(op.seed);

% A proposal distribution for drawing the latent variables of interest.
% func_handle : n -> (d' x n) where n is the number of samples to draw.
% Return a d' x n matrix.
if isOptionEmpty(op, 'proposal_dist')
    error('proposal_dist cannot be empty.');
end
proposal_dist = op.proposal_dist;
assert(isa(proposal_dist, 'function_handle'), 'proposal_dist must be a function handle.');

% Likelihood function handle. func : (theta, n) -> (d'' x n) where theta is one 
% drawn latent vector and n is the number of samples to draw.
% Cannot be empty.
if isOptionEmpty(op, 'likelihood_func')
    error('likelihood_func cannot be empty.');
end
assert(isa(op.likelihood_func, 'function_handle'), ...
    'likelihood_func must be a function handle');
likelihood_func = op.likelihood_func;

% list of epsilon's to run. Use as exp(-MMD(Obs, pseudo_data)^mmd_exponent / epsilon )
default_eps_list = logspace(-2, 0, 4);
op.epsilon_list = myProcessOptions(op, 'epsilon_list', default_eps_list);
epsilon_list = op.epsilon_list;

% number of latent variables (i.e., theta) of interest to draw
op.num_latent_draws = myProcessOptions(op, 'num_latent_draws', 200);
num_latent_draws = op.num_latent_draws;

% number of pseudo data to draw e.g., the data drawn from the likelihood function
% for each drawn latent variable.
op.num_pseudo_data = myProcessOptions(op, 'num_pseudo_data', 300);
num_pseudo_data = op.num_pseudo_data;

% kernel function for computing MMD between Obs and the drawn pseudo data.
% Default to a Gaussian kernel with median heuristic.
op.mmd_kernel = myProcessOptions(op, 'mmd_kernel', KGaussian(meddistance(Obs)^2)); 
mmd_kernel = op.mmd_kernel;

% The power to which MMD is raised. Default to 2.
op.mmd_exponent = myProcessOptions(op, 'mmd_exponent', 2);
mmd_exponent = op.mmd_exponent;
assert(op.mmd_exponent > 0 && op.mmd_exponent <= 2, 'mmd_exponent must be in (0, 2].')


%% run our ABC code %%
% draw latent variables once 
latent_samples = proposal_dist(num_latent_draws);
assert(size(latent_samples, 2) == num_latent_draws, ...
    'proposal_dist does not return a correct number of samples');

% raw MMDs between sets of pseudo_data and Obs for each sampled latent variable
mmds = zeros(num_latent_draws, 1);
% draw pseudo_data once 
for j=1:num_latent_draws
    latent_j = latent_samples(:, j);
    Pseudo_j = likelihood_func(latent_j, num_pseudo_data);
    assert(size(Pseudo_j, 2)==num_pseudo_data, ...
        'likelihood_func does not returna correct number of pseudo_data');
    assert(size(Pseudo_j, 1) == size(Obs, 1), ...
        'Sampled pseudo_data dimension does not match that of observations');
    % call mmd function
    mmds(j) = mmd(Pseudo_j, Obs, mmd_kernel);
end

% num_latent_draws x length(epsilon_list)
unnorm_weights = exp(-(mmds.^mmd_exponent)*(1./epsilon_list) );
% normalized weights
norm_weights = bsxfun(@rdivide, unnorm_weights, sum(unnorm_weights, 1) );

% a struct containing all results
results = struct();
results.latent_samples = latent_samples;
results.epsilon_list = epsilon_list;
results.unnorm_weights = unnorm_weights;
results.norm_weights = norm_weights;


% change seed back 
rng(oldRng);

end

