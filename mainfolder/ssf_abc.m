function [ results, op ] = ssf_abc( Obs, op )
%SSF_ABC Generic function for summary statistic-free ABC which relies on a 
%measure on observations and pseudo data. The measure operates on two sets.
%   - This function supports both rejection and soft ABC.
%
% @author Wittawat
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

% list of epsilon's to run. 
default_eps_list = logspace(-2, 0, 4);
op.epsilon_list = myProcessOptions(op, 'epsilon_list', default_eps_list);
epsilon_list = op.epsilon_list;

% a function for measuring the distance of drawn pseudo data and the observations
% Obs. Depending on the function specified, we can have stat-based / free ABC.
%
% func : (d x n1) , (d x n2)  -> a distance
% The distance will be fed to the threshold_func
%
if isOptionEmpty(op, 'pseudo_data_measure')
    error('pseudo_data_measure cannot be empty');
end
pseudo_data_measure = op.pseudo_data_measure;
assert(isa(pseudo_data_measure, 'function_handle'), ...
    'pseudo_data_measure must be a function handle.');

% a function taking a distance from pseudo_data_measure and an epsilon, 
% and output a weight. If the weight is 0-1, we have a rejection ABC.
% func : (  a-vector of distances, b-vector of epsilons ) -> a x b matrix
if isOptionEmpty(op, 'threshold_func')
    error('threshold_func cannot be empty');
end
threshold_func = op.threshold_func;
assert(isa(threshold_func, 'function_handle'));

% number of latent variables (i.e., theta) of interest to draw
op.num_latent_draws = myProcessOptions(op, 'num_latent_draws', 200);
num_latent_draws = op.num_latent_draws;

% number of pseudo data to draw e.g., the data drawn from the likelihood function
% for each drawn latent variable.
op.num_pseudo_data = myProcessOptions(op, 'num_pseudo_data', 300);
num_pseudo_data = op.num_pseudo_data;

% draw latent variables once 
latent_samples = proposal_dist(num_latent_draws);
assert(size(latent_samples, 2) == num_latent_draws, ...
    'proposal_dist does not return a correct number of samples');

% Distance between sets of pseudo_data and Obs for each sampled latent variable
dists = zeros(num_latent_draws, 1);
% draw pseudo_data once 
for j=1:num_latent_draws
    
    % visualization of sample number
%     if rem(j, 100)==0 % every 100th number will be shown 
%         [j num_latent_draws]
%     end
    
    latent_j = latent_samples(:, j);
    Pseudo_j = likelihood_func(latent_j, num_pseudo_data);
    assert(size(Pseudo_j, 2)==num_pseudo_data, ...
        'likelihood_func does not return a correct number of pseudo_data');
    assert(size(Pseudo_j, 1) == size(Obs, 1), ...
        'Sampled pseudo_data dimension does not match that of observations');
    % call distance measure function
    dists(j) = pseudo_data_measure(Pseudo_j, Obs );
end

% num_latent_draws x length(epsilon_list)
unnorm_weights = threshold_func(dists, epsilon_list);

% a struct containing all results
results = struct();
results.latent_samples = latent_samples;
results.epsilon_list = epsilon_list;
results.unnorm_weights = unnorm_weights;

% change seed back 
rng(oldRng);


end

