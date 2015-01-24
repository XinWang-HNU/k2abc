%function [ ] = demo_kabc_cond_embed( )
%DEMO_KABC_COND_EMBED Demonstrate how to use kabc_cond_embed
%@author Wittawat
%
seed = 3;
oldRng = rng();
rng(seed);

% The following functions are not needed by kabc_cond_embed.m
% Just for data construction.
%
% Likelihood function handle.
%  Gaussian likelihood as an example
%likelihood_func = @(theta, n)randn(1, n) + theta;
% 
% Exponential likelihood
likelihood_func = @(theta, n)exprnd(theta, 1, n);

% A proposal distribution for drawing the latent variables of interest.
%proposal_dist = @(n)randn(1, n)*sqrt(8);
%
% uniform 
proposal_dist = @(n)unifrnd(0.1, 10, 1, n);
% a function for computing a vector of summary statistics from a set of samples
% func : (d x n) -> p x 1 vector for some p
stat_gen_func = @(data) mean(data, 2);

% kabc needs a training set containing (summary stat, parameter) pairs.
% construct a training set
num_latent_draws = 400; % this is also the training size
num_pseudo_data = 200;
train_params = proposal_dist(num_latent_draws);
train_stats = zeros(1, num_latent_draws);
% for each single parameter, we need to construct a summary statistic of 
% observations generated by the parameter.
for i=1:size(train_params, 2)
    theta = train_params(:, i);
    observations = likelihood_func(theta, num_pseudo_data);
    stat = stat_gen_func(observations);
    % say we also have some zero-mean noise in our stat.
    train_stats(:, i) = stat + randn(1);
end


% ------- options for kabc ------------
% All options are described in kabc_cond_embed
op = struct();
op.seed = seed;

% a list of regularization parameter candidates in kabc. 
% Chosen by cross validation.
ntr = num_latent_draws;
op.kabc_reg_list = 10.^(-4:2:3)/sqrt(ntr);

% number of folds to perform in cross validation
op.kabc_cv_fold = 3;

% a list of Gaussian widths squared to be used as candidates for Gaussian kernel
op.kabc_gwidth2_list = [1/8, 1/4, 1, 2].* (meddistance(train_stats).^2)

% ---- training ------
[R, op] = kabc_cond_embed(train_stats, train_params, op);
% R contains a regression function mapping from a stat to its param
%

% ---- test phase --------------
% generate some observations
% Try to play with true_theta and check the result.
true_theta = 4;
num_obs = 200;
obs = likelihood_func(true_theta, num_obs );
observed_stat = stat_gen_func(obs);
% prediction of theta weights
% Use regression function mapping from a stat to its param
weights_func = R.regress_weights_func;
W = weights_func(observed_stat);

% plot weights 
figure 
hold on
stem(train_stats, W);
set(gca, 'fontsize', 16);
title(sprintf('true theta: %.2g, Observed stat: %.2g, likelihood = %s', ...
    true_theta, observed_stat, ...
    func2str(likelihood_func) ));
grid on
hold off 


% change seed back 
rng(oldRng);
%end

