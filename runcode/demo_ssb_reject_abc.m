%function [ ] = demo_ssb_reject_abc( )
%DEMO_SSB_REJECT_ABC Demonstrate how to do a summary statistic-based rejection ABC
seed = 4;
oldRng = rng();
rng(seed);

likelihood_func = @(theta, n)randn(1, n) + theta;
%observations 
true_theta = 3;
num_obs = 200;
obs = likelihood_func(true_theta, num_obs );

% options. All options are described in ssf_kernel_abc.
op = struct();
op.seed = seed;
% A proposal distribution for drawing the latent variables of interest.
% func_handle : n -> (d' x n) where n is the number of samples to draw.
% Return a d' x n matrix.
% 0-mean Gaussian proposal
op.proposal_dist = @(n)randn(1, n)*sqrt(8);
% Likelihood function handle. func : (theta, n) -> (d'' x n) where theta is one 
% drawn latent vector and n is the number of samples to draw.
% Gaussian likelihood 
op.likelihood_func = likelihood_func;
% a function for computing a vector of summary statistics from a set of samples
% func : (d x n) -> p x 1 vector for some p
op.stat_gen_func = @(data) mean(data, 2);

% a function for measuring the distance of two sets of summary statistics 
% of drawn pseudo data and the observations.
% Obs. Depending on the function specified, we can have stat-based / free ABC.
% func: p-vector, p-vector -> distance
% The distance will be fed to the threshold_func
op.stat_dist_func = @(stat1, stat2) norm(stat1 - stat2);

% a function taking a distance from stat_dist_func and an epsilon, 
% and output a weight. If the weight is 0-1, we have a rejection ABC.
% func : (  a-vector of distances, b-vector of epsilons ) -> a x b matrix
op.threshold_func = @(dists, epsilons) bsxfun(@lt, dists(:), epsilons(:)');

% my heuristic for setting epsilons
stat_scale = mean(abs(op.stat_gen_func(obs)));
op.epsilon_list = logspace(-1.5, 0, 9)*stat_scale;
% number of latent variables (i.e., theta) of interest to draw
op.num_latent_draws = 200;
% number of pseudo data to draw e.g., the data drawn from the likelihood function
% for each epsilon
op.num_pseudo_data = 300;


[R, op] = ssb_abc(obs, op);
% R contains latent samples and their weights for each epsilon.
%
figure 
hold on 
cols = 3;
num_eps = length(op.epsilon_list);
for ei = 1:num_eps
    subplot(ceil(num_eps/cols), cols, ei);
    ep = op.epsilon_list(ei);
    % plot empirical distribution of the drawn latent
    stem(R.latent_samples, R.unnorm_weights(:, ei));
    set(gca, 'fontsize', 16);
    title(sprintf('ep = %.2g ', ep ));
    grid on 

end

superTitle=sprintf('true theta = %.1f, likelihood = %s', true_theta, ...
    func2str(op.likelihood_func));
  annotation('textbox', [0 0.9 1 0.1], ...
      'String', superTitle, ...
      'EdgeColor', 'none', ...
      'HorizontalAlignment', 'center', ...
      'FontSize', 18)
hold off


% change seed back 
rng(seed);
%end

