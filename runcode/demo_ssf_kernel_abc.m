%function [ ] = demo_ssf_kernel_abc( )
%DEMO_SSF_KERNEL_ABC Demonstrate how to use ssf_kernel_abc
seed = 2;
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
op.proposal_dist = @(n)randn(1, n)*sqrt(5);
% Likelihood function handle. func : (theta, n) -> (d'' x n) where theta is one 
% drawn latent vector and n is the number of samples to draw.
% Gaussian likelihood 
op.likelihood_func = likelihood_func;
op.epsilon_list = logspace(-2, 1, 9);
% number of latent variables (i.e., theta) of interest to draw
op.num_latent_draws = 200;
% number of pseudo data to draw e.g., the data drawn from the likelihood function
op.num_pseudo_data = 200;

% width squared.
width2 = meddistance(obs)^2;
op.mmd_kernel = KGaussian(width2);
op.mmd_exponent = 2;

[R, op] = ssf_kernel_abc(obs, op);
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
    stem(R.latent_samples, R.norm_weights(:, ei));
    set(gca, 'fontsize', 16);
    title(sprintf('ep = %.2g ', ep ));
    grid on 

end

superTitle=sprintf('true theta = %.1f, ker = %s, likelihood = %s', true_theta, ...
    op.mmd_kernel.shortSummary(), func2str(op.likelihood_func));
  annotation('textbox', [0 0.9 1 0.1], ...
      'String', superTitle, ...
      'EdgeColor', 'none', ...
      'HorizontalAlignment', 'center', ...
      'FontSize', 18)
hold off

% compute means of theta 

% change seed back 
rng(seed);
%end

