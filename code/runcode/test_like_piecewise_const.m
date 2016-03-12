% Test our K2ABC on the piecewise constant likelihood = mixture of uniforms.
%@author Wittawat
seed = 4;
oldRng = rng();
rng(seed);

likelihood_func = @like_piecewise_const;
%observations 
%true_theta = [1, -1, 1]';
true_theta = [0.1, 0.4, 0.4, 0.1]';
num_obs = 500;
obs = likelihood_func(true_theta, num_obs );


% options. All options are described in ssf_kernel_abc.
op = struct();
op.seed = seed;
% A proposal distribution for drawing the latent variables of interest.
% func_handle : n -> (d' x n) where n is the number of samples to draw.
% Return a d' x n matrix.
% 0-mean Gaussian proposal
%op.proposal_dist = @(n)randn(1, n)*sqrt(3);
%op.proposal_dist = @(n)randn(3, n)*sqrt(5);
op.proposal_dist = @(n) sample_from_dirichlet(ones(length(true_theta), 1), n);

% Likelihood function handle. func : (theta, n) -> (d'' x n) where theta is one 
% drawn latent vector and n is the number of samples to draw.
% Gaussian likelihood 
op.likelihood_func = likelihood_func;
op.epsilon_list = logspace(-3, 0, 9);
% number of latent variables (i.e., theta) of interest to draw
op.num_latent_draws = 500;
% number of pseudo data to draw e.g., the data drawn from the likelihood function
% for each theta
op.num_pseudo_data = 500;

% width squared.
width2 = meddistance(obs)^2;
op.mmd_kernel = KGaussian(width2);

[R, op] = k2abc_lin(obs, op);
% R contains latent samples and their weights for each epsilon.

figure 
hold on 
cols = 3;
num_eps = length(op.epsilon_list);
for ei = 1:num_eps
    subplot(ceil(num_eps/cols), cols, ei);
    ep = op.epsilon_list(ei);
    % plot empirical distribution of the drawn latent
    %stem(R.latent_samples, R.norm_weights(:, ei));

    % plot mean probability vectors
    mean_theta = R.latent_samples*R.norm_weights(:, ei) ;
    est_probs = mean_theta;

    bar(est_probs);
    set(gca, 'fontsize', 16);
    title(sprintf('ep = %.2g ', ep ));
    grid on 

end

superTitle=sprintf('true probs = [%s], ker = %s, likelihood = %s', num2str(true_theta'), ...
    op.mmd_kernel.shortSummary(), func2str(op.likelihood_func));
  annotation('textbox', [0 0.9 1 0.1], ...
      'String', superTitle, ...
      'EdgeColor', 'none', ...
      'HorizontalAlignment', 'center', ...
      'FontSize', 18)
hold off

% compute means of theta 

% change seed back 
% rng(seed);
rng(oldRng);
%end

