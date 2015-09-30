%function [ ] = demo_k2abc_lin( )
%DEMO_K2ABC_LIN Demonstrate how to use K2ABC with linear MMD. Also compare 
%the result to the quadratic MMD version.
%
%@date 27 Sep 2015
%@author Wittawat
%

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
op.seed = seed + 1;
% A proposal distribution for drawing the latent variables of interest.
% func_handle : n -> (d' x n) where n is the number of samples to draw.
% Return a d' x n matrix.
% 0-mean Gaussian proposal
op.proposal_dist = @(n)randn(1, n)*sqrt(8);
% Likelihood function handle. func : (theta, n) -> (d'' x n) where theta is one 
% drawn latent vector and n is the number of samples to draw.
% Gaussian likelihood 
op.likelihood_func = likelihood_func;
op.epsilon_list = logspace(-3, 0, 9);
% number of latent variables (i.e., theta) of interest to draw
op.num_latent_draws = 800;
% number of pseudo data to draw e.g., the data drawn from the likelihood function
% for each theta
op.num_pseudo_data = 300;

% width squared.
width2 = meddistance(obs)^2;
op.mmd_kernel = KGaussian(width2);
op.mmd_exponent = 2;

[Rlin, op] = k2abc_lin(obs, op);
%[~, Ilin] = sort(Rlin.latent_samples);
% different seed just so that the sequence of proposal is different. Easy to see 
% in the plot.
op.seed = op.seed+2;
[R, op] = k2abc(obs, op);
%[~, I] = sort(R.latent_samples);
% Rlin contains latent samples and their weights for each epsilon.
%
figure 
cols = 3;
num_eps = length(op.epsilon_list);
for ei = 1:num_eps
    subplot(ceil(num_eps/cols), cols, ei);
    ep = op.epsilon_list(ei);
    % plot empirical distribution of the drawn latent
    hold on 
    %plot(Rlin.latent_samples(Ilin), Rlin.norm_weights(Ilin, ei), '-b');
    %plot(R.latent_samples(I), R.norm_weights(I, ei), '-r');
    stem(R.latent_samples, R.norm_weights(:, ei), 'r', 'Marker', 'none');
    stem(Rlin.latent_samples, Rlin.norm_weights(:, ei), 'b', 'Marker', 'none');

    set(gca, 'fontsize', 16);
    title(sprintf('ep = %.2g ', ep ));
    xlim([true_theta-3, true_theta+3]);
    ylim([0, 0.04]);
    grid on 

    hold off
end
legend('K2ABC-quad', 'K2ABC-lin')

superTitle=sprintf('Approx. Posterior. true theta = %.1f, ker = %s, likelihood = %s', true_theta, ...
    op.mmd_kernel.shortSummary(), func2str(op.likelihood_func));
  annotation('textbox', [0 0.9 1 0.1], ...
      'String', superTitle, ...
      'EdgeColor', 'none', ...
      'HorizontalAlignment', 'center', ...
      'FontSize', 16)

% compute means of theta 

% change seed back 
rng(oldRng);
%end



%end

