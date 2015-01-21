% to generate Figure 1 using a piece constant sigmoid likelihood with three
% dimensional theta.

% mijung wrote on jan 21,2015

% this script will run "run_iteration.m" that takes inputs
% (1) whichmethod: ssf_kernel_abc (ours), rejection_abc, ssb_abc, and ssf_abc.
% (2) opts:
%            opts.likelihood_func: determine likelihood function
%            opts.true_theta: a vector of true parameters
%            opts.num_obs: # of observations (actual observation)
%            opts.num_theta_samps: # of samples for theta
%            opts.num_pseudodata_samps: # of samples for pseudo-data
%            opts.epsilon_list : list of epsilon to test 
%            opts.prior_var: prior variance to draw theta
% (3) seed number

clear all;
clc;
close all;

maxiter = 10;

whichmethod = 'ssf_kernel_abc';

opts.likelihood_func = @like_sigmoid_pw_const;
opts.true_theta =  [1, -3, 2]';
opts.num_obs = 400;
opts.num_theta_samps = 1000;
opts.num_pseudodata_samps = 400;
opts.epsilon_list = logspace(-3, 0, 9);
opts.prior_var = 4; 

for iter = 1 : maxiter
    
    [iter maxiter]
    
    results = run_iteration(whichmethod, opts, iter);
    
    % save results 
    save(strcat('Fig1_method: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');
    
end

%% visualization

num_eps = length(opts.epsilon_list);
cols = length(opts.true_theta);

msemat_probs = zeros(maxiter, num_eps);
est_probs_mat = zeros(num_eps, cols, maxiter);

for iter  =1:maxiter
    load(strcat('Fig1_method: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'));
    
    mse_probs = @(a) sqrt(sum(bsxfun(@minus, a, results.dat.probs').^2, 2)/cols);
    
    est_probs_mat(:, :, iter) =  results.prob_post_mean; 
    
    for ei = 1:num_eps
        msemat_probs(iter, ei) = mse_probs(results.prob_post_mean(ei,:));
    end
    
end

subplot(4,3,1); hist(results.dat.samps); title('400 yobs');  set(gca, 'xlim', [0 3]);
subplot(4,3,2);  bar(results.dat.probs, 'r'); title('true prob'); set(gca, 'ylim', [0 max(results.dat.probs).*1.2]);
% mse plot
subplot(4,3,6);
mean_prob = mean(msemat_probs); 
var_prob = var(msemat_probs);
errorbar(opts.epsilon_list, mean_prob, sqrt(var_prob), '.-'); 
set(gca, 'xscale', 'log', 'xlim', [min(opts.epsilon_list)*0.9 max(opts.epsilon_list)*1.2])

[~, whichepsilonisthebest] = min(mean_prob); 
bestmean = mean(est_probs_mat(whichepsilonisthebest, :, :),3);
subplot(4,3,5); bar(bestmean, 'b');  set(gca, 'ylim', [0 max(results.dat.probs).*1.2]);

dat_sim = like_piecewise_const(bestmean, opts.num_obs); 
subplot(4,3,4); hist(dat_sim); title('400 simulated');  set(gca, 'xlim', [0 3]);



