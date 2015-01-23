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
% clf;
% close all;

maxiter = 1;

whichmethod = 'ssf_kernel_abc';
% whichmethod = 'rejection_abc';
% whichmethod = 'ssb_abc';
% whichmethod = 'ssf_abc';

% generate true theta
theta_before_trs = [1, -2,  3, -2, 4]';
sig = 1./(1+exp(-theta_before_trs));
norm_sig = sig/sum(sig);
true_theta = norm_sig;

opts.likelihood_func = 'like_sigmoid_pw_const';
opts.true_theta =  true_theta;
opts.num_obs = 400;
opts.num_theta_samps = 1000;
opts.num_pseudodata_samps = 400;
% opts.prior_var = 4; 

%%
for iter = 1 : maxiter
    
    [iter maxiter]
    
    results = run_iteration(whichmethod, opts, iter);
    
%     save results 
    save(strcat('Fig1_method: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');
    
end

%%

% visualization
% clf;

load(strcat('Fig1_method: ', num2str(whichmethod), '_thIter', num2str(1), '.mat'));
num_eps = length(results.epsilon_list);
cols = length(opts.true_theta);

msemat_probs = zeros(maxiter, num_eps);
est_probs_mat = zeros(num_eps, cols, maxiter);
est_probs_var_mat = zeros(num_eps, cols, maxiter);

for iter  =1:maxiter
    
    load(strcat('Fig1_method: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'));
    
%     mse_probs = @(a) sqrt(sum(bsxfun(@minus, a, results.dat.probs').^2, 2)/cols);
    mse_probs = @(a) sqrt(sum(bsxfun(@minus, a, true_theta').^2, 2)/cols);
 
%     est_probs_mat(:, :, iter) =  results.prob_post_mean; 

    est_probs_mat(:, :, iter) =  results.post_mean;        
    est_probs_var_mat(:, :, iter) = results.post_var;        
    
    for ei = 1:num_eps
        msemat_probs(iter, ei) = mse_probs(results.post_mean(ei,:));
    end
    
end

histy = hist(results.dat.samps, length(opts.true_theta));

subplot(3,4,1);  bar(results.dat.probs, 'k'); title('true prob'); set(gca, 'ylim', [0 max(results.dat.probs).*1.2],  'xlim', [0.5 length(opts.true_theta)+.5]); box off; 
subplot(3,4,5);  hist(results.dat.samps, length(opts.true_theta));  hold on; 
meanyobs = mean(results.dat.samps); 
plot(meanyobs, 0:max(histy), 'r'); 
stdy = std(results.dat.samps);
plot(meanyobs:0.1:meanyobs+stdy, 50*ones(length(meanyobs:0.1:meanyobs+stdy),1), 'r');
plot(meanyobs-stdy:0.1:meanyobs, 50*ones(length(meanyobs:0.1:meanyobs+stdy),1), 'r');
set(gca, 'xlim', [-0.1 length(opts.true_theta)+.1], 'ylim', [0 max(histy)+10]);box off; 

% mean_prob = mean(msemat_probs); 
% var_prob = var(msemat_probs);
% [~, whichepsilonisthebest] = min(mean_prob); 
% bestmean = mean(est_probs_mat(whichepsilonisthebest, :, :),3);
[~, whichepsilonisthebest] = min(msemat_probs); 
bestmean = est_probs_mat(whichepsilonisthebest, :);
dat_sim = like_piecewise_const(bestmean, opts.num_obs); 

% ours (4,5,6), rejction (7 8 9), 
if strcmp(num2str(whichmethod),'ssf_kernel_abc')
    idx_strt = 2;
elseif strcmp(num2str(whichmethod),'rejection_abc')
    idx_strt = 3;
elseif strcmp(num2str(whichmethod),'ssb_abc')
    idx_strt = 4;
end

subplot(3,4,idx_strt+4); hist(dat_sim,  length(opts.true_theta)); 
hold on; 
meansim = mean(dat_sim); 
plot(meansim, 0:max(histy), 'r'); 
stdsim = std(dat_sim);
plot(meansim:0.1:meansim+stdsim, 50*ones(length(meansim:0.1:meansim+stdsim),1), 'r')
plot(meansim-stdsim:0.1:meansim, 50*ones(length(meansim:0.1:meansim+stdsim),1), 'r')
set(gca, 'xlim', [-0.1 length(opts.true_theta)+.1], 'ylim', [0 max(histy)+10]);box off; 


subplot(3,4,idx_strt); bar(bestmean, 'b');  set(gca, 'ylim', [0 max(results.dat.probs).*1.2],  'xlim', [.5 length(opts.true_theta)+.5]); title('est prob'); box off; 
subplot(3,4,idx_strt+8); plot(results.epsilon_list, msemat_probs, 'o-');
% errorbar(results.epsilon_list, mean_prob, sqrt(var_prob), 'o-');
set(gca, 'xscale', 'log', 'xlim', [min(results.epsilon_list)*0.6 max(results.epsilon_list)*1.4], 'ylim', [0 0.2], 'xtick', [results.epsilon_list(1) results.epsilon_list(5) results.epsilon_list(end)]); 
ylabel('error on prob'); box off; 
% xlabel('epsilon')


