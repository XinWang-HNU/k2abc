% to test indirect inference with score (auxiliary model example) on blowflydata 
% mijung wrote on May 3, 2015

clear all;
clc;
clf;

%% (1) load data
load flydata.mat

seed = 1;
oldRng = rng();
rng(seed);

n = length(flydata);

%% test ssf-abc

maxiter = 1;

%whichmethod = 'kabc_cond_embed';
% whichmethod = 'ssf_kernel_abc';
% whichmethod = 'rejection_abc';
% whichmethod = 'ssb_abc';
whichmethod = 'indirect_score';

opts.num_obs = n;
opts.num_theta_samps = 10000;
opts.num_pseudodata_samps = 4*n;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata; 

for iter = 1 : maxiter
    
    [iter maxiter]

    results = run_iteration_blowflydata(whichmethod, opts, iter);

    save(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');

end

%%


% clear all;
% clf;
% clc;
% 
% load flydata.mat
% n = length(flydata);
% 
% whichmethod =  'ssf_kernel_abc';
% iter = 4; 
% load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'))
% 
% params_ours = results.post_mean(1,:); 
% simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);
% 
% 
% whichmethod = 'kabc_cond_embed';
% iter = 1; 
% load(strcat('blowflydata_', num2str(whichmethod), '_thIter', num2str(iter), '_2.mat'))
% 
% params = results.post_mean; 
% simuldat = gendata_pop_dyn_eqn(params, n);
% 
% params_sl = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
% simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);
% 
% subplot(211); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
% set(gca, 'ylim', [0 max(simuldat/1000) + 1])
% subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat./1000, 'r-'); title('simulated data');
% set(gca, 'ylim', [0 max(simuldat/1000) + 1])
% 
% % compute chosen summary statistics
% s = ss_for_blowflydata(flydata);
% s_ours =  ss_for_blowflydata(simuldat_ours);
% s_kabc = ss_for_blowflydata(simuldat);
% s_sl = ss_for_blowflydata(simuldat_sl);
% 
% mse = @(a) norm(s-a);
% [mse(s) mse(s_ours) mse(s_kabc) mse(s_sl)]
% 


% rng(oldRng);

