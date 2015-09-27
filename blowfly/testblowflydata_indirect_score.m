% to test indirect inference with score (auxiliary model example) on blowflydata 
% mijung wrote on May 3, 2015
%
% Wittawat modified on Aug 7, 2015 to make it work without using run_iteration_blowflydata()
% which does not support indirect_score_abc as used by Mijung.


%% (1) load data
%load flydata.mat
data = load('blowfly_simul_s10');
yobs = data.simuldat;
flydata = yobs;

oldRng = rng();

n = length(flydata);
maxiter = 1;


opts.num_obs = n;
opts.num_theta_samps = 10000;
opts.num_pseudodata_samps = 4*n;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata; 

op = opts;
op.likelihood_func = @ gendata_pop_dyn_eqn; 
op.proposal_dist = @(n) sample_from_prior_blowflydata(n); 
op.num_latent_draws = opts.num_theta_samps;
op.num_pseudo_data = opts.num_pseudodata_samps;
op.dim_theta = opts.dim_theta; 
% The number of Gaussian mixture components. This may need some tuning ?
op.numComp = 3;

funcs = funcs_global();

for iter = 1:maxiter
    
    seed = iter;
    rng(seed);

    [iter maxiter]
    [results, op] = indirect_score_abc(yobs, op);
    latent_samples = results.latent_samples;
    fname = sprintf('testblowfly_is-s%d_ntheta%d', iter, opts.num_theta_samps);
    fpath = funcs.runcodeSavedFile(fname);
    timestamp = clock();
    
    save(fpath, 'timestamp', 'op', 'latent_samples', 'seed');

end

rng(oldRng);
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



