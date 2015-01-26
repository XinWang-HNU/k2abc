% to run several iterations to report MSE on chosen summary statistics
% mijung wrote on jan 26, 2015

clear all;
clc;
clf;

num_sampls_tot = 10.^[1:5]; 

%% (1) load data
load ../experiments/flydata.mat

n = length(flydata);

%% test ssf-abc

maxiter = length(num_sampls_tot);

whichmethod = 'ssf_kernel_abc';
% whichmethod = 'rejection_abc';
% whichmethod = 'ssb_abc';

opts.num_obs = n;
opts.num_pseudodata_samps = 4*n;

% num_pseudodata_samps = n and width2 = meddistance(opts.yobs)^2/4

opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata'; 
width2mat = meddistance(opts.yobs)^2.*logspace(-2,2,maxiter); 
opts.width2 = width2mat(4);

for iter = 1 : maxiter
    
    [iter maxiter]
    opts.num_theta_samps = num_sampls_tot(iter);
    
    results = run_iteration_blowflydata(whichmethod, opts, iter);

%     save results 
    save(strcat('blowflydata: ', num2str(whichmethod), 'num_samps', num2str(opts.num_theta_samps), '.mat'), 'results');
    
end

%%

whichmethod =  'ssf_kernel_abc';
iter = 4; 
% load(strcat('blowflydata: ', num2str(whichmethod), 'num_samps', num2str(opts.num_theta_samps), '.mat'))
load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'))

params_ours = results.post_mean(1,:); 
simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

params_sl = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);

subplot(211); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data (ours)');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])
