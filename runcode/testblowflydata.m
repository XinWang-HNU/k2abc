% to test alrogithms on blowflydata
% mijung wrote on jan 23, 2015

clear all;
clc;
clf;

%% (1) load data
load ../experiments/flydata.mat

% test gendata code to see if this matches the data
% with relatively accurate params

%%%%%%%%%%%%%%%%%%%%%%%%%
% remember: params are in this order
%%%%%%%%%%%%%%%%%%%%%%%%%
% P = params(1);
% delta = params(2);
% N0 = params(3);
% sig_d = params(4);
% sig_p = params(5);
% tau = parmas(6); 

% %%%%%%%%% this is what Ted's code gives me (best in terms of mse on ss %%%%%%%%
% logparams = [ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]; 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% n = length(flydata);
% 
% simuldat = gendata_pop_dyn_eqn(exp(logparams), n);
% 
% subplot(211); plot(flydata/1000); title('true data');
% subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat./1000, 'r-'); title('simulated data');
% set(gca, 'ylim', [0 max(simuldat/1000)])

%% test ssf-abc

maxiter = 10;

whichmethod = 'ssf_kernel_abc';
% whichmethod = 'rejection_abc';
% whichmethod = 'ssb_abc';

opts.num_obs = n;
opts.num_theta_samps = 10000;
opts.num_pseudodata_samps = 4*n;

% num_pseudodata_samps = n and width2 = meddistance(opts.yobs)^2/4

opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata'; 
width2mat = meddistance(opts.yobs)^2.*logspace(-2,2,maxiter); 

for iter = 1 : maxiter
    
    [iter maxiter]
    opts.width2 = width2mat(iter); 
    
    results = run_iteration_blowflydata(whichmethod, opts, iter);

%     save results 
    save(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');
    
end

%%

whichmethod =  'ssf_kernel_abc';
iter = 4; 
load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'))

params_ours = results.post_mean(1,:); 
simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

params_sl = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);

subplot(211); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data (ours)');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

% compute chosen summary statistics
s = ss_for_blowflydata(dat);
s_ours =  ss_for_blowflydata(simuldat_ours);
s_sl = ss_for_blowflydata(simuldat_sl);

mse = @(a) norm(s-a);
[mse(s) mse(s_ours) mse(s_sl)]





