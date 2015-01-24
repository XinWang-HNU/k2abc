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

%%%%%%%%% this is what Ted's code gives me (best in terms of mse on ss %%%%%%%%
% logparams = [ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(flydata);

% simuldat = gendata_pop_dyn_eqn(logparams, n);

% subplot(211); plot(flydata/1000); title('true data');
% subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat./1000, 'r-'); title('simulated data');
% set(gca, 'ylim', [0 max(simuldat/1000)])

%% test ssf-abc

maxiter = 1;

whichmethod = 'ssf_kernel_abc';
% whichmethod = 'rejection_abc';
% whichmethod = 'ssb_abc';

opts.num_obs = n;
opts.num_theta_samps = 1000;
opts.num_pseudodata_samps = 400;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata'; 

for iter = 1 : maxiter
    
    [iter maxiter]
    
    results = run_iteration_blowflydata(whichmethod, opts, iter);
    
%     save results 
    save(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');
    
end
