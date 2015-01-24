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


logparams = [ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]; 


% logparams = [  2.20359284,  -0.27877481,   9.59734467,   0.23639181,        -0.21054363,  log(21)        ];
% logparams = [  4.13014314,  -0.28815957,   7.65188285,   1.66995813,  -0.49799054,  log(10)];
% logparams = [0.89704569,  -0.62841292,   7.20992833,   0.93892478,  -0.73770579,  log(25)];
% logparams = log([4, 0.2, 450, 0.5, 1.5, 13]);
% logparams = [1.9, -1.9, 5.9, -0.75, -0.5, log(14)];

% seed = rand+1; 
n = length(flydata);

simuldat = gendata_pop_dyn_eqn(logparams, n);

% plot(simuldat/1000)
subplot(211); plot(flydata/1000); title('true data');
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat/1000)])











%% test ssf-abc

maxiter = 1;

whichmethod = 'ssf_kernel_abc';
% whichmethod = 'rejection_abc';
% whichmethod = 'ssb_abc';

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

for iter = 1 : maxiter
    
    [iter maxiter]
    
    results = run_iteration_blowflydata(whichmethod, opts, iter);
    
%     save results 
    save(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');
    
end
