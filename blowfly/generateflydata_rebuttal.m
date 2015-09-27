%% generate data from the fly dynamics equation for making our NIPS2015 rebuttal "stronger"!
% mijung wrote on 
% The main purpose of this is to test ABC algorithms when the true generating 
% parameter is known. 

clear all;
clc;

seed = 10;
oldRng = rng();
rng(seed);

seqleng = 180;

% I will set parameters that are close to posterior mean from k2abc (but
% this can be anything really).
params = [29 0.2 260 0.6 0.3 7]; 
% P = params(1);
% delta = params(2);
% N0 = params(3);
% sig_d = params(4);
% sig_p = params(5);
% tau = round(params(6));
simuldat = gendata_pop_dyn_eqn(params, seqleng);

subplot(211); 
plot(simuldat)
subplot(212); 
hist(simuldat)

% Save to the data folder 
desc = ['A Blowfly population trajectory drawn from the posterior mean of k2-abc.', ...
    ' Generator: generateflydata_rebuttal.m'];
funcs = funcs_global();
fpath = funcs.inDataFolder(sprintf('blowfly_simul_s%d.mat', seed));
save(fpath, 'desc', 'params', 'simuldat');

rng(oldRng);
