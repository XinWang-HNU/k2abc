%% generate data from the fly dynamics equation for making our rebuttal "stronger"!
% mijung wrote on 

clear all;
clc;

seed = 10;
oldRng = rng();
rng(seed);

seqleng = 100;

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

subplot(211); plot(1:seqleng, simuldat)
subplot(212); hist(simuldat)

save simuldat simuldat
save params params

rng(oldRng);