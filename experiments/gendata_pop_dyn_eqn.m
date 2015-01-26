% population dynamic equation to generate data for blowfly data
% mijung wrote on jan 23, 2015

function dat = gendata_pop_dyn_eqn(params, n)

% inputs
%         (1) logparams: theta in log
%         (2) n: number of observations
% outputs
%         (1) dat

% if nargin < 3
%     seed = 1;
% end
%
% oldRng = rng();
% rng(seed);

% transform logparams to params
% params = exp(logparams);
% params are in this order
P = params(1);
delta = params(2);
N0 = params(3);
sig_d = params(4);
sig_p = params(5);
tau = round(params(6));

if tau==0
    tau = tau +1;
end

lag = tau;

burnin = 50;
dat = zeros(lag+n+burnin,1);

dat(1:lag) = 180*ones(lag,1);


% draw for eps
eps_s = gamrnd(1/(sig_d^2), sig_d^2, 1, n+burnin);

% draw for e
e_s = gamrnd(1/(sig_p^2), sig_p^2, 1, n+burnin);
for i = 1 : n+burnin
    
    t = i + lag ;
    
    %     eps = 1;
    %     e = 1;
    eps = eps_s(i);
    e = e_s(i);
    
    tau_t = t - lag;
    dat(t) = P*dat(tau_t)*exp(-dat(tau_t)/N0)*e + dat(t-1)*exp(-delta*eps);
end

dat = dat(end-n+1:end);
dat = dat';

% rng(oldRng);
