% population dynamic equation to generate data for blowfly data
% mijung wrote on jan 23, 2015

function dat = gendata_pop_dyn_eqn(logparams, n)

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
params = exp(logparams);
% params are in this order
P = params(1);
delta = params(2);
N0 = params(3);
sig_d = params(4);
sig_p = params(5);
tau = round(params(6));

lag = tau;

burnin = 50;
dat = zeros(lag+n+burnin,1);

dat(1:lag) = 180*ones(lag,1);

for i = 1 : n+burnin
    
    t = i + lag ;
    
    %     eps = 1;
    %     e = 1;
    
    % draw for eps
    eps = gamrnd(1/(sig_d^2), sig_d^2);
    
    % draw for e
    e = gamrnd(1/(sig_p^2), sig_p^2);
    
    tau_t = t - lag;
    dat(t) = P*dat(tau_t)*exp(-dat(tau_t)/N0)*e + dat(t-1)*exp(-delta*eps);
end

dat = dat(end-n+1:end);

% rng(oldRng);