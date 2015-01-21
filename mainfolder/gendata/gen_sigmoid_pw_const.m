% generate data from a piecewise sigmoid likelihood function

function dat = gen_sigmoid_pw_const(params, n, seed)

% inputs
%         params: theta
%         n: number of observations
%         seed: 
%
% outputs
%        dat: 
%           (1) dat.samps: samples
%           (2) dat.probs: true probabilities

if nargin < 3
    seed = 1;
end

oldRng = rng();
rng(seed);

% generate data
[ samples, probs ] = like_sigmoid_pw_const( params, n );

% output
dat.samps = samples;
dat.probs = probs;

rng(oldRng);
