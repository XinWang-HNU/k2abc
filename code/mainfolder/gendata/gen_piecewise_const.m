function  dat = gen_piecewise_const( params, n, seed )
%GEN_PIECEWISE_CONST Generate dataset drawn from piecewise constant distribution 
%(equivalently mixture of 5 uniform distributions).
%
% inputs
%         params: theta, a probability vector.
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
probs = params;

[ samples ] = like_piecewise_const( probs, n );

% output
dat.samps = samples;
dat.probs = probs;

rng(oldRng);

end

