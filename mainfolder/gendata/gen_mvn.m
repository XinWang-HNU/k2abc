% generate data from a multivariate normal distribution 

function dat = gen_mvn(mu, sig, howmanysamps, seed)

% inputs
%         mu: mean
%         sig: covariance
%         howmanysamps: how many samples to generate
%
% outputs
%        dat: data

if nargin < 4
    seed = 1;
end

oldRng = rng();
rng(seed);

dat = mvnrnd(mu, sig, howmanysamps);
dat = dat';

rng(oldRng);
