% generate data from a multivariate normal distribution 

function dat = gen_mvn(mu, sig, howmanysamps)

% inputs
%         mu: mean
%         sig: covariance
%         howmanysamps: how many samples to generate
%
% outputs
%        dat: data

dat = mvnrnd(mu, sig, howmanysamps);
dat = dat';