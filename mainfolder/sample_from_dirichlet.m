function samps = sample_from_dirichlet(alpha, n)

% mijung wrote on jan 23, 2015

% currently, it only supports for alpha=1
% update it later in case we want something different than alpha=1

% input
% (1) alpha : concentration param
% (2) n: number of samps to draw

dim_samp = length(alpha);
samps_before_normalisation = rand(dim_samp, n);
samps = bsxfun(@times, samps_before_normalisation, 1./sum(samps_before_normalisation));