function [ results, op ] = k2abc_lin( Obs, op )
%K2ABC_LIN Perform inference with K2ABC using linear MMD. 
%   - This function is a wrapper of ssf_abc to specialize to k2abc with linear MMD.
%   The usage is the same as ssf_kernel_abc which is K2ABC with full MMD.
%   - Will try many epsilon's (soft rejection threshold) in the candidate list.
%
% Input:
%   - Obs : D x M where D is dimension and M is the number of observations
%   - op : a struct specifying options. See the code for possible options.
%
% Out:
%   - results : a struct containing latent samples and their weights for each 
%   epsilon
%   - op : option struct.
%

assert(isnumeric(Obs));

% kernel function for computing MMD between Obs and the drawn pseudo data.
% Default to a Gaussian kernel with median heuristic.
op.mmd_kernel = myProcessOptions(op, 'mmd_kernel', KGaussian(meddistance(Obs)^2)); 
%mmd_kernel = op.mmd_kernel;


op.pseudo_data_measure = @(data1, data2)mmd_lin(data1, data2, op.mmd_kernel);
op.threshold_func = @(dists, epsilons) exp(-dists(:)*(1./epsilons(:)'));
[results, op] = ssf_abc(Obs, op);

% num_latent_draws x length(epsilon_list)
unnorm_weights = results.unnorm_weights;
% normalized weights
norm_weights = bsxfun(@rdivide, unnorm_weights, sum(unnorm_weights, 1) );

% a struct results containing all results
results.norm_weights = norm_weights;


end

