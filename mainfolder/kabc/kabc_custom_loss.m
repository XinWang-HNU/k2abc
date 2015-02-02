function [results, op ] = kabc_custom_loss(stats, params, op )
%KABC_CUSTOM_LOSS Implementation of kernel ABC of Nakagome et. al.  Regress 
%from summary statistics (stats) to parameters (params). Use a custom loss 
%function for selecting parameters. Whether to do CV depends on the specified 
%loss function.
%
% Details: 
%   - We use conditional mean embedding instead of kernel Bayes's rule as 
%   stated in the paper. Specifically, we regress from the input summary 
%   statistic to the parameter of interest. 
%
% Assumptions:
%   - stats can be averaged. This is needed in the objective of cross
%   validation.
%
% Input: 
%   - stats : D x n where D is dimension and n is the number of simulations.
%   - op : option struct.
%
% Out:
%   - results : a struct containing latent samples and their weights 
%   - op : option struct used 
%
% TODO: 
%   - Current implementation explicitly forms a full gram matrix. This is not 
%   scalable if n in stats is large. 
%   Improve it later if needed. 
%

% random seed. This will affect the partitioning of the data in cross
% validation.
op.seed = myProcessOptions(op, 'seed', 1);
oldRng = rng();
rng(op.seed);

% a list of regularization parameter candidates. Chosen by cross validation.
op.kabc_reg_list = myProcessOptions(op, 'kabc_reg_list', 10.^(-4:2:3));
reg_list = op.kabc_reg_list;

% a list of Gaussian widths squared to be used as candidates for Gaussian kernel.
default_gwidth2s = [1/2, 1, 2].* (meddistance(stats).^2);
op.kabc_gwidth2_list = myProcessOptions(op, 'kabc_gwidth2_list', default_gwidth2s);
gwidth2_list = op.kabc_gwidth2_list;

% Loss function to be used to measure the goodness of parameters (Gaussian width ,
% regularization parameter). A loss function takes the form 
% f: (weights_func, train_stats ) -> real number.  
% Lower is better.
%
if isOptionEmpty(op, 'kabc_loss_func')
    error('kabc_loss_func cannot be empty.');
end
loss_func = op.kabc_loss_func;
assert(isa(loss_func, 'function_handle'), 'kabc_loss_func must be a function handle.');


ParamErr = zeros(length(gwidth2_list), length(reg_list));

% linsolve option
linsolve_opts = struct();
%linsolve_opts.POSDEF = true;
%linsolve_opts.SYM = true;

% distance matrix only once.
sumStat2 = sum(stats.^2, 1);
Dist2 = bsxfun(@plus, sumStat2, sumStat2') - 2*stats'*stats;
for gi = 1:length(gwidth2_list)
 
    gwidth2 = gwidth2_list(gi);
    K = exp(-Dist2./(2*gwidth2));

    for ri=1:length(reg_list)
        reg = reg_list(ri);
        % A function taking a stat and returning a 1xsize(params, 2) weight vector
        % to be used on parameters in (params).
        weights_func = @(test_stats)regress_weights_func(K, stats, gwidth2,...
            reg, linsolve_opts, test_stats);
        err = loss_func(weights_func, stats);
        ParamErr(gi, ri) = err;

        fprintf('gw2: %.2g, reg: %.2g => err: %.3g\n', ...
            gwidth2, reg, err);

    end
end

% best param combination
[minerr, ind] = min(ParamErr(:));
[bgi, bri] = ind2sub(size(ParamErr), ind);
best_gwidth2 = gwidth2_list(bgi);
best_reg = reg_list(bri);

% a struct results containing all results
results = struct();
% 1 x n. Regression weights corresponding to the best parameter combinations.
% Can be negative.
% Return a function which takes in stats (d x 1) and output weights (n x 1).
% Can also take in (d x n') to produce (n x n').
ker = KGaussian(best_gwidth2);
K = ker.eval(stats, stats);
results.regress_weights_func = @(test_stats)regress_weights_func(K, stats, ...
    best_gwidth2, best_reg, linsolve_opts, test_stats);
results.best_gwidth2 = best_gwidth2;
results.best_reg = best_reg;
results.min_err = minerr;
results.param_err = ParamErr;

rng(oldRng);
end

function W = regress_weights_func(K, train_stats, gwidth2, reg, ...
        linsolve_opts, test_stats)
    % final regression function giving the weights (can be negative) 
    % on each param
    
    ker = KGaussian(gwidth2);
    Krs = ker.eval(train_stats, test_stats);
    ntr = size(K, 1);
    W = linsolve(K + reg*eye(ntr), Krs, linsolve_opts);
end




