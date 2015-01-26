function [results, op ] = kabc_cond_embed(stats, params, op )
%KABC_COND_EMBED Implementation of kernel ABC of Nakagome et. al.  Regress 
%from summary statistics (stats) to parameters (params).
%
% Paper: Nakagome et. al. "Kernel approximate Bayesian computation in
% population genetic inferences". 2013.
%
% Details: 
%   - We use conditional mean embedding instead of kernel Bayes's rule as 
%   stated in the paper. Specifically, we regress from the input summary 
%   statistic to the parameter of interest. 
%   - Cross validation is done with objective of kernel Bayes's rule. 
%   That is to minimize the discrepancy between the prior and marginalized 
%   joint. The objective is described at the end of section 2 of the paper.
%   - For efficiency in cross validation, we consider only a Gaussian kernel 
%   here (as used in the original paper).
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

[d, n] = size(stats);
% random seed. This will affect the partitioning of the data in cross
% validation.
op.seed = myProcessOptions(op, 'seed', 1);
seed = op.seed;
oldRng = rng();
rng(op.seed);

% a list of regularization parameter candidates. Chosen by cross validation.
op.kabc_reg_list = myProcessOptions(op, 'kabc_reg_list', 10.^(-4:2:3));
reg_list = op.kabc_reg_list;

% number of folds to perform in cross validation
op.kabc_cv_fold = myProcessOptions(op, 'kabc_cv_fold', 10);
cv_fold = op.kabc_cv_fold;

% a list of Gaussian widths squared to be used as candidates for Gaussian kernel.
default_gwidth2s = [1/2, 1, 2].* (meddistance(stats).^2);
op.kabc_gwidth2_list = myProcessOptions(op, 'kabc_gwidth2_list', default_gwidth2s);
gwidth2_list = op.kabc_gwidth2_list;

% number of subsamples to use for cross validation. This is for speeding up 
% the computation in cross validation. After the best parameter combination 
% is chosen, full dataset will be used to train.
op.kabc_cv_num_subsamples = myProcessOptions(op, 'kabc_cv_num_subsamples', ...
    min(n, 5000) );
cv_subsamples = op.kabc_cv_num_subsamples;

% partitioning the data for CV
% 0-1 fold x n matrix
Icv = strafolds(cv_subsamples, cv_fold, seed );
Isub = randperm(n, cv_subsamples);
% subsamples for speeding up cross validation
sub_stats = stats(:, Isub);
sub_params = params(:, Isub);
% distance matrix only once.
sumStat2 = sum(sub_stats.^2, 1);
Dist2 = bsxfun(@plus, sumStat2, sumStat2') - 2*sub_stats'*sub_stats;

CVErr = zeros(cv_fold, length(gwidth2_list), length(reg_list));
% linsolve option
linsolve_opts = struct();
%linsolve_opts.POSDEF = true;
%linsolve_opts.SYM = true;
for fi=1:cv_fold
    % test indices 
    teI = Icv(fi, :);
    % training indices
    trI = ~teI;
    ntr = sum(trI);
    
    %prior mean in the test set 
    priorMeanParam = mean(sub_params(:, teI), 2);
    for gi = 1:length(gwidth2_list)
        gwidth2 = gwidth2_list(gi);
        Ktr = exp(-Dist2(trI, trI)./(2*gwidth2));
        Krs = exp(-Dist2(trI, teI)./(2*gwidth2));

        for ri=1:length(reg_list)
            reg = reg_list(ri);
            % ntr x nte
            Wtr = linsolve(Ktr + reg*eye(ntr), Krs, linsolve_opts);
            PostMeans = sub_params(:, trI)*Wtr;
            % marginalize to get an estimate prior mean of params
            estPriorMeanParam = mean(PostMeans, 2);

            % compare to prior mean in the test set 
            err = sum( (estPriorMeanParam - priorMeanParam).^2);
            CVErr(fi, gi, ri) = err;
            fprintf('fold: %d, gw2: %.2g, reg: %.2g => err: %.3g\n', fi, ...
                gwidth2, reg, err);

        end
    end
end

% length(gwidth2_list) x length(reg_list);
error_grid = shiftdim(mean(CVErr, 1), 1);
assert(all(size(error_grid) == [length(gwidth2_list), length(reg_list)]));

% best param combination
[minerr, ind] = min(error_grid(:));
[bgi, bri] = ind2sub(size(error_grid), ind);
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
%K = exp(-Dist2./(2*best_gwidth2));
%OutKinv = linsolve(K + best_reg*eye(n), params', linsolve_opts)';
%assert(size(OutKinv, 2)==n);
results.regress_weights_func = @(test_stats)regress_weights_func(K, stats, ...
    best_gwidth2, best_reg, linsolve_opts, test_stats);
results.best_gwidth2 = best_gwidth2;
results.best_reg = best_reg;
results.min_cv_err = minerr;

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




