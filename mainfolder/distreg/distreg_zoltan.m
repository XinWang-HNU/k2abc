function [ results, op ] = distreg_zoltan( bags, target, op )
%DISTREG_ZOLTAN 2-staged distribution regression of Zoltan et. al.
% Input:
%   - bags: a 1xn cell array of bags of samples. Each element in the cell is 
%   a dxn_i matrix. This forms the input to the regression function.
%   - target: a dout x n matrix of regression targets.
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
% @author Wittawat Jitkrittum
% @date: 2014/04/09
%

assert(iscell(bags), 'bags must be a cell array');
assert(isnumeric(target), 'regression target must be a vector/matrix');

n = length(bags);
assert(n==size(target, 2), '#bags must be = #columns of target');

% random seed. This will affect the partitioning of the data in cross
% validation.
op.seed = myProcessOptions(op, 'seed', 1);
seed = op.seed;
oldRng = rng();
rng(op.seed);

% a list of regularization parameter candidates. Chosen by cross validation.
op.drz_reg_list = myProcessOptions(op, 'drz_reg_list', 10.^(-4:2:3));
reg_list = op.drz_reg_list;

% number of folds to perform in cross validation
op.drz_cv_fold = myProcessOptions(op, 'drz_cv_fold', 5);
cv_fold = op.drz_cv_fold;

% a list of Gaussian widths squared to be used as candidates for (embedding)
% Gaussian kernel.
funcs = funcs_distreg_zoltan();
med = funcs.meddistance_bags(bags, 200);
default_gwidth2s = [1/2, 1, 2].* (med.^2);
op.drz_gwidth2_list = myProcessOptions(op, 'drz_gwidth2_list', default_gwidth2s);
gwidth2_list = op.drz_gwidth2_list;

% The width^2 for the outer Gaussian. In ABC application, these are the epsilon's.
op.drz_outwidth2_list = myProcessOptions(op, 'drz_outwidth2_list', 10.^(-3:0) );
outwidth2_list = op.drz_outwidth2_list;

% number of subsamples to use for cross validation. This is for speeding up 
% the computation in cross validation. After the best parameter combination 
% is chosen, full dataset will be used to train.
op.drz_cv_num_subsamples = myProcessOptions(op, 'drz_cv_num_subsamples', ...
    min(n, 1e4) );
cv_subsamples = op.drz_cv_num_subsamples;

% partitioning the data for CV
% 0-1 fold x n matrix
Icv = strafolds(cv_subsamples, cv_fold, seed );
Isub = randperm(n, cv_subsamples);
% subsamples for speeding up cross validation
sub_bags = bags(:, Isub);
sub_target = target(:, Isub);

CVErr = zeros(cv_fold, length(gwidth2_list), length(outwidth2_list),...
    length(reg_list));

% linsolve option
linsolve_opts = struct();
%linsolve_opts.POSDEF = true;
linsolve_opts.SYM = true;
for fi=1:cv_fold
    % test indices 
    teI = Icv(fi, :);
    % training indices
    trI = ~teI;
    ntr = sum(trI);
    nte = sum(teI);
    Ytr = sub_target(:, trI);
    Yte = sub_target(:, teI);
    
    for gi = 1:length(gwidth2_list)
        gwidth2 = gwidth2_list(gi);
        ker_gi = KGaussian(gwidth2); 

        mmd2_gi_tr = KGGauss.selfEvalMMD(sub_bags(trI), ker_gi).^2;
        mmd2_gi_rs = KGGauss.evalMMD(sub_bags(trI), sub_bags(teI), ker_gi).^2;

        for oi = 1:length(outwidth2_list)
            outwidth2 = outwidth2_list(oi);
            Ktr = exp(-0.5*mmd2_gi_tr/outwidth2);
            Krs = exp(-0.5*mmd2_gi_rs/outwidth2);

            for ri=1:length(reg_list)
                reg = reg_list(ri);
                % A: dy x ntr
                A = linsolve(Ktr + reg*eye(ntr), Ytr', linsolve_opts)';
                AKrs = A*Krs;

                err = (1.0/nte)*( AKrs(:)'*AKrs(:) - 2*(Yte(:)'*AKrs(:)) +Yte(:)'*Yte(:) );
                CVErr(fi, gi, oi, ri) = err;
                fprintf('fold: %d, gw2: %.2g, outw2: %.2g, reg: %.2g => err: %.3g\n', fi, ...
                    gwidth2, outwidth2, reg, err);

            end
        end
    end
end

% length(gwidth2_list) x length(outwidth2_list) x length(reg_list)
error_grid = shiftdim(mean(CVErr, 1), 1);
assert(all(size(error_grid) == [length(gwidth2_list), length(outwidth2_list), length(reg_list)]));

% best param combination
[minerr, ind] = min(error_grid(:));
[bgi, boi, bri] = ind2sub(size(error_grid), ind);
best_gwidth2 = gwidth2_list(bgi);
best_outwidth2 = outwidth2_list(boi);
best_reg = reg_list(bri);

% a struct results containing all results
results = struct();

% 1 x n. Regression weights corresponding to the best parameter combinations.
% Can be negative.
% Return a function which takes in a bag (a matrix) and output weights (n x 1).
% Can also take in (1 x n' cell array) to produce (n x n').
ker = KGGauss(best_gwidth2, best_outwidth2);
K = ker.selfEval(bags);

results.regress_weights_func = @(test_bags)regress_weights_func(K, bags, ...
    best_gwidth2, best_outwidth2, best_reg, linsolve_opts, test_bags);
results.best_gwidth2 = best_gwidth2;
results.best_outwidth2 = best_outwidth2;
results.best_reg = best_reg;
results.min_cv_err = minerr;
results.cverr = CVErr;
results.cverr_foldmean = error_grid;

rng(oldRng);
end

function W = regress_weights_func(K, train_bags, gwidth2, outwidth2, reg, ...
        linsolve_opts, test_bags)
    % final regression function giving the weights (can be negative) 
    % on each param (regression target)
    
    ker = KGGauss(gwidth2, outwidth2);
    Krs = ker.eval(train_bags, test_bags);
    ntr = size(K, 1);
    W = linsolve(K + reg*eye(ntr), Krs, linsolve_opts);
end

