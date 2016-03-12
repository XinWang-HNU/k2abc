function [ results, op ] = distreg_zoltan_custom_loss( bags, target, op )
%DISTREG_ZOLTAN_CUSTOM_LOSS 2-staged distribution regression of Zoltan et. al.
%   Use a specified custom loss function instead of squared loss for parameter
%   selection.
%
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
% @date: 2014/04/11
%

assert(iscell(bags), 'bags must be a cell array');
assert(isnumeric(target), 'regression target must be a vector/matrix');

n = length(bags);
assert(n==size(target, 2), '#bags must be = #columns of target');

% random seed. This will affect the partitioning of the data in cross
% validation.
op.seed = myProcessOptions(op, 'seed', 1);
oldRng = rng();
rng(op.seed);

% a list of regularization parameter candidates. Chosen by cross validation.
op.drz_reg_list = myProcessOptions(op, 'drz_reg_list', 10.^(-4:2:3));
reg_list = op.drz_reg_list;


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

% Loss function to be used to measure the goodness of parameters (Gaussian width, 
% outer width (epsilon),  regularization parameter). A loss function takes the form 
% f: (weights_func, train_bags, train_target ) -> real number.  
% Lower is better.
%
if isOptionEmpty(op, 'drz_custom_loss_func')
    error('kabc_loss_func cannot be empty.');
end
loss_func = op.drz_custom_loss_func;
assert(isa(loss_func, 'function_handle'), 'drz_custom_loss_func must be a function handle.');


ParamErr = zeros(length(gwidth2_list), length(outwidth2_list), length(reg_list));
% linsolve option
linsolve_opts = struct();
%linsolve_opts.POSDEF = true;
linsolve_opts.SYM = true;

funcs = funcs_distreg_zoltan();
%Ytr = target;
%ntr = size(target, 2);
for gi = 1:length(gwidth2_list)
    gwidth2 = gwidth2_list(gi);
    ker_gi = KGaussian(gwidth2); 

    mmd2_gi_tr = KGGauss.selfEvalMMD(bags, ker_gi).^2;

    for oi = 1:length(outwidth2_list)
        outwidth2 = outwidth2_list(oi);
        Ktr = exp(-0.5*mmd2_gi_tr/outwidth2);

        for ri=1:length(reg_list)
            reg = reg_list(ri);

            % A function taking a stat and returning a 1xsize(params, 2) weight vector
            % to be used on parameters in (params).
            weights_func = @(test_bags)funcs.regress_weights_func(Ktr, bags, gwidth2,...
                outwidth2, reg, linsolve_opts, test_bags);
            err = loss_func(weights_func, bags, target);

            ParamErr(gi, oi, ri) = err;
            fprintf('gw2: %.2g, outw2: %.2g, reg: %.2g => err: %.3g\n', ...
                gwidth2, outwidth2, reg, err);
        end
    end
end

% best param combination
[minerr, ind] = min(ParamErr(:));
[bgi, boi, bri] = ind2sub(size(ParamErr), ind);
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

results.regress_weights_func = @(test_bags)funcs.regress_weights_func(K, bags, ...
    best_gwidth2, best_outwidth2, best_reg, linsolve_opts, test_bags);
results.best_gwidth2 = best_gwidth2;
results.best_outwidth2 = best_outwidth2;
results.best_reg = best_reg;
results.min_err = minerr;
results.param_err = ParamErr;

rng(oldRng);
end

