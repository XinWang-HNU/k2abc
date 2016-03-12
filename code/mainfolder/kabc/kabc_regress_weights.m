function [ W ] = kabc_regress_weights(train_stats, gwidth2, reg, test_stats)
%KABC_REGRESS_WEIGHTS Predict the posterior weights to be used on the training statistics
%in kernel ABC.
%   - This function plays the same role as regress_weights_func() in kabc_cond_embed.
%   - Assume Gaussian kernel.
%  Input 
%   - train_stats: dxn training summary statistics 
%   - gwidth2: Gaussian width squared
%   - reg: regularization parameter (scalar)
%   - test_stats: dxn' test statistics to predict the weights.
%  Output 
%   - W: nxn' of weights.
%
%@author Wittawat
%

    ker = KGaussian(gwidth2);
    Krs = ker.eval(train_stats, test_stats);
    % This may blow up the memory.
    K = ker.eval(train_stats, train_stats);
    ntr = size(K, 1);
    % ntr x nte
    W = linsolve(K + reg*eye(ntr), Krs);
end

