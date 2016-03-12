function [ funcs ] = funcs_distreg_zoltan( )
%FUNCS_DISTREG_ZOLTAN A collection of functions related to distreg_zoltan
%
    funcs = struct();
    funcs.meddistance_bags = @meddistance_bags;
    funcs.regress_weights_func = @regress_weights_func;

end

function med = meddistance_bags(bags, subbags )
    % compute the pairwise median heuristic on the samples in all bags. 
    % subsamples bags to (subbags) to reduce computation.
    assert(iscell(bags));
    b = length(bags);
    I = randperm(b, min(subbags, b));
    sub = bags(I);
    flat = [sub{:}];
    n = size(flat, 2);
    Is = randperm(n, min(6000, n));
    med = meddistance(flat(:, Is));
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

