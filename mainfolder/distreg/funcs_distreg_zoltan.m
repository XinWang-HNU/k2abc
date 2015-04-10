function [ funcs ] = funcs_distreg_zoltan( )
%FUNCS_DISTREG_ZOLTAN A collection of functions related to distreg_zoltan
%
    funcs = struct();
    funcs.meddistance_bags = @meddistance_bags;

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




