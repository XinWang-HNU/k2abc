function [ mm ] = mmd_lin( X, Y, ker )
% MMD_LIN Linear unbiased MMD estimator. Complexity = O(max(nx, ny))
%   - X : d x nx matrix of X samples 
%   - Y : d x ny matrix of Y samples
%
%   This linear MMD estimator is a little different from the one proposed in 
%   "Kernel Two-sample test" of Arthur et. al. JMLR. There, deriving the distribution 
%   of the MMD based on the central limit theorem (CLT) is important. So, terms in 
%   the U-statistic form are intentionally made independent to use CLT. We do
%   not need CLT here as we do not care about the distribution. We will use
%   a slight variant of that involving more terms (dependent), but is still linear.
%
% @date 26 Sep 2015
% @author Wittawat Jitkrittum
%

assert(isa(ker, 'Kernel'));
assert(isnumeric(X));
assert(isnumeric(Y));

nx = size(X, 2);
Kxvec = ker.pairEval( X(:, 1:(nx-1)), X(:, 2:nx) );
xx = mean(Kxvec);

ny = size(Y, 2);
Kyvec = ker.pairEval( Y(:, 1:(ny-1)), Y(:, 2:ny) );
yy = mean(Kyvec);

% cross term.
% First make sure that X, Y have the same length. If not, cycle the shorter 
% one (S) to match the length of the longer one (L).
if nx <= ny 
    S = X;
    L = Y;
else
    S = Y;
    L = X;
end
s = size(S, 2);
l = size(L, 2);
assert(s <= l);
M = repmat( (1:s)', 1, ceil(l/s) );
S = S(:, M(1:l));
% S will now have the same length as L with repeated observations.
assert(size(S, 2) == size(L, 2));
Ksl = ker.pairEval(S, L);
xy = mean(Ksl);

% unbiased mmd.
mm2 = xx - 2*xy + yy;
if mm2 < 0
   mm = 0;
   %display(sprintf('mmd_lin. negative mm2: %.3f', mm2));
else
   mm = sqrt(mm2);
end
%mm = abs(sqrt(mm2));

end

