function [ mm2 ] = mmd_lin( X, Y, ker )
% MMD_LIN Linear unbiased MMD^2 estimator. Complexity = O(max(nx, ny))
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

% deterministic shuffle of Obs. This will break dependency when we apply this method to 
% non-i.i.d. data. This shuffling should not have a negative effect for i.i.d.
% case.
oldRng = rng();
rng(19280);
X = X(:, randperm(size(X, 2)));
Y = Y(:, randperm(size(Y, 2)));
rng(oldRng);
%
nx = size(X, 2);
Kxvec = ker.pairEval( X(:, 1:(nx-1)), X(:, 2:nx) );
xx = mean(Kxvec);
%xx = mean([Kxvec, ker.pairEval(X, X)]);

ny = size(Y, 2);
Kyvec = ker.pairEval( Y(:, 1:(ny-1)), Y(:, 2:ny) );
yy = mean(Kyvec);
%yy = mean([Kyvec, ker.pairEval(Y, Y)]);

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

%if mm2 < 0
%mm2 = 0;
%%display(sprintf('mmd_lin. negative mm2: %.3f', mm2));
%end

end

