function [mm2] = mmd(X, Y, ker)
% MMD compute unbiased estimator of MMD^2 (maximum mean discrepancy).
%   - X : d x nx matrix of X samples 
%   - Y : d x ny matrix of Y samples
%
% ** TODO: This version of MMD does not yet support large nx, ny. 
% Not so difficult to fix. Just compute by chunks. Do it later when needed. **
%
% @author Wittawat
%
assert(isa(ker, 'Kernel'));
assert(isnumeric(X));
assert(isnumeric(Y));

nx = size(X, 2);
Kx = ker.selfEval(X);
diagIx = 1:(nx+1):numel(Kx);
xx = (sum(Kx(:)) - sum(Kx(diagIx)) )/(nx*(nx-1));
%xx = mean(Kx(:));
clear Kx

ny = size(Y, 2);
Ky = ker.selfEval(Y);
diagIy = 1:(ny+1):numel(Ky);
yy = (sum(Ky(:)) - sum(Ky(diagIy)) )/(ny*(ny-1));
%yy = mean(Ky(:));
clear Ky

% nx x ny
Kxy = ker.eval(X, Y);
xy = mean(Kxy(:));
clear Kxy

% unbiased mmd can be negative.  
%
mm2 = xx - 2*xy + yy;
%if mm2 < 0
%    mm = 0;
%    %display(sprintf('mmd. negative mm2: %.3f', mm2));
%else
%    mm = sqrt(mm2);
%end


end
