function [mm] = mmd(X, Y, ker)
% MMD compute maximum mean discrepancy
%   - X : d x nx matrix of X samples 
%   - Y : d x ny matrix of Y samples
%
% ** TODO: This version of MMD does not yet support large nx, ny. 
% Not so difficult to fix. Just compute by chunks. Do it later when needed. **
% TODO: May want to consider unbiased mmd.
%
% @author Wittawat
%
assert(isa(ker, 'Kernel'));
assert(isnumeric(X));
assert(isnumeric(Y));

Kx = ker.eval(X, X);
xx = mean(Kx(:));
clear Kx

Ky = ker.eval(Y, Y);
yy = mean(Ky(:));
clear Ky

% nx x ny
Kxy = ker.eval(X, Y);
xy = mean(Kxy(:));
clear Kxy
mm = sqrt(xx - 2*xy + yy);

end
