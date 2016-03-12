function [ mm2 ] = mmd_rf( X, Y, fm )
%MMD_RF MMD^2 estimator using random Fourier features. Complexity = O(ND) 
%where N = sample size, D = #random features

% MMD_LIN Linear unbiased MMD^2 estimator. Complexity = O(max(nx, ny))
%   - X : d x nx matrix of X samples 
%   - Y : d x ny matrix of Y samples
%   - fm : an object of type FeatureMap
%
% @date 1 Oct 2015
% @author Wittawat Jitkrittum
%

assert(isa(fm, 'FeatureMap'));

% D x nx. D = #random features.
Zx = fm.genFeatures(X); 
Zy = fm.genFeatures(Y); 
mx = mean(Zx, 2);
my = mean(Zy, 2);
mm2 = sum( (mx-my).^2 );

end

