function demo_mmd_rf()
% This script dmonstrates the behaviour of MMD using random features. 
%

seed = 3;
oldRng = rng();
rng(seed);

demo_increasing_rf();

rng(oldRng);

end

function demo_increasing_rf()
% toy data. X, Y follow a zero mean Gaussian. X has a variance of 1. 
% Y has a variance of ...

d = 2;
nx = 500;
ny = 500;
X = randn(d, nx);
sdy = 2;
Y = randn(d, ny)*sdy ;
med = meddistance(X);
ker = KGaussian(med^2);

% number of random features to try
rfs = 100:30:1000;
MMDrf = zeros(1, length(rfs));
for i=1:length(rfs)
    % random feature map
    fm = ker.getRandFeatureMap(rfs(i), d);
    %med = meddistance([X, Y]);

    MMDrf(i) = mmd_rf(X, Y, fm);
end

% plot 
figure 
hold on
plot(rfs, MMDrf, 'o-b', 'linewidth', 2);
plot(rfs, mmd(X, Y, ker)*ones(1, length(rfs)), '-r', 'linewidth', 2 );
plot(rfs, mmd_lin(X, Y, ker)*ones(1, length(rfs)), '-k', 'linewidth', 2 );
set(gca, 'fontsize', 20);
title(sprintf('MMD^2 with Random features. X \\sim N(0, 1), Y \\sim N(0, %.1f).', sdy^2));
xlabel('#random features');
ylabel('mmd\_rf(X, Y)^2');
legend('MMD-RF', 'Full MMD', 'MMD-Lin');
grid on
hold off

end
