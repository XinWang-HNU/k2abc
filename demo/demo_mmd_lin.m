function demo_mmd_lin()
% This script demonstrates the behaviour of linear MMD estimator (mmd_lin).
% @author Wittawat
%

%
seed = 3;
oldRng = rng();
rng(seed);

demo_increasing_mean();
demo_increasing_sd();

rng(oldRng);
end

function demo_increasing_mean()
% toy data. X, Y follow a zero mean Gaussian. X has a variance of 1. 
% Y has a increasing mean.
d = 2;
nx = 180;
ny = 400;
X = randn(d, nx);
med = meddistance(X);

% mean of Y
means = 0:0.2:4;
MMDL = zeros(1, length(means));
MMDF = zeros(1, length(means));
for i=1:length(means)
    Y = randn(d, ny) + means(i);
    %med = meddistance([X, Y]);
    ker = KGaussian(med^2);

    MMDL(i) = mmd_lin(X, Y, ker);
    MMDF(i) = mmd(X, Y, ker);
end

% plot 
figure 
hold on
plot(means, MMDL, 'o-b', 'linewidth', 2);
plot(means, MMDF, 'x-r', 'linewidth', 2);
set(gca, 'fontsize', 20);
title('Linear MMD^2 vs full MMD^2. X \sim N(0, 1), Y \sim N(mean, 1).');
xlabel('mean of Y');
ylabel('MMD(X, Y)^2');
legend('Linear MMD', 'Full MMD');
grid on
hold off

end

function demo_increasing_sd()
% toy data. X, Y follow a zero mean Gaussian. X has a variance of 1. 
% Y has a increasing variance.
d = 2;
nx = 180;
ny = 400;
X = randn(d, nx);
med = meddistance(X);

% standard deviations of Y
stds = 1:20;
MMDL = zeros(1, length(stds));
MMDF = zeros(1, length(stds));
for i=1:length(stds)
    Y = randn(d, ny)*stds(i);
    %med = meddistance([X, Y]);
    ker = KGaussian(med^2);

    MMDL(i) = mmd_lin(X, Y, ker);
    MMDF(i) = mmd(X, Y, ker);
end

% plot 
figure 
hold on
plot(stds, MMDL, 'o-b', 'linewidth', 2);
plot(stds, MMDF, 'x-r', 'linewidth', 2);
set(gca, 'fontsize', 20);
title('Linear MMD^2 vs full MMD^2. X \sim N(0, 1), Y \sim N(0, sd^2).');
xlabel('sd of Y');
ylabel('MMD(X, Y)^2');
legend('Linear MMD', 'Full MMD');
grid on
hold off

end
