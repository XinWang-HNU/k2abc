%% test abc

% startup

clear all;
clc;

%% (1) generate data

d = 1; % dimension of theta (we only consider mean first)
theta_mean = 0;
theta_var = 1;
prior_var = 10;
howmanysamps = 100;

% size(yobs) = (dim of y) by (# samples)
yobs = gen_mvn(theta_mean, theta_var, howmanysamps); % observed data

% sample theta M times
M = 100;
theta_samps = zeros(M, 1);

% these will vary later
kernelparams = meddistance(yobs)^2;
howmanyepsilon = 6;
epsilon = logspace(-4, 8, howmanyepsilon);
% epsilon = 1000; 

muhat = zeros(howmanyepsilon,1);

for count = 1:howmanyepsilon
    
    % we sample y L times, where each y consists of Ns samples
    L = 50;
    Ns = 200;
    k = zeros(M, L);
    
    for j=1:M
        
        %% (2) draw parameters from the prior (theta_j)
        % e.g., fix sigma, and draw mean from a Gaussian
        
        theta_samps(j) = mvnrnd(zeros(1, d)+2, prior_var);
        
        %% (3) sample y from the parameters (y_i^j)
        
        parfor l = 1:L
            
            [count j l]
            
            y = gen_mvn(theta_samps(j), theta_var, Ns);
            
            %% (4) compute MMD for each y_i^j and y*_i^j
            ker = KGaussian(kernelparams);
            k(j, l) = exp(-mmd(y, yobs, ker)^2/epsilon(count));
            
        end
        
    end
    
    %% (5) compute w_j which gives us posterior mean of theta
    
    wj_numerator = sum(k, 2)/L;
    wj_denominator = sum(sum(k))/L;
    
    muhat(count) = sum(wj_numerator.*theta_samps)/wj_denominator;
    
    [theta_mean muhat(count)]
    
end

%% (6) compute f(sigma, epsilon) = squared distance between theta_mean and theta_true

mse = @(a) (a-theta_mean).^2;
subplot(211); semilogx(epsilon, muhat); ylabel('muhat'); title('fixed length scale');
subplot(212); loglog(epsilon, mse(muhat)); xlabel('epsilon'); ylabel('mse'); 

% as epsilon gets larger, our estimate gets closer to prior mean.
% as epsilon gets smaller, our estimate gets closer to observations.




