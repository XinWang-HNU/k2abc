%% test abc

%% (1) generate data

d = 1; % dimension of theta (we only consider mean first)
theta_mean = 0;
theta_var = 1;
howmanysamps = 100;

% size(yobs) = (dim of y) by (# samples)
yobs = gen_mvn(theta_mean, theta_var, howmanysamps); % observed data

% sample theta M times
M = 100;
theta_samps = zeros(M, 1);

% these will vary later
kernelparams = 1;
epsilon = 1; 

% we sample y L times, where each y consists of Ns samples
L = 50;
Ns = 200;
k = zeros(M, L);

for j=1:M
    
    %% (2) draw parameters from the prior (theta_j)
    % e.g., fix sigma, and draw mean from a Gaussian
    
    theta_samps(j) = mvnrnd(zeros(1, d), theta_var);
    
    %% (3) sample y from the parameters (y_i^j)
    
    for l = 1:L
        
        y = gen_mvn(theta_samps(j), theta_var, Ns);
        
        %% (4) compute MMD for each y_i^j and y*_i^j
        k(j, l) = exp(-mmd(y, yobs, kernelparams)/epsilon);
        
    end
    
end

%% (5) compute w_j which gives us posterior mean of theta

wj_numerator = sum(k, 2)/L;
wj_denominator = sum(sum(k));

muhat = sum((wj_numerator.*theta_samps)/wj_denominator);

%% (6) compute f(sigma, epsilon) = squared distance between theta_mean and theta_true



%% to do:



