function [theta, summary_stats_samp, abc_discs] = bayes_ind_inf_is_gandk(y,M,n,cov_rw,abc_tol,numComp)

%% Christopher Drovandi's code 

%% mijung comments : input arguments seem like
% (1) y: true observation, size(y) = n by 1
% (2) M : number of posterior samples of parameters (theta)
% (3) n:  number of true observations
% (4) cov_rw: prior covariance, size(cov_rw) = number params by num params
% (5) abc_tol: tolerence for discrepancy between log datalikelihood given auxiliary model
% (6) numComp: number of components for mixture model for auxiliary model

%% mijung comments : size(theta) is how many samples by number of parameters
theta = zeros(M,4);

%% mijung comments : theta_curr is prior mean and cov_rw is prior cov, and numComp: how many mixture models to use

% MH - IL
theta_curr = [3,1.02,2.09,0.48];

%%
abc_discs = zeros(M,1);
abc_disc = abc_tol;
summary_stats_samp = zeros(M,3*numComp-1);
summary_stats = zeros(1,3*numComp-1);

% mijung comments: fitting data to gaussian mixture
obj = gmdistribution.fit(y,numComp,'Options',statset('MaxIter',100000,'TolFun',1e-10));

theta_d = [obj.PComponents(1:(numComp-1)) obj.mu' reshape(obj.Sigma,numComp,1)'];

%if (numComp == 4)
    %weight_matrix = inv(-1*hessian_mixture(theta_d,y));
    %weight_matrix = eye(11);
    weight_matrix = compute_obs_inf(theta_d,y,obj,numComp);
    weight_matrix = inv(weight_matrix);
%else
%    weight_matrix=inv(-hessian(@(x)log_like_mixture(x,y,numComp),theta_d));
%end

for i = 1:M
    i
    theta_prop = mvnrnd(theta_curr,cov_rw);
    if (any(theta_prop<0) || any(theta_prop>10))
        theta(i,:) = theta_curr;
        summary_stats_samp(i,:) = summary_stats;
        abc_discs(i) = abc_disc;
        continue;
    end
    
    % mijung comments : here y_s is simulated data, size(y_s) = n by 1
    y_s = simulate_gk(n,[theta_prop(1:2) 0.8 theta_prop(3:4)]);
    
    %[grad,~,~] = gradest(@(x)log_like_mixture(x,y_s),theta_d);
    grad = compute_grad(theta_d,y_s,obj,numComp);
    dist_prop = grad*weight_matrix*grad';
    
    if (dist_prop <= abc_tol)
        theta_curr = theta_prop;
        abc_disc = dist_prop;
        summary_stats = grad;
    end
    theta(i,:) = theta_curr;
    summary_stats_samp(i,:) = summary_stats;
    abc_discs(i) = abc_disc;
    
end



function log_like = log_like_mixture(theta,y,numComp)
    w = theta(1:(numComp-1));
    w(numComp) = 1-sum(theta(1:(numComp-1)));
    mu = theta(numComp:(2*numComp-1));
    sigma = sqrt(theta((2*numComp):(3*numComp-1)));
    like = zeros(length(y),1);
    for i = 1:numComp
        like = like + w(i)*normpdf(y,mu(i),sigma(i));
    end
    log_like = sum(log(like));

function the_grad = compute_grad(theta,y,obj,numComp)
    w = theta(1:(numComp-1));
    w(numComp) = 1-sum(theta(1:(numComp-1)));
    mu = theta(numComp:(2*numComp-1));
    sigma = sqrt(theta((2*numComp):(3*numComp-1)));
    the_grad = zeros(1,(3*numComp-1));
    for i = 1:(numComp-1)
        the_grad(i) = sum(1./pdf(obj,y).*(normpdf(y,mu(i),sigma(i)) - normpdf(y,mu(numComp),sigma(numComp))));
    end
    for i = 1:numComp
        the_grad(i+numComp-1) = sum(1./pdf(obj,y).*(w(i)*normpdf(y,mu(i),sigma(i))*1/sigma(i)^2.*(y-mu(i))));
    end
    for i = 1:numComp
        the_grad(i+2*numComp-1) = sum(1./pdf(obj,y)*w(i).*normpdf(y,mu(i),sigma(i)).*(-0.5/sigma(i)^2 + 0.5/sigma(i)^4*(y-mu(i)).^2));
    end
    
function W = compute_obs_inf(theta,y,obj,numComp)
% mijung comments: this seems like computing the information matrix
% J(\phi((y))
        
w = theta(1:(numComp-1));
w(numComp) = 1-sum(theta(1:(numComp-1)));
mu = theta(numComp:(2*numComp-1));
sigma = sqrt(theta((2*numComp):(3*numComp-1)));
the_grad = zeros(1,(3*numComp-1));
W = zeros((3*numComp-1));

for t = 1:length(y)
    
    for i = 1:(numComp-1)
        the_grad(i) = sum(1./pdf(obj,y(t)).*(normpdf(y(t),mu(i),sigma(i)) - normpdf(y(t),mu(numComp),sigma(numComp))));
    end
    for i = 1:numComp
        the_grad(i+numComp-1) = sum(1./pdf(obj,y(t)).*(w(i)*normpdf(y(t),mu(i),sigma(i))*1/sigma(i)^2.*(y(t)-mu(i))));
    end
    for i = 1:numComp
        the_grad(i+2*numComp-1) = sum(1./pdf(obj,y(t))*w(i).*normpdf(y(t),mu(i),sigma(i)).*(-0.5/sigma(i)^2 + 0.5/sigma(i)^4*(y(t)-mu(i)).^2));
    end
    
    W = W + the_grad'*the_grad;
    
end


   

        
        


