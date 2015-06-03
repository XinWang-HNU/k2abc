function [theta theta_d_samp abc_discs] = bayes_ind_inf_ip_gandk(y,M,n,cov_rw,abc_tol,numComp,regularize)

theta = zeros(M,4);

% MH - IL
theta_curr = [3 1 2 0.5];

abc_discs = zeros(M,1);
abc_disc = abc_tol;

theta_d_samp = zeros(M,3*numComp-1);

% fit auxiliary model
obj = gmdistribution.fit(y,numComp,'Regularize',regularize,'Options',statset('MaxIter',10000,'TolFun',1e-6));
 [mu ix] = sort(obj.mu);
 p =  obj.PComponents;
 p = p(ix);
 vars = reshape(obj.Sigma,numComp,1);
 vars = vars(ix);
 
 theta_d = [p(1:(numComp-1)) mu' vars'];
 summary_stats = theta_d;

 weight_matrix = compute_obs_inf(theta_d,y,obj,numComp);

for i = 1:M
    i
    theta_prop = mvnrnd(theta_curr,cov_rw);
    if (any(theta_prop<0) || any(theta_prop>10))
        theta(i,:) = theta_curr;
        abc_discs(i) = abc_disc;
        theta_d_samp(i,:) = summary_stats;
        continue;
    end
    y_s = simulate_gk(n,[theta_prop(1:2) 0.8 theta_prop(3:4)]);
    obj = gmdistribution.fit(y_s,numComp,'Regularize',regularize,'Options',statset('MaxIter',10000,'TolFun',1e-6));
    % fit auxiliary model
    [mu ix] = sort(obj.mu);
    p =  obj.PComponents;
    p = p(ix);
    vars = reshape(obj.Sigma,numComp,1);
    vars = vars(ix);
    
    theta_d_prop = [p(1:(numComp-1)) mu' vars'];
    diff_summ = theta_d_prop-theta_d;
    
    dist_prop = diff_summ*weight_matrix*diff_summ';
    
    if (dist_prop <= abc_tol)
        theta_curr = theta_prop;
        abc_disc = dist_prop;
        summary_stats = theta_d_prop;
    end
    theta(i,:) = theta_curr;
    theta_d_samp(i,:) = summary_stats;
    abc_discs(i) = abc_disc;
    
end

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
