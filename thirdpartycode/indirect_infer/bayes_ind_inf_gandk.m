function [theta loglike_ind conv] = bayes_ind_inf_gandk(y,M,n,cov_rw,numComp,K)

theta = zeros(M,4);
loglike_ind = zeros(M,1);
conv = zeros(M,1);

% MH - IL
theta_curr = [3 1 2 0.5];

loglikes = zeros(K,1);

for k = 1:K  
    y_s = simulate_gk(n,[theta_curr(1:2) 0.8 theta_curr(3:4)]);
    % fit auxiliary model
    obj = gmdistribution.fit(y_s,numComp,'Options',statset('MaxIter',10000,'TolFun',1e-6));
    loglikes(k) = sum(log(pdf(obj,y)));
end
loglike_ind_curr = -log(K) + logsumexp(loglikes);

for i = 1:M
    i
    theta_prop = mvnrnd(theta_curr,cov_rw);
    if (any(theta_prop<0) || any(theta_prop>10))
        theta(i,:) = theta_curr;
        loglike_ind(i) = loglike_ind_curr;
        continue;
    end
    for k = 1:K
        y_s = simulate_gk(n,[theta_prop(1:2) 0.8 theta_prop(3:4)]);
        % fit auxiliary model
        obj = gmdistribution.fit(y_s,numComp,'Options',statset('MaxIter',10000,'TolFun',1e-6));
        loglikes(k) = sum(log(pdf(obj,y)));
    end
    conv(i) = obj.Converged;
    loglike_ind_prop = -log(K) + logsumexp(loglikes);
    
    if (exp(loglike_ind_prop - loglike_ind_curr) > rand)
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;
    end
    theta(i,:) = theta_curr;
    loglike_ind(i) = loglike_ind_curr;
    
end

end
