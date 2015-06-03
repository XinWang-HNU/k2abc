function [theta theta_d_samp abc_discs] = bayes_ind_inf_il2_gandk(y,M,n,cov_rw,abc_tol,numComp)

theta = zeros(M,4);

% MH - IL
theta_curr = [3 1 2 0.5];

abc_discs = zeros(M,1);
abc_disc = abc_tol;

theta_d_samp = zeros(M,3*numComp-1);

% fit auxiliary model
obj = gmdistribution.fit(y,numComp,'Options',statset('MaxIter',10000,'TolFun',1e-6));
loglike_ind_obs = sum(log(pdf(obj,y)));

theta_d = [obj.PComponents(1:(numComp-1)) obj.mu' sqrt(reshape(obj.Sigma,numComp,1))'];

for i = 1:M
    i
    theta_prop = mvnrnd(theta_curr,cov_rw);
    if (any(theta_prop<0) || any(theta_prop>10))
        theta(i,:) = theta_curr;
        abc_discs(i) = abc_disc;
        theta_d_samp(i,:) = theta_d;
        continue;
    end
    y_s = simulate_gk(n,[theta_prop(1:2) 0.8 theta_prop(3:4)]);
    % fit auxiliary model
    obj = gmdistribution.fit(y_s,numComp,'Options',statset('MaxIter',10000,'TolFun',1e-6));
    loglike_ind_sim = sum(log(pdf(obj,y)));
    
    theta_d_prop = [obj.PComponents(1:(numComp-1)) obj.mu' sqrt(reshape(obj.Sigma,numComp,1))'];
    
    dist_prop = abs(loglike_ind_sim-loglike_ind_obs);
    
    if (dist_prop <= abc_tol)
        theta_curr = theta_prop;
        abc_disc = dist_prop;
        theta_d = theta_d_prop;
    end
    theta(i,:) = theta_curr;
    theta_d_samp(i,:) = theta_d;
    abc_discs(i) = abc_disc;
    
end

end
