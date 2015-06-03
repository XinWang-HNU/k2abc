function [ results, op ] = indirect_score_abc( Obs, op )

%% I copied Christopher Drovandi's code , then modified a few things to use

% op has the following:
% op.likelihood_func = @ gendata_pop_dyn_eqn; 
% op.proposal_dist = @(n) sample_from_prior_blowflydata(n); 
% op.num_latent_draws = opts.num_theta_samps;
% op.num_pseudo_data = opts.num_pseudodata_samps;
% op.dim_theta = opts.dim_theta; 

%% mijung comments : size(theta) is how many samples by number of parameters

theta = zeros(op.num_pseudo_data,op.dim_theta);
% abc_discs = zeros(op.num_latent_draws,1);
% abc_disc = op.abc_tol;
% summary_stats_samp = zeros(op.num_latent_draws,3*op.numComp-1);
% summary_stats = zeros(1,3*op.numComp-1);

% mijung comments: fitting data to gaussian mixture
obj = gmdistribution.fit(Obs,op.numComp,'Options',statset('MaxIter',100000,'TolFun',1e-10));
theta_d = [obj.PComponents(1:(op.numComp-1)) obj.mu' reshape(obj.Sigma,op.numComp,1)'];

weight_matrix = compute_obs_inf(theta_d,Obs,obj,op.numComp);
weight_matrix = inv(weight_matrix);

grad_true =  compute_grad(theta_d,Obs,obj,op.numComp); 
dist_prop_tol = grad_true*weight_matrix*grad_true';

% theta_curr = zeros(1, op.dim_theta);

for i = 1:op.num_pseudo_data
%     i
%     theta_prop = mvnrnd(theta_curr,cov_rw);
    theta_prop = op.proposal_dist(1);

        % mijung comments : here y_s is simulated data, size(y_s) = n by 1
%     y_s = simulate_gk(n,[theta_prop(1:2) 0.8 theta_prop(3:4)]);
    y_s = op.likelihood_func(theta_prop, op.num_pseudo_data)'; 
    
    %[grad,~,~] = gradest(@(x)log_like_mixture(x,y_s),theta_d);
    grad = compute_grad(theta_d,y_s,obj,op.numComp);
    
    dist_prop = grad*weight_matrix*grad';
    
    if (dist_prop <= dist_prop_tol*1e8)
%         [dist_prop]
        theta(i,:)  = theta_prop;
%         abc_disc = dist_prop;
%         summary_stats = grad;
    end
%     summary_stats_samp(i,:) = summary_stats;
%     abc_discs(i) = abc_disc;
    
end


results = struct();
results.latent_samples = theta;
% results.unnorm_weights = unnorm_weights;

%%
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
%     
%     [t pdf(obj,y(t))]
    
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