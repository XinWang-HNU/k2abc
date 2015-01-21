function results = run_iteration(whichmethod, opts, iter)
% mijung wrote on jan 21,2015

% inputs: 
% (1) whichmethod: ssf_kernel_abc (ours), rejection_abc, ssb_abc, and ssf_abc.
% (2) opts:
%            opts.likelihood_func: determine likelihood function
%            opts.true_theta: a vector of true parameters
%            opts.num_obs: # of observations (actual observation)
%            opts.num_theta_samps: # of samples for theta
%            opts.num_pseudodata_samps: # of samples for pseudo-data
%            opts.epsilon_list : list of epsilon to test 
%            opts.prior_var: prior variance to draw theta
% (3) seed number

%% (1) generate observations

% if opts.likelihood_func == 
dat = gen_sigmoid_pw_const(opts.true_theta, opts.num_obs, iter);

% figure(2);
% hist(dat.samps)

%% (2) test the chosen algorithm

% op. All options are described in ssf_kernel_abc.
op = struct();
op.seed = iter;
op.proposal_dist = @(n)randn(length(opts.true_theta), n)*sqrt(opts.prior_var);
op.likelihood_func = opts.likelihood_func;
op.epsilon_list = opts.epsilon_list; 
op.num_latent_draws = opts.num_theta_samps; 
op.num_pseudo_data = opts.num_pseudodata_samps;

% width squared.
% width2 = meddistance(dat.samps)^2/2;
width2 = meddistance(dat.samps)/2;
op.mmd_kernel = KGaussian(width2);
op.mmd_exponent = 2;

[R, op] = ssf_kernel_abc(dat.samps, op);

%% (3) outputing results of interest

cols = length(opts.true_theta);
num_eps = length(op.epsilon_list);
post_mean = zeros(num_eps, cols);
prob_post_mean = zeros(num_eps, cols);

for ei = 1:num_eps

    post_mean(ei,:) = R.latent_samples*R.norm_weights(:, ei) ;
    [~, prob_post_mean(ei,:)] = like_sigmoid_pw_const(post_mean(ei,:), 1);
    
end

results.post_mean = post_mean;
results.prob_post_mean = prob_post_mean;
results.dat = dat; 