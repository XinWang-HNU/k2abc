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
op.likelihood_func = @like_piecewise_const;
seed = iter;
dat = gen_piecewise_const(opts.true_theta, opts.num_obs, seed);

% figure(2);
% hist(dat.samps)

%% (2) test the chosen algorithm

% op. All options are described in each subfunction below.
op.seed = iter;

alpha = ones(length(opts.true_theta),1);
% samps = sample_from_dirichlet(alpha, n)
op.proposal_dist = @(n) sample_from_dirichlet(alpha,n);
% op.proposal_dist = @(n)randn(length(opts.true_theta), n)*sqrt(opts.prior_var);

op.num_latent_draws = opts.num_theta_samps;
op.num_pseudo_data = opts.num_pseudodata_samps;

if ismember(whichmethod, {'ssf_kernel_abc', 'k2abc' })
    %% (1) quadratic k2abc 
    
    % width squared.
%     width2 = meddistance(dat.samps)^2/2;
     width2 = meddistance(dat.samps)/32;
    op.mmd_kernel = KGaussian(width2);
    op.mmd_exponent = 2;
    
    op.epsilon_list = logspace(-3, 0, 9);
    
    [R, op] = ssf_kernel_abc(dat.samps, op);
    
    cols = length(opts.true_theta);
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
%     prob_post_mean = zeros(num_eps, cols);
    
    for ei = 1:num_eps    
        post_mean(ei,:) = R.latent_samples*R.norm_weights(:, ei) ;
        post_var(ei,:) = (R.latent_samples.^2)*R.norm_weights(:, ei) - (post_mean(ei,:).^2)'; 
    end
    
elseif strcmp(whichmethod, 'k2abc_lin')
    % K2ABC with linear MMD
     width2 = meddistance(dat.samps)/32;
    op.mmd_kernel = KGaussian(width2);
    op.mmd_exponent = 2;
    
    op.epsilon_list = logspace(-3, 0, 9);
    
    [R, op] = k2abc_lin(dat.samps, op);
    
    cols = length(opts.true_theta);
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
%     prob_post_mean = zeros(num_eps, cols);
    
    for ei = 1:num_eps    
        post_mean(ei,:) = R.latent_samples*R.norm_weights(:, ei) ;
        post_var(ei,:) = (R.latent_samples.^2)*R.norm_weights(:, ei) - (post_mean(ei,:).^2)'; 
    end

elseif strcmp(num2str(whichmethod),'rejection_abc')
    
    %% (2) rejection_abc
     % additional op for rejection abc
    op.stat_gen_func = @(data) [mean(data, 2) var(data,0,2)];
    op.stat_dist_func = @(stat1, stat2) norm(stat1 - stat2);
    op.threshold_func = @(dists, epsilons) bsxfun(@lt, dists(:), epsilons(:)');
    stat_scale = mean(abs(op.stat_gen_func(dat.samps)));
%     op.epsilon_list = logspace(-3, 0, 9);
    op.epsilon_list = logspace(-1.8, 0, 9)*stat_scale;
    
    [R, op] = ssb_abc(dat.samps, op);
    
    cols = length(opts.true_theta);
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
%     prob_post_mean = zeros(num_eps, cols);
    accpt_rate = zeros(num_eps, 1); 
    
    for ei = 1:num_eps
        idx_accpt_samps = R.unnorm_weights(:, ei);
        accpt_rate(ei) = sum(idx_accpt_samps)/opts.num_theta_samps;
        
        if accpt_rate(ei)>0
            post_mean(ei, :) = mean(R.latent_samples(:, idx_accpt_samps), 2) ;
            post_var(ei, :) = mean(R.latent_samples(:, idx_accpt_samps).^2, 2) - (post_mean(ei, :).^2)';
        end
        
    end
    
    results.accpt_rate = accpt_rate;

elseif strcmp(num2str(whichmethod),'ssb_abc')
    
  %% (3) soft abc  
    op.stat_gen_func = @(data) [mean(data, 2) var(data,0,2)];
    op.stat_dist_func = @(stat1, stat2) norm(stat1 - stat2);
    op.threshold_func = @(dists, epsilons) exp(-bsxfun(@times, dists(:), 1./epsilons(:)'));
    stat_scale = mean(abs(op.stat_gen_func(dat.samps)));
    op.epsilon_list = logspace(-2, 0, 9)*stat_scale;
    
    [R, op] = ssb_abc(dat.samps, op);
    
    cols = length(opts.true_theta);
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
%     prob_post_mean = zeros(num_eps, cols);
    
    for ei = 1:num_eps
        post_mean(ei,:) = R.latent_samples*R.unnorm_weights(:, ei)/sum(R.unnorm_weights(:, ei)) ;
        post_var(ei,:) = (R.latent_samples.^2)*R.unnorm_weights(:, ei)/sum(R.unnorm_weights(:, ei)) - (post_mean(ei,:).^2)';
    end
        
elseif strcmp(num2str(whichmethod),'ssf_abc')
    
else 
    
     disp('shit, sorry! we do not know which method you are talking about');

end

%% (3) outputing results of interest

results.post_mean = post_mean;
results.post_var = post_var;
% results.prob_post_mean = prob_post_mean;
results.dat = dat; 
results.epsilon_list = op.epsilon_list; 
