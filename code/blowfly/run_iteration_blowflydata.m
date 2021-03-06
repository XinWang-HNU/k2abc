function results = run_iteration_blowflydata(whichmethod, opts, iter)
% mijung wrote on jan 24,2015

% inputs: 
% (1) whichmethod: ssf_kernel_abc (ours), rejection_abc, ssb_abc, and ssf_abc.
% (2) opts:
%            opts.likelihood_func: determine likelihood function
%            opts.num_obs: # of observations (actual observation)
%            opts.num_theta_samps: # of samples for theta
%            opts.num_pseudodata_samps: # of samples for pseudo-data
%            opts.dim_theta: dimensionality of theta, it's 6 in this case
%            opts.yobs: observed data
% (3) seed number

%% (1) generate observations

% op. All options are described in each subfunction below.
op = opts;
op.seed = iter;

op.likelihood_func = @ gendata_pop_dyn_eqn; 
op.proposal_dist = @(n) sample_from_prior_blowflydata(n); 
op.num_latent_draws = opts.num_theta_samps;
op.num_pseudo_data = opts.num_pseudodata_samps;
op.dim_theta = opts.dim_theta; 

if ismember(whichmethod, {'ssf_kernel_abc', 'k2abc' })
    
    %% (1) ssf_kernel_abc
    
    % width squared.
%       width2 = meddistance(opts.yobs)^2*3;
%    width2 = meddistance(opts.yobs)^2;
%      width2 = meddistance(opts.yobs)^2;
    width2 = opts.width2; 
    op.mmd_kernel = KGaussian(width2);
    
    if size(opts.epsilon_list,1) ~=0
        op.epsilon_list = opts.epsilon_list;
    else
        op.epsilon_list = logspace(-5, 1, 10);
    end
%     op.epsilon_list = logspace(-3, 1, 10);
%     op.epsilon_list = logspace(-5, 0, 9);
%     op.epsilon_list = op.epsilon_list(1); 
%     op.epsilon_list = 1e-3; 
    
    [R, op] = ssf_kernel_abc(opts.yobs, op);
    
    cols = op.dim_theta;
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
%     prob_post_mean = zeros(num_eps, cols);
    
    for ei = 1:num_eps  
        latent_samples = R.latent_samples; 
        post_mean(ei,:) = latent_samples*R.norm_weights(:, ei) ;
        post_var(ei,:) = (latent_samples.^2)*R.norm_weights(:, ei) - (post_mean(ei,:).^2)'; 

    end
    
elseif strcmp(whichmethod, 'k2abc_lin')
    % K2ABC with linear MMD
    %
    width2 = opts.width2; 
    op.mmd_kernel = KGaussian(width2);
    
    if size(opts.epsilon_list,1) ~=0
        op.epsilon_list = opts.epsilon_list;
    else
        op.epsilon_list = logspace(-5, 1, 10);
    end
    
    [R, op] = k2abc_lin(opts.yobs, op);
    
    cols = op.dim_theta;
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
    
    for ei = 1:num_eps  
        latent_samples = R.latent_samples; 
        post_mean(ei,:) = latent_samples*R.norm_weights(:, ei) ;
        post_var(ei,:) = (latent_samples.^2)*R.norm_weights(:, ei) - (post_mean(ei,:).^2)'; 

    end

elseif strcmp(whichmethod, 'k2abc_rf')
    % K2ABC with linear MMD
    %
    width2 = opts.width2; 
    ker = KGaussian(width2);
    nfeatures = 50;
    input_dim = size(opts.yobs, 2);
    fm = ker.getRandFeatureMap(nfeatures, input_dim);
    % Set the random feature map for k2abc_rf
    op.feature_map = fm;
    
    if size(opts.epsilon_list,1) ~=0
        op.epsilon_list = opts.epsilon_list;
    else
        op.epsilon_list = logspace(-5, 1, 10);
    end
    
    [R, op] = k2abc_rf(opts.yobs, op);
    
    cols = op.dim_theta;
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
    
    for ei = 1:num_eps  
        latent_samples = R.latent_samples; 
        post_mean(ei,:) = latent_samples*R.norm_weights(:, ei) ;
        post_var(ei,:) = (latent_samples.^2)*R.norm_weights(:, ei) - (post_mean(ei,:).^2)'; 

    end

elseif strcmp(num2str(whichmethod),'rejection_abc')
    
    %% (2) rejection_abc
     % additional op for rejection abc
    op.stat_gen_func = @(data) [mean(data, 2) var(data,0,2)];
    op.stat_dist_func = @(stat1, stat2) norm(stat1 - stat2);
    op.threshold_func = @(dists, epsilons) bsxfun(@lt, dists(:), epsilons(:)');
    stat_scale = mean(abs(op.stat_gen_func(op.yobs)));
%     op.epsilon_list = logspace(-3, 0, 9);
    op.epsilon_list = logspace(-1.8, 0, 9)*stat_scale;
    
    [R, op] = ssb_abc(op.yobs, op);
    
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
    stat_scale = mean(abs(op.stat_gen_func(op.yobs)));
    op.epsilon_list = logspace(-2, 0, 9)*stat_scale;
    
    [R, op] = ssb_abc(op.yobs, op);
    
    cols = length(opts.true_theta);
    num_eps = length(op.epsilon_list);
    post_mean = zeros(num_eps, cols);
    post_var = zeros(num_eps, cols);
%     prob_post_mean = zeros(num_eps, cols);
    
    for ei = 1:num_eps
        post_mean(ei,:) = R.latent_samples*R.unnorm_weights(:, ei)/sum(R.unnorm_weights(:, ei)) ;
        post_var(ei,:) = (R.latent_samples.^2)*R.unnorm_weights(:, ei)/sum(R.unnorm_weights(:, ei)) - (post_mean(ei,:).^2)';
    end
        
%elseif strcmp(num2str(whichmethod),'ssf_abc')
%
elseif strcmp(num2str(whichmethod),'kabc_cond_embed')
    % kernel abc of Nakagome et.al.

    stat_gen_func = @ss_for_blowflydata;
    % construct a training set
    train_params = op.proposal_dist(op.num_latent_draws);
    train_stats = [];
    %assert(size(train_params, 2) == size(train_stats, 2));
    % for each single parameter, we need to construct a summary statistic of 
    % observations generated by the parameter.
    for i=1:size(train_params, 2)
        theta = train_params(:, i);
        observations = op.likelihood_func(theta, op.num_pseudo_data);
        stat = stat_gen_func(observations);
        train_stats(:, i) = stat;
    end

    % ------- options for kabc ------------
    % a list of regularization parameter candidates in kabc. 
    % Chosen by cross validation.
    ntr = op.num_latent_draws;
    op.kabc_reg_list = 10.^(-6:1:2);
    % number of subsamples to be used during cross validation. 
    % Lower => speed up
    op.kabc_cv_num_subsamples = min(ntr, 5000) ;
    % number of folds to perform in cross validation
    op.kabc_cv_fold = 3;
    % a list of Gaussian widths squared to be used as candidates for Gaussian kernel
    op.kabc_gwidth2_list = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16].* (meddistance(train_stats).^2);

    % ---- training ------
    [R, op] = kabc_cond_embed(train_stats, train_params, op);
    % R contains a regression function mapping from a stat to its param
    %

    % test on the actual observations
    test_stats = stat_gen_func(opts.yobs);
    unnorm_weights = R.regress_weights_func(test_stats);
    R.latent_samples = train_params;
    R.unnorm_weights = unnorm_weights;
    R.train_stats = train_stats;
    R.train_params = train_params;

    post_mean = R.latent_samples*R.unnorm_weights(:);
    % Don't need to divide by the sum of weights here.
    post_var = (R.latent_samples.^2)*R.unnorm_weights(:) - (post_mean.^2);

else 

    disp('shit, sorry! we do not know which method you are talking about');

end

%% (3) outputing results of interest

results.post_mean = post_mean;
results.post_var = post_var;
% results.prob_post_mean = prob_post_mean;
% results.dat = dat; 
results.R = R; 
results.op = op;
if isfield(op, 'epsilon_list')
    results.epsilon_list = op.epsilon_list; 
end
