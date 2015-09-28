% cross-validation for testblowdata
% this code is what I used for Fig 2
% mijung wrote and tested on jan 27, 2015

clear all;
clc;
clf;

%% (1) load data

load flydata.mat

seed = 11;
oldRng = rng();
rng(seed);

n = length(flydata);

%% test ssf-abc

%whichmethod =  'ssf_kernel_abc';
whichmethod =  'k2abc_lin';

opts.num_theta_samps = 1000;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata';

% howmanyscalelength = 10;
% width2mat = meddistance(opts.yobs)^2.*logspace(-1,0,howmanyscalelength);
% width2mat = meddistance(opts.yobs)^2.;
howmanyscalelength = 20;
% width2mat = meddistance(opts.yobs)^2.*logspace(-4,2,howmanyscalelength);
width2mat = meddistance(opts.yobs)^2.*2.^linspace(-10, 4, howmanyscalelength);
maxiter = length(width2mat); 

% split data into training and test data
cv_fold = 1;
last_idx_trn = ceil(n*3/4);
idx_trn = [ones(1, last_idx_trn), zeros(1, n - last_idx_trn)];
assert(length(idx_trn) == n);

% for fi=1:cv_fold

% training indices
ntr = sum(idx_trn);

opts.num_obs = ntr;
% opts.num_pseudodata_samps = 1000; 
opts.num_pseudodata_samps = 4*ntr;
opts.yobs = flydata(1:ntr)';

howmanyepsilon = 10;
opts.epsilon_list = logspace(-6, 1, howmanyepsilon);

for iter = 1 : howmanyscalelength
    
    [iter howmanyscalelength] 
    
    opts.width2 = width2mat(iter);
    
    seed = iter + 20; 
    results = run_iteration_blowflydata(whichmethod, opts, seed);
    
    %     save results
    save(strcat('blowflydata: ', num2str(whichmethod), '_thLengthScale', num2str(iter), '_thxvset_w_higher_epsilon', '.mat'), 'results');
 
end

%% choose which one is the best

avg_loss_mat = zeros(howmanyscalelength, howmanyepsilon);

%whichmethod =  'ssf_kernel_abc';
opts.likelihood_func = @ gendata_pop_dyn_eqn;
opts.num_rep = 100;
% 
% % for fi=1:cv_fold
idx_tst = [zeros(1, n*3/4) ones(1, n/4)];
testdat = flydata(n*3/4+1:n)';
% 

s_true = ss_for_blowflydata(testdat);
% nbins = 10; 
opts.obj = @(a) norm(hist(testdat)-hist(a));
% % opts.obj = @(a) sqrt(sum((testdat-a).^2)/n);
% opts.obj = @(a) norm(s_true-ss_for_blowflydata(a));
% % opts.obj = @(a, b) sqrt(sum((a-b).^2)/(n/4));
opts.num_samps = length(testdat);


for i=1:howmanyscalelength
    
       load(strcat('blowflydata: ', num2str(whichmethod), '_thLengthScale', num2str(i), '_thxvset_w_higher_epsilon', '.mat'));
%     load(strcat('blowflydata: ', num2str(whichmethod), '_thLengthScale', num2str(i), '_thxvset', '.mat'));
    
    for j=1:howmanyepsilon
        [i j]
        
        opts.params = results.post_mean(j,:);
        %%
        %         simuldat_ours = gendata_pop_dyn_eqn(opts.params, n/4);
        %         subplot(212); plot(1:n/4, testdat/1000, 'k', 1:n/4, simuldat_ours./1000, 'r-'); title('simulated data');
        %         set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])
        %         [alignedX, alignedY] = samplealign(sort(testdat)', sort(simuldat_ours)');
        %          avg_loss_mat(i,j) = opts.obj(alignedX, alignedY);
        %         pause;
        %%
        
        if sum(isnan(opts.params))>0
            avg_loss_mat(i,j) = 100;
        else
            avg_loss_mat(i,j) = compute_loss_for_epsilon_kernelparam(testdat, opts);
        end
    end
    
    %     end
    
end


[minIdx1, minIdx2] = ind2sub([howmanyscalelength, howmanyepsilon], find(min(min(avg_loss_mat)) == avg_loss_mat,2))
subplot(211); plot(avg_loss_mat'); 
% legend('l=2^-5*median', 'l=2^-4*median', 'l=2^-3*median', 'l=2^-2*median', 'l=2^-1*median', 'l=median', 'l=2*median', 'l=2^2*median', 'l=2^3*median', 'l=2^4*median', 'l=2^5*median');
% ylabel('prediction on test data (hist)'); xlabel('epsilon'); 
% set(gca, 'xscale', 'log');

params_ours = results.post_mean(minIdx2,:); 
% params_ours = results.post_mean(1, :);

% err = mean(abs(params_ours - true_params)./true_params)

simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

%%

% now we have optimal params which we use to run our method using the entire dataset 

opts.epsilon_list = logspace(-6, 1, howmanyepsilon);
opts.epsilon_list = opts.epsilon_list(minIdx2);

opts.num_obs = n;
opts.num_theta_samps = 1000;
opts.num_pseudodata_samps = n*4;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata';

opts.width2 = width2mat(minIdx1);
% results = run_iteration_blowflydata(whichmethod, opts, 4);
results = run_iteration_blowflydata(whichmethod, opts, 40);

% save(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results');
% load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results'); 
% params_ours = results.post_mean(1,:);

%%
simuldat_ours = gendata_pop_dyn_eqn(results.post_mean, n);
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

s = ss_for_blowflydata(flydata);
s_ours =  ss_for_blowflydata(simuldat_ours);
% s_kabc = ss_for_blowflydata(simuldat);
% s_sl = ss_for_blowflydata(simuldat_sl);

mse = @(a) norm(s-a);
[mse(s) mse(s_ours)]

