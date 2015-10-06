% compute an error on a held-out blowfly data set
% this code is what I used for Fig 2
% mijung wrote and tested on jan 27, 2015

clear all;
clf;
close all;

%% (1) load data

load flydata.mat

seed = 12;
oldRng = rng();
rng(seed);

n = length(flydata);

%opts.num_theta_samps = 1000;
opts.num_theta_samps = 2000;
%opts.num_pseudodata_samps = 4*ntr;
opts.num_pseudodata_samps = 2000;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata';

%whichmethod = 'ssf_kernel_abc';
whichmethod = 'k2abc_lin';
%whichmethod = 'k2abc_rf';
med = meddistance(opts.yobs);

if strcmp(whichmethod, 'ssf_kernel_abc')
    
    howmanyscalelength = 20;
    width2mat = med^2.*2.^linspace(-10, 4, howmanyscalelength);
    
    howmanyepsilon = 10;
    opts.epsilon_list = logspace(-6, 1, howmanyepsilon);
    
elseif strcmp(whichmethod, 'k2abc_lin')
    
    howmanyscalelength = 5;
    %width2mat = meddistance(opts.yobs)^2.*2.^linspace(-8, -1, howmanyscalelength);
    
    med_factors =  2.^linspace(2, 5, howmanyscalelength);
    width2mat = (med^2)*med_factors;
    %howmanyscalelength = length(med_factors);
    
    howmanyepsilon = 30;
    opts.epsilon_list = logspace(-5, -1, howmanyepsilon);
    
elseif strcmp(whichmethod, 'k2abc_rf')
    howmanyscalelength = 10;
    %width2mat = meddistance(opts.yobs)^2.*2.^linspace(-8, -1, howmanyscalelength);
    
    med_factors =  2.^linspace(-1, 5, howmanyscalelength);
    %med_factors = med_factors(6:end);
    %howmanyscalelength = length(med_factors);

    width2mat = (med^2)*med_factors;
    %howmanyscalelength = length(med_factors);
    
    howmanyepsilon = 30;
    opts.epsilon_list = logspace(-5, -1, howmanyepsilon);


end
display(sprintf('Observation median dist^2: %.3f', med^2 ));
%%
maxiter = length(width2mat);

% split data into training and test data
last_idx_trn = ceil(n*3/4);
idx_trn = [ones(1, last_idx_trn), zeros(1, n - last_idx_trn)];
assert(length(idx_trn) == n);

% training indices
ntr = sum(idx_trn);

opts.num_obs = ntr;
opts.yobs = flydata(1:ntr)';

for iter = 1 : howmanyscalelength

    display(sprintf('(%d/%d) Running %s with Gauss. width^2: %.3f', ...
        iter, howmanyscalelength, whichmethod, width2mat(iter)));
    
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
idx_tst = [zeros(1, last_idx_trn), ones(1, n-last_idx_trn)];
testdat = flydata( (last_idx_trn+1):n);
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
        %[i j]
        
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


[minIdx1, minIdx2] = ind2sub([howmanyscalelength, howmanyepsilon], find(min(min(avg_loss_mat)) == avg_loss_mat,2));
display(sprintf('best width2: %.3f, index: %d', width2mat(minIdx1), minIdx1));
display(sprintf('best eps: %.3g, index: %d', opts.epsilon_list(minIdx2) , minIdx2));
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

epsilon_opt = opts.epsilon_list(minIdx2);
width_opt = width2mat(minIdx1);

%%

% now we have optimal params which we use to run our method using the entire dataset

% opts.epsilon_list = logspace(-6, 1, howmanyepsilon);
opts.epsilon_list = epsilon_opt;

opts.num_obs = n;
opts.dim_theta = 6; % dim(theta)
opts.yobs = flydata';

opts.width2 = width_opt;
% opts.width2 =  width2mat(minIdx1);

if strcmp(whichmethod, 'ssf_kernel_abc')
    
    results = run_iteration_blowflydata(whichmethod, opts, 40);
    save(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results');
    load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results');
    
elseif strcmp(whichmethod, 'k2abc_lin')
    
    results = run_iteration_blowflydata(whichmethod, opts, 3);
    opt_k2abc_lin = results.post_mean;
    save opt_k2abc_lin opt_k2abc_lin;
    
elseif strcmp(whichmethod, 'k2abc_rf')
    
    results = run_iteration_blowflydata(whichmethod, opts, 9);
    opt_k2abc_rf = results.post_mean;
    save opt_k2abc_rf opt_k2abc_rf;
    
else
    
    results = run_iteration_blowflydata(whichmethod, opts, 9);
    
end

% this is what I used for k2abc

simuldat_ours = gendata_pop_dyn_eqn(results.post_mean, n);
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

s = ss_for_blowflydata(flydata);
mse = @(a) norm(s-a);

s_ours =  ss_for_blowflydata(simuldat_ours);
% mse(s_ours)

load flydata.mat
num_rept_mse = 100;
msemat = zeros(num_rept_mse, 1);

for i=1:num_rept_mse
    simuldat_ours = gendata_pop_dyn_eqn(results.post_mean, n);
    
    s_ours =  ss_for_blowflydata(simuldat_ours);
    msemat(i) = mse(s_ours);
end

display(sprintf('MSE between stats. of generated data and observations: %.4f', mean(msemat) ));


