% to make Figure 2
clear all;
clc;
clf;

load flydata.mat
n = length(flydata);

% Fig2 (A): show prior histogram and posterior histgram (our method)

prior_samps = sample_from_prior_blowflydata(10000);
subplot(2, 6, 1); hist(log(prior_samps(1,:))); xlabel('logP'); ylabel('prior'); set(gca, 'xlim', 2+2.*[-5 5]); box off;
subplot(2, 6, 2); hist(log(prior_samps(2,:))); xlabel('log delta'); set(gca, 'xlim', -1+0.4.*[-5 5]);  box off;
subplot(2, 6, 3); hist(log(prior_samps(3,:))); xlabel('log N0'); set(gca, 'xlim', 5+0.5.*[-5 5]);  box off;
subplot(2, 6, 4); hist(log(prior_samps(4,:))); xlabel('log sig d'); set(gca, 'xlim', -0.5+ [-5 5]);  box off;
subplot(2, 6, 5); hist(log(prior_samps(5,:))); xlabel('log sig p'); set(gca, 'xlim', -0.5+ [-5 5]);  box off;
subplot(2, 6, 6); hist(log(prior_samps(6,:))); xlabel('log tau'); set(gca, 'xlim', 2+ [-5 5]);  box off;

% plot(1:n, flydata/1000,'k.-'); ylabel('yobs (x1000)'); xlabel('time');

whichmethod = 'ssf_kernel_abc';
minIdx1= 11;
minIdx2 = 1; 
load(strcat('blowflydata: ', num2str(whichmethod), '_thLengthScale', num2str(minIdx1), '_thxvset', '.mat'));
params_ours = results.post_mean(minIdx2,:);

posterior_samps = bsxfun(@times, (results.R.latent_samples'), results.R.norm_weights(:, minIdx2)); 
subplot(2, 6, 7); hist(log(posterior_samps(:, 1))); xlabel('logP'); ylabel('posterior'); 
% set(gca, 'xlim', 2+2.*[-5 5]); box off;
subplot(2, 6, 8); hist(log(posterior_samps(:, 2))); xlabel('log delta'); 
% set(gca, 'xlim', -1+0.4.*[-5 5]);  box off;
subplot(2, 6, 9); hist(log(posterior_samps(:, 3))); xlabel('log N0'); 
% set(gca, 'xlim', 5+0.5.*[-5 5]);  box off;
subplot(2, 6, 10); hist(log(posterior_samps(:, 4))); xlabel('log sig d'); 
% set(gca, 'xlim', -0.5+ [-5 5]);  box off;
subplot(2, 6, 11); hist(log(posterior_samps(:, 5))); xlabel('log sig p'); 
% set(gca, 'xlim', -0.5+ [-5 5]);  box off;
subplot(2, 6, 12); hist((posterior_samps(:, 6))); xlabel('log tau'); 
% set(gca, 'xlim', 2+ [-5 5]);  box off;






%%

params_ours = results.post_mean(minIdx2,:);
simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])













%%

% load thetas_sl.mat; % this is synthetic likelihood abc results, acptrate
% = 0.42
% load thetas_sl_ep_point1.mat % accpt rate is 0.26
% load thetas_sl_ep_point01.mat

% whichmethod =  'ssf_kernel_abc';
% iter = 4; 
% load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'))

% params_ours = results.post_mean(1,:); 
% simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

%%%%%%%%%%% kernel mean embedding %%%%%%%%%%%%%%
% whichmethod = 'kabc_cond_embed';
% iter = 1; 
% load(strcat('blowflydata_', num2str(whichmethod), '_thIter', num2str(iter), '.mat'))
% 
% params = results.post_mean; 
% simuldat = gendata_pop_dyn_eqn(params_ours, n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(161); hist(thetas(:,1));
subplot(162); hist(thetas(:,2));
subplot(163); hist(thetas(:,3));
subplot(164); hist(thetas(:,4));
subplot(165); hist(thetas(:,5));
subplot(166); hist(thetas(:,6));

%%
% params_sl = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
params_sl = mean(thetas);
params_sl = [exp(params_sl(1:5)) params_sl(end)];
simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);

subplot(211); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_sl/1000) + 1])
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_sl/1000) + 1])

% compute chosen summary statistics
s = ss_for_blowflydata(flydata);
% s_ours =  ss_for_blowflydata(simuldat_ours);
s_kabc = ss_for_blowflydata(flydata);
s_sl = ss_for_blowflydata(simuldat_sl);

mse = @(a) norm(s-a);
[mse(s) mse(s_kabc) mse(s_sl)]

%% compute avg loss

howmanyscalelength = 10; 
width2_list = meddistance(flydata)^2.*logspace(-2,2,howmanyscalelength); 
howmanyepsilon = 9; 
epsilon_list = logspace(-5, 0, howmanyepsilon);

avg_loss_mat = zeros(howmanyscalelength, howmanyepsilon);

whichmethod =  'ssf_kernel_abc';
opts.likelihood_func = @ gendata_pop_dyn_eqn; 

s_true = ss_for_blowflydata(flydata);
% opts.obj = @(a) norm(hist(flydata)-hist(a));
opts.obj = @(a) sqrt(sum((flydata'-a).^2)/n);
% opts.obj = @(a) norm(s_true-ss_for_blowflydata(a));
opts.num_samps = n;
opts.num_rep = 100;

for i=1:howmanyscalelength
    
    load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(i), '.mat'));
    
    for j=1:howmanyepsilon
        [i j]
        opts.params = results.post_mean(j,:); 
        avg_loss_mat(i,j) = compute_loss_for_epsilon_kernelparam(opts); 
    end
    
end

%%

[minIdx1, minIdx2] = ind2sub([howmanyscalelength, howmanyepsilon], find(min(min(avg_loss_mat)) == avg_loss_mat,2));
subplot(211); plot(avg_loss_mat')

load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(minIdx1), '.mat'));
params_ours = results.post_mean(minIdx2,:); 
simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

% load(strcat('blowflydata: ', num2str(whichmethod), '_medianHeuristic', '.mat'));
% params_ours = results.post_mean(2,:); 
% simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

%%

opts_sl = opts; 
opts_sl.params = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
% simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);
avg_loss_sl = compute_loss_for_epsilon_kernelparam(opts_sl); 
[min(min(avg_loss_mat)) avg_loss_sl]
