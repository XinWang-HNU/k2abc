% to make Figure 2
clear all;
clc;
% clf;

oldRng = rng();
seed = 1;
rng(seed);

load flydata.mat
n = length(flydata);

% Fig2 (A): show prior histogram and posterior histgram (our method)

prior_samps = sample_from_prior_blowflydata(10000);
nbin = 10; 
subplot(3, 6, 1); hist(log(prior_samps(1,:)), nbin); xlabel('logP'); 
hold on; plot(mean(log(prior_samps(1,:))), 1:5000, 'r-');
set(gca, 'xlim', 2+2.*[-5 5]); box off;

subplot(3, 6, 2); hist(log(prior_samps(2,:)), nbin); xlabel('log delta'); 
hold on; plot(mean(log(prior_samps(2,:))), 1:5000, 'r-');
set(gca, 'xlim', -1+0.4.*[-5 5]);  box off;

subplot(3, 6, 3); hist(log(prior_samps(3,:)), nbin); xlabel('log N0'); 
hold on; plot(mean(log(prior_samps(3,:))), 1:5000, 'r-');
set(gca, 'xlim', 5+0.5.*[-5 5]);  box off;

subplot(3, 6, 4); hist(log(prior_samps(4,:)), nbin); xlabel('log sig d'); 
hold on; plot(mean(log(prior_samps(4,:))), 1:5000, 'r-');
set(gca, 'xlim', -0.5+ [-5 5]);  box off;

subplot(3, 6, 5); hist(log(prior_samps(5,:)), nbin); xlabel('log sig p'); 
hold on; plot(mean(log(prior_samps(5,:))), 1:5000, 'r-');
set(gca, 'xlim', -0.5+ [-5 5]);  box off;

subplot(3, 6, 6); hist(log(prior_samps(6,:)), nbin); xlabel('log tau'); 
hold on; plot(mean(log(prior_samps(6,:))), 1:5000, 'r-');
set(gca, 'xlim', 2+ [-5 5]);  box off;

% plot(1:n, flydata/1000,'k.-'); ylabel('yobs (x1000)'); xlabel('time');


%% synthetic likelihood ABC

% load thetas_sl.mat
% load thetas_sl_ep_point01.mat
load thetas_sl_ep_point1.mat % accpt rate is 0.26

nbin = 3; 

subplot(3,6,7); hist(thetas(:,1), nbin); 
hold on; plot(mean(thetas(:,1)), 1:5000, 'r-');
set(gca, 'xlim', 2+2.*[-5 5], 'ylim', [0 5000]); box off;

subplot(3,6,8); hist(thetas(:,2), nbin); 
hold on; plot(mean(thetas(:,2)), 1:5000, 'r-'); set(gca, 'xlim', -1+0.4.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,9); hist(thetas(:,3), nbin); 
hold on; plot(mean(thetas(:,3)), 1:5000, 'r-'); set(gca, 'xlim', 5+0.5.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,10); hist(thetas(:,4), nbin); 
hold on; plot(mean(thetas(:,4)), 1:5000, 'r-'); set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,11); hist(thetas(:,5), nbin);  
hold on; plot(mean(thetas(:,5)), 1:5000, 'r-'); set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,12); hist(log(thetas(:,6)), nbin); 
hold on; plot(mean(log(thetas(:,6))), 1:5000, 'r-'); set(gca, 'xlim', 2+ [-5 5], 'ylim', [0 5000]);  box off;


%%
whichmethod = 'ssf_kernel_abc';

% load(strcat('blowflydata: ', num2str(whichmethod), '_thLengthScale', num2str(minIdx1), '_thxvset', '.mat'));

minIdx1= 11;
minIdx2 =2 ;
load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results'); 

% minIdx1= 16;
% minIdx2 =2;
% load(strcat('blowflydata: ', num2str(whichmethod), '_thLengthScale', num2str(minIdx1), '_thxvset_w_higher_epsilon', '.mat'));
% 
% params_ours = results.post_mean(minIdx2,:);
weightvec = results.R.norm_weights(:, minIdx2);
theta_samps_prior  = results.R.latent_samples; 

idx_to_samp = discrete_rnd(weightvec', 1, 1e4);

subplot(3, 6, 13);hist(log(theta_samps_prior(1,idx_to_samp)), nbin);
hold on; plot(mean(log(theta_samps_prior(1,idx_to_samp))), 1:5000, 'r-');
set(gca, 'xlim', 2+2.*[-5 5], 'ylim', [0 5000]); box off;

subplot(3, 6, 14); hist(log(theta_samps_prior(2,idx_to_samp)), nbin); 
hold on; plot(mean(log(theta_samps_prior(2,idx_to_samp))), 1:5000, 'r-');
set(gca, 'xlim', -1+0.4.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3, 6, 15); hist(log(theta_samps_prior(3,idx_to_samp)), nbin); 
hold on; plot(mean(log(theta_samps_prior(3,idx_to_samp))), 1:5000, 'r-');
set(gca, 'xlim', 5+0.5.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3, 6, 16); hist(log(theta_samps_prior(4,idx_to_samp)), nbin);  
hold on; plot(mean(log(theta_samps_prior(4,idx_to_samp))), 1:5000, 'r-');
set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3, 6, 17); hist(log(theta_samps_prior(5,idx_to_samp)), nbin); 
hold on; plot(mean(log(theta_samps_prior(5,idx_to_samp))), 1:5000, 'r-');
set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3, 6, 18); hist(log(theta_samps_prior(6,idx_to_samp)), nbin); 
hold on; plot(mean(log(theta_samps_prior(6,idx_to_samp))), 1:5000, 'r-');
set(gca, 'xlim', 2+ [-5 5], 'ylim', [0 5000]);  box off;

% params_ours = results.post_mean(minIdx2,:);
% simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);
% 
% subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
% set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

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

% subplot(161); hist(thetas(:,1));
% subplot(162); hist(thetas(:,2));
% subplot(163); hist(thetas(:,3));
% subplot(164); hist(thetas(:,4));
% subplot(165); hist(thetas(:,5));
% subplot(166); hist(thetas(:,6));

% params_sl = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]);

%% indirect_score_abc results

iter = 1; 
whichmethod =  'indirect_score';
load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(iter), '.mat'), 'results');

thetas = results.R.latent_samples;
TF = thetas==0;
TFrow = ~all(TF,2);
thetas = thetas(TFrow,:);

nbin = 3; 

subplot(3,6,7); hist(thetas(:,1), nbin); 
hold on; plot(mean(thetas(:,1)), 1:5000, 'r-');
set(gca, 'xlim', 2+2.*[-5 5], 'ylim', [0 5000]); box off;

subplot(3,6,8); hist(thetas(:,2), nbin); 
hold on; plot(mean(thetas(:,2)), 1:5000, 'r-'); set(gca, 'xlim', -1+0.4.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,9); hist(thetas(:,3), nbin); 
hold on; plot(mean(thetas(:,3)), 1:5000, 'r-'); set(gca, 'xlim', 5+0.5.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,10); hist(thetas(:,4), nbin); 
hold on; plot(mean(thetas(:,4)), 1:5000, 'r-'); set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,11); hist(thetas(:,5), nbin);  
hold on; plot(mean(thetas(:,5)), 1:5000, 'r-'); set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,12); hist(log(thetas(:,6)), nbin); 
hold on; plot(mean(log(thetas(:,6))), 1:5000, 'r-'); set(gca, 'xlim', 2+ [-5 5], 'ylim', [0 5000]);  box off;

indirect_score_abc = mean(thetas); 


%% rejection-SA_ABC     

load rejection-SA-ABC.mat

thetas = zeros(1000, 6);
thetas(:,1) = PosteriorSample.P;
thetas(:,2) = PosteriorSample.delta;
thetas(:,3) = PosteriorSample.N0;
thetas(:,4) = PosteriorSample.sigd;
thetas(:,5) = PosteriorSample.sigp;
thetas(:,6) = PosteriorSample.tau;

nbin = 3; 

subplot(3,6,7); hist(thetas(:,1), nbin); 
hold on; plot(mean(thetas(:,1)), 1:5000, 'r-');
set(gca, 'xlim', 2+2.*[-5 5], 'ylim', [0 5000]); box off;

subplot(3,6,8); hist(thetas(:,2), nbin); 
hold on; plot(mean(thetas(:,2)), 1:5000, 'r-'); set(gca, 'xlim', -1+0.4.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,9); hist(thetas(:,3), nbin); 
hold on; plot(mean(thetas(:,3)), 1:5000, 'r-'); set(gca, 'xlim', 5+0.5.*[-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,10); hist(thetas(:,4), nbin); 
hold on; plot(mean(thetas(:,4)), 1:5000, 'r-'); set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,11); hist(thetas(:,5), nbin);   
hold on; plot(mean(thetas(:,5)), 1:5000, 'r-'); set(gca, 'xlim', -0.5+ [-5 5], 'ylim', [0 5000]);  box off;

subplot(3,6,12); hist(log(thetas(:,6)), nbin); 
hold on; plot(mean(log(thetas(:,6))), 1:5000, 'r-'); set(gca, 'xlim', 2+ [-5 5], 'ylim', [0 5000]);  box off;

reject_sa_abc = mean(thetas); 

%% 

load rejection-SA-ABCQ.mat

thetas = zeros(1000, 6);
thetas(:,1) = PosteriorSample.P;
thetas(:,2) = PosteriorSample.delta;
thetas(:,3) = PosteriorSample.N0;
thetas(:,4) = PosteriorSample.sigd;
thetas(:,5) = PosteriorSample.sigp;
thetas(:,6) = PosteriorSample.tau;

reject_sa_abc_q = mean(thetas); 

%%

load rejection-SA-ABC-WoodSS.mat

thetas = zeros(1000, 6);
thetas(:,1) = PosteriorSample.P;
thetas(:,2) = PosteriorSample.delta;
thetas(:,3) = PosteriorSample.N0;
thetas(:,4) = PosteriorSample.sigd;
thetas(:,5) = PosteriorSample.sigp;
thetas(:,6) = PosteriorSample.tau;

reject_sa_abc_woodss = mean(thetas); 


%%

load weighted-SA-ABC-WoodSS.mat

thetas = zeros(10000, 6);
thetas(:,1) = PosteriorSample.P;
thetas(:,2) = PosteriorSample.delta;
thetas(:,3) = PosteriorSample.N0;
thetas(:,4) = PosteriorSample.sigd;
thetas(:,5) = PosteriorSample.sigp;
thetas(:,6) = PosteriorSample.tau;

weighted_sa_abc_woodss = mean(thetas); 

%%

load weighted-SA-ABCQ.mat
thetas = zeros(10000, 6);
thetas(:,1) = PosteriorSample.P;
thetas(:,2) = PosteriorSample.delta;
thetas(:,3) = PosteriorSample.N0;
thetas(:,4) = PosteriorSample.sigd;
thetas(:,5) = PosteriorSample.sigp;
thetas(:,6) = PosteriorSample.tau;

weighted_sa_abc_q= mean(thetas); 

%%

load weighted-SA-ABC.mat

thetas = zeros(10000, 6);
thetas(:,1) = PosteriorSample.P;
thetas(:,2) = PosteriorSample.delta;
thetas(:,3) = PosteriorSample.N0;
thetas(:,4) = PosteriorSample.sigd;
thetas(:,5) = PosteriorSample.sigp;
thetas(:,6) = PosteriorSample.tau;

weighted_sa_abc = mean(thetas); 

%% computing mse on ss

load flydata.mat
n = length(flydata);

num_rept_mse = 100;
msemat = zeros(num_rept_mse, 10);
s = ss_for_blowflydata(flydata);
mse = @(a) norm(s-a);
    
for i=1:num_rept_mse
    
    
    %% synthetic likelihood abc
    load thetas_sl_ep_point1.mat % accpt rate is 0.26
%  load thetas_sl_ep_point01.mat
    params_sl = mean(thetas);
    params_sl = [exp(params_sl(1:5)) params_sl(end)];
    simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);
    
%     subplot(311); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
%     set(gca, 'ylim', [0 max(simuldat_sl/1000) + 1])
    hold on; plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
    set(gca, 'ylim', [0 max(simuldat_sl/1000) + 1]); ylabel('synthetic likelihood abc');
    
    mse(ss_for_blowflydata(simuldat_sl))
    %% kabc (conditional mean embedding)
    
%     whichmethod = 'kabc_cond_embed';
%     iter = 1;
%     load(strcat('blowflydata_', num2str(whichmethod), '_thIter', num2str(iter), '_2.mat'))
%     params_kabc = results.post_mean;
    load theta_opt.mat;
    params_kabc = theta_opt;
    simuldat_kabc = gendata_pop_dyn_eqn(params_kabc, n);
    hold on; plot(1:180, flydata/1000, 'k', 1:180, simuldat_kabc./1000, 'k--'); 
    set(gca, 'ylim', [0 max(simuldat_kabc/1000) + 1]); ylabel('k abc');

    
    %% ours
    whichmethod = 'ssf_kernel_abc';
    minIdx1= 11;
    minIdx2 = 2;
    load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results');
    params_ours = results.post_mean(minIdx2,:);
%     minIdx1= 16;
%     minIdx2 = 2;
%     load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV_higherepsilon', '.mat'), 'results');
%     params_ours = results.post_mean; 
    simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);
    plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-');
    set(gca, 'ylim', [0 12]); ylabel('k abc');
    
    %% ours  with different MMD estimators
    
%     load  opt_k2abc_lin;
%     load opt_k2abc_rf; 
%     simuldat_ours_lin = gendata_pop_dyn_eqn(opt_k2abc_rf, n);

    %% indirect_score_abc
    
    simuldat_is = gendata_pop_dyn_eqn(indirect_score_abc, n);
    plot(1:180, flydata/1000, 'k', 1:180, simuldat_is./1000, 'm-');
    set(gca, 'ylim', [0 12]); ylabel('is abc');
    
    %% rejection_SA_ABC
   
    simuldat_reject_sa = gendata_pop_dyn_eqn(reject_sa_abc, n);
    simuldat_weighted_sa = gendata_pop_dyn_eqn(weighted_sa_abc, n);
    plot(1:180, flydata/1000, 'k', 1:180, simuldat_reject_sa./1000, 'm-');
    set(gca, 'ylim', [0 12]); ylabel('is abc');
    
    %% rejection_SA_ABCQ
    
    simuldat_reject_sa_q = gendata_pop_dyn_eqn(reject_sa_abc_q,n);
    simuldat_weighted_sa_q = gendata_pop_dyn_eqn(weighted_sa_abc_q,n);
    
    %% rejection_SA_ABC_woodss
    
    simuldat_reject_sa_woodss = gendata_pop_dyn_eqn(reject_sa_abc_woodss, n);
    simuldat_weighted_sa_woodss = gendata_pop_dyn_eqn(weighted_sa_abc_woodss, n);
    
    %% compute chosen summary statistics
    s_ours =  ss_for_blowflydata(simuldat_ours);
%     s_ours_lin =  ss_for_blowflydata(simuldat_ours_lin);
    s_kabc = ss_for_blowflydata(simuldat_kabc);
    s_sl = ss_for_blowflydata(simuldat_sl);
    s_is = ss_for_blowflydata(simuldat_is);
    s_reject_sa = ss_for_blowflydata(simuldat_reject_sa);
    s_reject_sa_q = ss_for_blowflydata(simuldat_reject_sa_q);
    s_reject_sa_woodss = ss_for_blowflydata(simuldat_reject_sa_woodss);
    s_weighted_sa = ss_for_blowflydata(simuldat_weighted_sa);
    s_weighted_sa_q = ss_for_blowflydata(simuldat_weighted_sa_q);
    s_weighted_sa_woodss = ss_for_blowflydata(simuldat_weighted_sa_woodss);

    msemat(i,:) = [mse(s_ours)  mse(s_sl) mse(s_reject_sa_woodss)  mse(s_is) mse(s_reject_sa)  mse(s_reject_sa_q) mse(s_weighted_sa_woodss) mse(s_weighted_sa)  mse(s_weighted_sa_q) mse(s_kabc) ];
    
end

mean(msemat)
std(msemat)

%%
% boxplot(msemat, {'k2', 'sl', 'sa-woods', 'aux', 'sa', 'saq', 'k'}); 

boxplot(msemat, {'k2', 'sl', 'sa-woods', 'aux', 'sa', 'saq', 'sa-woods-w', 'sa-w', 'saq-w','k'}); 
% set(gca, 'xticklabel',method_names);
% legend('ours', 'synthetic likelihood abc', 'kabc')

%% save some results for a separate box plot

mse_k2abc = msemat(:,1);
mse_sl = msemat(:,2);

save mse_k2abc mse_k2abc;
save mse_sl mse_sl; 

% change seed back
rng(oldRng);

%% how about showing trajectories of y given 

% load flydata.mat
% n = length(flydata);
% 
% subplot(311);
% plot(1:180, flydata/1000, 'k', 1:180, gendata_pop_dyn_eqn(sample_from_prior_blowflydata(1), n)./1000,  'm');
% set(gca, 'ylim', [0 12]);
% 
% subplot(312);
% plot(1:180, flydata/1000, 'k', 1:180, gendata_pop_dyn_eqn(sample_from_prior_blowflydata(1), n)./1000,  'b');
% set(gca, 'ylim', [0 12]);
% 
% subplot(313);
% plot(1:180, flydata/1000, 'k', 1:180, gendata_pop_dyn_eqn(sample_from_prior_blowflydata(1), n)./1000,  'r');
% set(gca, 'ylim', [0 12]);

