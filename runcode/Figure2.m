% to make Figure 2
clear all;
clc;
% clf;

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

%% computing mse on ss

num_rept_mse = 100;
msemat = zeros(num_rept_mse, 3);
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

    
    % compute chosen summary statistics
    s_ours =  ss_for_blowflydata(simuldat_ours);
    s_kabc = ss_for_blowflydata(simuldat_kabc);
    s_sl = ss_for_blowflydata(simuldat_sl);
    
    mse(s_ours)
    msemat(i,:) = [mse(s_ours) mse(s_sl) mse(s_kabc)];
    
end

mean(msemat)
std(msemat)
%%
boxplot(msemat); 
% legend('ours', 'synthetic likelihood abc', 'kabc')

%% computing mse on ss with posterior samples of theta

% num_rept_mse = 5000;
% msemat = zeros(num_rept_mse, 3);
% s = ss_for_blowflydata(flydata);
% mse = @(a) norm(s-a);
% 
% % draw samples for theta from posterior 
% whichmethod = 'ssf_kernel_abc';
% minIdx1= 11;
% minIdx2 = 2;
% load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results');
% weightvec = results.R.norm_weights(:, minIdx2);
% theta_samps_prior  = results.R.latent_samples;
% idx_to_samp = discrete_rnd(weightvec', 1, 1e4);
% oursamps = theta_samps_prior(:, idx_to_samp);
%     
% for i=1:num_rept_mse
%     
%     
%     %% synthetic likelihood abc
%     load thetas_sl_ep_point1.mat % accpt rate is 0.26
% %  load thetas_sl_ep_point01.mat
% %     params_sl = mean(thetas);
%     params_sl = thetas(i,:);
%     params_sl = [exp(params_sl(1:5)) params_sl(end)];
%     simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);
%     
% %     subplot(311); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
% %     set(gca, 'ylim', [0 max(simuldat_sl/1000) + 1])
%     subplot(311); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
%     set(gca, 'ylim', [0 max(simuldat_sl/1000) + 1]); ylabel('synthetic likelihood abc');
%     
% %     mse(ss_for_blowflydata(simuldat_sl))
%     %% kabc (conditional mean embedding)
%     
%     load theta_opt.mat;
%     params_kabc = theta_opt;
%     simuldat_kabc = gendata_pop_dyn_eqn(params_kabc, n);
%     subplot(312); plot(1:180, flydata/1000, 'k', 1:180, simuldat_kabc./1000, 'k--'); 
%     set(gca, 'ylim', [0 max(simuldat_kabc/1000) + 1]); ylabel('k abc');
%     
%     %% ours
% 
%     params_ours = oursamps(:,i); 
%     simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);
%     subplot(313); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-');
%     set(gca, 'ylim', [0 12]); ylabel('k abc');
% 
%     % compute chosen summary statistics
%     s_ours =  ss_for_blowflydata(simuldat_ours);
%     s_kabc = ss_for_blowflydata(simuldat_kabc);
%     s_sl = ss_for_blowflydata(simuldat_sl);
%     
% %     mse(s_ours)
%     msemat(i,:) = [mse(s_ours) mse(s_sl) mse(s_kabc)];
%     
% end
% 
% mean(msemat)
% std(msemat)
% %%
% boxplot(msemat); 
% % legend('ours', 'synthetic likelihood abc', 'kabc')
