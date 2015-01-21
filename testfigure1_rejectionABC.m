% test rejection ABC

% startup

clear all;
clc;
close all;

%% load data generated from bimodal p(y|theta)

load fig1data_400.mat; 

% summary statistic from yobs
ss = [mean(yobs) var(yobs)];

% run rejection-ABC for 100 times

niter = 20;

%%

for iter=1:niter
   
    [iter niter]
    
    %% run rejection-ABC code
    
    % sample theta M times
    M = 100;
    howmanytheta = length(theta);
%     howmanyepsilon = 4;
%     epsilon = sum(abs(yobs))/length(yobs).* logspace(-3, 0, howmanyepsilon);
    howmanyepsilon = 4;
    epsilon = sum(abs(yobs))/length(yobs).* logspace(-2, 0, howmanyepsilon);
    
    prior_var = 4*eye(howmanytheta);
   
    muhat_rejectABC = zeros(howmanyepsilon, howmanytheta);
    accptrate = zeros(howmanyepsilon, 1);
    
    %%
    for count = 1:howmanyepsilon
        
        % we sample y L times, where each y consists of Ns samples
        %     L = 100;
        Ns = 100;
        %     k = zeros(M, L);
        
        %% (2) draw parameters from the prior (theta_j)
        % e.g., fix sigma, and draw mean from a Gaussian
        theta_samps = mvnrnd(zeros(1, howmanytheta), prior_var, M);
        
        theta_selected = [];
        
        for j=1:M
            
            
            % draw samples for y given theta
            unnorprob_samps = [f(theta_samps(j,:))'];
            prob_samps = unnorprob_samps./sum(unnorprob_samps);
            
            
            %% (3) sample y from the parameters (y_i^j)
                        
            % first draw discrete variables [1 4]
            discvar_samps = randsample(maxy, Ns, true, prob_samps);
            
            % then draw y from uniform distribution in each range
            y = zeros(1,Ns);
            
            for i=1:maxy
                idx_samps = (discvar_samps==i);
                y(idx_samps) = (i-1) + rand(sum(idx_samps), 1);
            end
                        
            %% (4) compare summary statistic whether accept or reject theta
            
            ss_samps = [mean(y) var(y)];
            squareddiff = sum((ss-ss_samps).^2)/length(ss_samps);
            
            if squareddiff < epsilon(count)
                
                theta_selected = [theta_selected; theta_samps(j,:)];
                %             else
                %
                %                 theta_selected(j,:) = [];
            end
            
            %             k(j, l) = exp(-mmd(ss, ss_samps, ker)^2/epsilon(count));
            
            %         end
            
        end
        
        %% compute the posterior mean in each case
        
        accptrate(count, :) = size(theta_selected,1)/M;
        muhat_rejectABC(count, :) = mean(theta_selected);
        
    end
    
    
    mse = @(a) sum(bsxfun(@minus, a, theta').^2, 2);
    % /howmanytheta;
    
    figure(2);
    subplot(2,2,[1 2]); semilogx(epsilon, muhat_rejectABC, 'r.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); ylabel('muhat'); title('fixed length scale = median(obs)');
    subplot(2,2,[3 4]); loglog(epsilon, mse(muhat_rejectABC), 'k.--'); xlabel('epsilon'); ylabel('mse'); hold on;
    
    minmse_rejectABC = min(mse(muhat_rejectABC)); 

    % save results
    
    FN = strcat('results_fig1_rejectABC_400','_thIter',num2str(iter),'.mat');
    save(FN, 'muhat_rejectABC', 'minmse_rejectABC', 'accptrate', 'yobs', 'theta', 'howmanytheta', 'howmanyepsilon', 'epsilon');

   
end

%% check the results

load fig1data_400.mat; 

howmanytheta = length(theta);
howmanyepsilon = 4;
epsilon = sum(abs(yobs))/length(yobs).* logspace(-3, 0, howmanyepsilon);

matminmse_rejectABC = zeros(20,1);
msemat = zeros(20, howmanyepsilon);
accptratemat = zeros(20, howmanyepsilon); 
meanofmean_rejectABC = zeros(howmanyepsilon, howmanytheta, 20);

%%
for iter = 1:20
%     load(strcat('results_fig1_rejectABC','_thIter',num2str(iter),'.mat'));
    load(strcat('results_fig1_rejectABC_400','_thIter',num2str(iter),'.mat'));
    
    mse = @(a) sum(bsxfun(@minus, a, theta').^2, 2);
    figure(3);
    subplot(2,2,[1 2]); semilogx(epsilon, muhat_rejectABC, 'r.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); ylabel('muhat'); title('fixed length scale = median(obs)');
    subplot(2,2,[3 4]); loglog(epsilon, mse(muhat_rejectABC), 'k.--'); xlabel('epsilon'); ylabel('mse'); hold on;
    
    matminmse_rejectABC(iter) = min(mse(muhat_rejectABC));
    msemat(iter,:) = mse(muhat_rejectABC);
    accptratemat(iter,:) = accptrate;
    
    meanofmean_rejectABC(:,:,iter) = muhat_rejectABC; 
end

%%
% mean(accptratemat) = 0.0084    0.2038    1.0000    1.0000    1.0000
% mean(msemat) = 1.6667    1.6489   14.4913   14.7748   14.3310

% figure(5);
% subplot(2,2,[1 2]); semilogx(epsilon, mean(meanofmean_rejectABC,3), 'k.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); ylabel('mean of muhat');  title('rejection ABC (200 yobs)');
% subplot(2,2,[3 4]); loglog(epsilon, mse(mean(meanofmean_rejectABC,3)), 'k.-'); xlabel('epsilon'); set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); ylabel('mse of mean of muhat (20 iterations)'); hold on;
subplot(3,1,[2 3]); semilogx(epsilon, mean(msemat), 'k.-'); xlabel('epsilon'); 
% set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); ylabel('mse of mean of muhat (20 iterations)'); hold on;

%%
min(mean(msemat))
% min(mse(mean(meanofmean_rejectABC,3)))

bestthetasamps_rejectABC = reshape(squeeze(meanofmean_rejectABC(2,:,:)), length(theta), []);

[mean(bestthetasamps_rejectABC(1,:)) mean(bestthetasamps_rejectABC(2,:)) mean(bestthetasamps_rejectABC(3,:))]

%%
figure(106);
hold on;
subplot(311); hist(bestthetasamps_rejectABC(1,:)); set(gca, 'xlim', [-4 4]); hold on; plot(theta(1), 0:0.01:8, 'r-', mean(bestthetasamps_rejectABC(1,:)),0:0.01:8, 'b-' );
subplot(312); hist(bestthetasamps_rejectABC(2,:)); set(gca, 'xlim', [-4 4]); hold on; plot(theta(2), 0:0.01:8, 'r-', mean(bestthetasamps_rejectABC(2,:)),0:0.01:8, 'b-' );
subplot(313); hist(bestthetasamps_rejectABC(3,:)); set(gca, 'xlim', [-4 4]); hold on; plot(theta(3), 0:0.01:8, 'r-', mean(bestthetasamps_rejectABC(3,:)),0:0.01:8, 'b-' );

[mean(bestthetasamps_rejectABC(1,:)) mean(bestthetasamps_rejectABC(2,:)) mean(bestthetasamps_rejectABC(3,:))]