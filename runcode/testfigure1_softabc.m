% test softabc
%
% Softabc refers to the use of Gaussian kernel on sufficient statistics to create 
% weights for parameters. No rejection.
%

clear all;
clc;
close all;

% load fig1data_400.mat;
load fig1data_400.mat;

% summary statistic from yobs
ss = [mean(yobs) var(yobs)];

%%

niter = 20;

for iter = 1:niter
    
    [iter niter]
    
    %% run our ABC code
    
    % sample theta M times
    M = 100;
    howmanytheta = length(theta);
    
%     howmanyepsilon = 4;
%     epsilon = sum(abs(yobs))/length(yobs).* logspace(-3, 0, howmanyepsilon);
    howmanyepsilon = 4;
    epsilon = sum(abs(yobs))/length(yobs).* logspace(-2, 0, howmanyepsilon);
    
    muhat_softabc = zeros(howmanyepsilon,howmanytheta);
    
    prior_var = 4*eye(howmanytheta);
    
    %%
    for count = 1:howmanyepsilon
        
        % we sample y , Ns times
        Ns = 100;
        k = zeros(M,1);
        
        %% (2) draw parameters from the prior (theta_j)
        % e.g., fix sigma, and draw mean from a Gaussian
        theta_samps = mvnrnd(zeros(1, howmanytheta), prior_var, M);
        
        for j=1:M
            
            
            % draw samples for y given theta
            unnorprob_samps = [f(theta_samps(j,:))'];
            prob_samps = unnorprob_samps./sum(unnorprob_samps);
            
            
            %% (3) sample y from the parameters (y_i^j)
            
%             parfor l = 1:L
                
                % first draw discrete variables [1 maxy]
                discvar_samps = randsample(maxy, Ns, true, prob_samps);
                % hist(discvar)
                
                % then draw y from uniform distribution in each range
                y = zeros(1,Ns);
                
                for i=1:maxy
                    idx_samps = (discvar_samps==i);
                    y(idx_samps) = (i-1) + rand(sum(idx_samps), 1);
                end
                                
                %% (4) compute MMD for each y_i^j and y*_i^j
                
                ss_samps = [mean(y) var(y)];
                squareddiff = sum((ss-ss_samps).^2)/length(ss_samps);
                
                k(j) = exp(-squareddiff/epsilon(count));
                
%             end
            
        end
        
        %% (5) compute w_j which gives us posterior mean of theta
        
%         wj_numerator = sum(k, 2)/L;
%         wj_denominator = sum(sum(k))/L;
        wj_numerator = k;
        wj_denominator = sum(k);
        
        %     muhat(count, :) = sum(wj_numerator.*theta_samps)/wj_denominator;
        muhat_softabc(count, :) = sum(bsxfun(@times, wj_numerator, theta_samps))./wj_denominator;
        
        %     [theta';  muhat(count, :)]
        
    end
    
    FN = strcat('results_fig1_softabc_400','_thIter',num2str(iter),'.mat');
    save(FN, 'muhat_softabc', 'yobs', 'theta', 'howmanytheta', 'howmanyepsilon', 'epsilon');
    
end


%%

%% check the results
load fig1data_400.mat;

howmanytheta = length(theta);

% howmanyepsilon = 4;
% epsilon = sum(abs(yobs))/length(yobs).* logspace(-2, 0, howmanyepsilon);

niter = 20; 

matminmse_softABC = zeros(niter,1);
msemat = zeros(niter, howmanyepsilon);
meanofmean_softABC = zeros(howmanyepsilon, howmanytheta, niter);

%%
for iter = 1:niter
%     load(strcat('results_fig1_softabc','_thIter',num2str(iter),'.mat'));
    
    load(strcat('results_fig1_softabc_400','_thIter',num2str(iter),'.mat'));
    
    mse = @(a) sum(bsxfun(@minus, a, theta').^2, 2);
    figure(4);
    subplot(2,2,[1 2]); semilogx(epsilon, muhat_softabc, 'b.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); hold on;  ylabel('muhat'); title('fixed length scale = median(obs)');
    subplot(2,2,[3 4]); loglog(epsilon, mse(muhat_softabc), 'b.--'); hold on;  xlabel('epsilon'); ylabel('mse'); hold on;
    
    matminmse_softABC(iter) = min(mse(muhat_softabc));
    msemat(iter,:) = mse(muhat_softabc);
    
    meanofmean_softABC(:,:,iter) = muhat_softabc; 
end

%%

% figure(5);
% subplot(2,2,[1 2]); semilogx(epsilon, mean(meanofmean_softABC,3), 'b.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); hold on; set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); hold on; ylabel('mean of muhat (20 iterations)');  title('soft ABC (200 yobs)');
subplot(3,1,[2,3]); semilogx(epsilon, mean(msemat), 'b.-'); xlabel('epsilon'); ylabel('mean of mse (20 iterations)'); hold on;
% subplot(2,2,[3 4]); loglog(epsilon, mse(mean(meanofmean_softABC,3)), 'b.-'); set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); xlabel('epsilon'); ylabel('mse of mean of muhat (20 iterations)'); hold on;

% min(mse(mean(meanofmean_softABC,3)))
min(mean(msemat))

%%

bestthetasamps_softABC = reshape(squeeze(meanofmean_softABC(1,:,:)), length(theta), []);

figure(100);
hold on;
subplot(311); hist(bestthetasamps_softABC(1,:)); set(gca, 'xlim', [-4 4]); hold on; plot(theta(1), 0:0.01:8, 'r-', mean(bestthetasamps_softABC(1,:)),0:0.01:8, 'b-' );
subplot(312); hist(bestthetasamps_softABC(2,:)); set(gca, 'xlim', [-4 4]); hold on; plot(theta(2), 0:0.01:8, 'r-', mean(bestthetasamps_softABC(2,:)),0:0.01:8, 'b-' );
subplot(313); hist(bestthetasamps_softABC(3,:)); set(gca, 'xlim', [-4 4]); hold on; plot(theta(3), 0:0.01:8, 'r-', mean(bestthetasamps_softABC(3,:)),0:0.01:8, 'b-' );

[mean(bestthetasamps_softABC(1,:)) mean(bestthetasamps_softABC(2,:)) mean(bestthetasamps_softABC(3,:))]
