% test softabc
%
% Softabc refers to the use of Gaussian kernel on sufficient statistics to create 
% weights for parameters. No rejection.
%

clear all;
clc;
close all;

load fig1data_200.mat;

% summary statistic from yobs
ss = [mean(yobs) var(yobs)];

%%

niter = 20;

for iter = 1:niter
    
    [iter niter]
    
    %% run our ABC code
    
    % sample theta M times
    M = 200;
    howmanytheta = length(theta);
%     theta_samps = zeros(M, howmanytheta);
    
    % these will vary later
    % kernelparams = meddistance(yobs)^2;
    % kernelparams = meddistance(yobs)^2*2;
    howmanyepsilon = 5;
    epsilon = logspace(-4, 2, howmanyepsilon);
    % epsilon = 1000;
    
    muhat_softabc = zeros(howmanyepsilon,howmanytheta);
    
    prior_var = 10*eye(howmanytheta);
    
    %%
    for count = 1:howmanyepsilon
        
        % we sample y L times, where each y consists of Ns samples
%         L = 100;
%         Ns = 200;
        L = 100;
        Ns = 100;
        k = zeros(M, L);
        
        %% (2) draw parameters from the prior (theta_j)
        % e.g., fix sigma, and draw mean from a Gaussian
        theta_samps = mvnrnd(zeros(1, howmanytheta), prior_var, M);
        %theta_samps = randn(M, 1)*sqrt(prior_var) + 2;
        
        for j=1:M
            
            
            % draw samples for y given theta
            unnorprob_samps = [f(theta_samps(j,:))'; 1];
            prob_samps = unnorprob_samps./sum(unnorprob_samps);
            
            
            %% (3) sample y from the parameters (y_i^j)
            
            parfor l = 1:L
                
%                 [count j l]
                
                %             % draw samples for y given theta
                %             unnorprob_samps = [f(theta_samps(j,:))'; 1];
                %             prob_samps = unnorprob_samps./sum(unnorprob_samps);
                
                % first draw discrete variables [1 4]
                discvar_samps = randsample(maxy, Ns, true, prob_samps);
                % hist(discvar)
                
                % then draw y from uniform distribution in each range
                y = zeros(1,Ns);
                
                for i=1:maxy
                    idx_samps = (discvar_samps==i);
                    y(idx_samps) = (i-1) + rand(sum(idx_samps), 1);
                end
                
                %y = gen_mvn(theta_samps(j, :), theta_var, Ns);
                %y = randn(1, Ns)*sqrt(theta_var) + theta_samps(j);
                
                %% (4) compute MMD for each y_i^j and y*_i^j
                
                ss_samps = [mean(y) var(y)];
                squareddiff = sum((ss-ss_samps).^2)/length(ss_samps);
%                 squareddiff = sum((ss-ss_samps).^2);
                
                k(j, l) = exp(-squareddiff/epsilon(count));
                
            end
            
        end
        
        %% (5) compute w_j which gives us posterior mean of theta
        
        wj_numerator = sum(k, 2)/L;
        wj_denominator = sum(sum(k))/L;
        
        %     muhat(count, :) = sum(wj_numerator.*theta_samps)/wj_denominator;
        muhat_softabc(count, :) = sum(bsxfun(@times, wj_numerator, theta_samps))./wj_denominator;
        
        %     [theta';  muhat(count, :)]
        
    end
    
    FN = strcat('results_fig1_softabc_200','_thIter',num2str(iter),'.mat');
    save(FN, 'muhat_softabc', 'yobs', 'theta', 'howmanytheta', 'howmanyepsilon', 'epsilon');
    
end


%%

%% check the results

matminmse_softABC = zeros(20,1);
msemat = zeros(20, howmanyepsilon);
meanofmean_softABC = zeros(howmanyepsilon, howmanytheta, 20);

%%
for iter = 1:niter
%     load(strcat('results_fig1_softabc','_thIter',num2str(iter),'.mat'));
    
    load(strcat('results_fig1_softabc_200','_thIter',num2str(iter),'.mat'));
    
    mse = @(a) sum(bsxfun(@minus, a, theta').^2, 2);
    figure(3);
    subplot(2,2,[1 2]); hold on; semilogx(epsilon, muhat_softabc, 'b.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); ylabel('muhat'); title('fixed length scale = median(obs)');
    subplot(2,2,[3 4]); loglog(epsilon, mse(muhat_softabc), 'b.--'); hold on;  xlabel('epsilon'); ylabel('mse'); hold on;
    
    matminmse_softABC(iter) = min(mse(muhat_softabc));
    msemat(iter,:) = mse(muhat_softabc);
    
    meanofmean_softABC(:,:,iter) = muhat_softabc; 
end

%%

figure(5);
subplot(2,2,[1 2]); semilogx(epsilon, mean(meanofmean_softABC,3), 'b.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.');set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); hold on; ylabel('mean of muhat (20 iterations)');  title('soft ABC (200 yobs)');
% subplot(2,2,[3 4]); loglog(epsilon, mean(msemat), 'b.-'); xlabel('epsilon'); ylabel('mean of mse (20 iterations)'); hold on;
subplot(2,2,[3 4]); loglog(epsilon, mse(mean(meanofmean_softABC,3)), 'b.-'); set(gca, 'xlim', [min(epsilon)/2 max(epsilon)*1.5]); xlabel('epsilon'); ylabel('mse of mean of muhat (20 iterations)'); hold on;

min(mse(mean(meanofmean_softABC,3)))
% min(mean(msemat))
