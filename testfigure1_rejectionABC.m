% test rejection ABC

% startup

clear all;
clc;
close all;

%% load data generated from bimodal p(y|theta)

load fig1data.mat; 

% summary statistic from yobs
ss = [mean(yobs) var(yobs)];

% run rejection-ABC for 100 times

niter = 20;

for iter=1:niter
   
    [iter niter]
    
    %% run rejection-ABC code
    
    % which we use for sampling y in ABC
    f = @(a) log(exp(a)+1);
    maxy = 4;
    
    % sample theta M times
    M = 500;
    howmanytheta = length(theta);
    theta_samps = zeros(M, howmanytheta);
    
    howmanyepsilon = 5;
    epsilon = logspace(-2, 4, howmanyepsilon);
    % epsilon = 1000;
    
    prior_var = 10*eye(howmanytheta);
    % theta_selected = zeros(M, howmanytheta);
    
    muhat_rejectABC = zeros(howmanyepsilon, howmanytheta);
    accptrate = zeros(howmanyepsilon, 1);
    
    %%
    for count = 1:howmanyepsilon
        
        % we sample y L times, where each y consists of Ns samples
        %     L = 100;
        Ns = 1000;
        %     k = zeros(M, L);
        
        %% (2) draw parameters from the prior (theta_j)
        % e.g., fix sigma, and draw mean from a Gaussian
        theta_samps = mvnrnd(zeros(1, howmanytheta), prior_var, M);
        
        theta_selected = [];
        
        for j=1:M
            
            
            % draw samples for y given theta
            unnorprob_samps = [f(theta_samps(j,:))'; 1];
            prob_samps = unnorprob_samps./sum(unnorprob_samps);
            
            
            %% (3) sample y from the parameters (y_i^j)
            
            %         parfor l = 1:L
            
            %             [count j l]
            
            % first draw discrete variables [1 4]
            discvar_samps = randsample(maxy, Ns, true, prob_samps);
            
            % then draw y from uniform distribution in each range
            y = zeros(1,Ns);
            
            for i=1:maxy
                idx_samps = (discvar_samps==i);
                y(idx_samps) = (i-1) + rand(sum(idx_samps), 1);
            end
            
            %y = gen_mvn(theta_samps(j, :), theta_var, Ns);
            %y = randn(1, Ns)*sqrt(theta_var) + theta_samps(j);
            
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
    
    FN = strcat('results_fig1_rejectABC','_thIter',num2str(iter),'.mat');
    save(FN, 'muhat_rejectABC', 'minmse_rejectABC', 'accptrate', 'yobs', 'theta', 'howmanytheta', 'howmanyepsilon', 'epsilon');

   
end

%% check the results

matminmse_rejectABC = zeros(20,1);
msemat = zeros(20, howmanyepsilon);
accptratemat = zeros(20, howmanyepsilon); 
meanofmean_rejectABC = zeros(howmanyepsilon, howmanytheta, 20);

%%
for iter = 1:20
    load(strcat('results_fig1_rejectABC','_thIter',num2str(iter),'.mat'));
    
    mse = @(a) sum(bsxfun(@minus, a, theta').^2, 2);
    figure(2);
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

figure(2);
subplot(2,2,[1 2]); semilogx(epsilon, mean(meanofmean_rejectABC,3), 'r.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); ylabel('mean of muhat (20 iterations)');  title('rejection ABC');
subplot(2,2,[3 4]); loglog(epsilon, mean(msemat), 'k.-'); xlabel('epsilon'); ylabel('mean of mse (20 iterations)'); hold on;

