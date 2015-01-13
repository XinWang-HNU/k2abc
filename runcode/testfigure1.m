% test some scenarious for figure 1

% the aim of figure 1 is to show efficiency of our method
% that doesn't need to predefine sufficient number
% of summary statistics

% in this example, we will draw samples from some exponential family distribution
% with high order sufficient statistics.
% our hope is to let other methods use non-sufficient summary statistics,
% maybe first and second moments only, and fail to capture p(y|theta)

% startup

% clear all;
% clc;
% close all;

%% (1) generate data

% dimy = 1; % 1d y
% 
% maxy = 4;
% % theta = randn(maxy-1,1);
% theta = [1.2 -3 -2]';
% 
% % y \sim p(y|theta)
% %        = exp(theta(1)) if 0<=y<=1,
% %        = exp(theta(2)) if 1<y<=2,
% %        = exp(theta(3)) if 2<y<=3,
% %        = 1 if 3<y<=4,
% 
% nsamps = 1000;
% 
% f = @(a) log(exp(a)+1);
% 
% unnorprob = [f(theta); 1];
% prob = unnorprob./sum(f(theta)+1);
% 
% % first draw discrete variables [1 4]
% discvar = randsample(maxy, nsamps, true, prob);
% % hist(discvar)
% 
% % then draw y from uniform distribution in each range
% yobs = zeros(1, nsamps);
% 
% for i=1:maxy
%     idx = (discvar==i);
%     yobs(idx) = (i-1) + rand(sum(idx), 1);
% end
% 
% subplot(221); hist(discvar)
% subplot(222); hist(yobs)
% 
% 
% save(strcat('fig1data.mat'), 'yobs', 'theta', 'maxy', 'discvar', 'prob', 'f'); 

%%


clear all;
clc;
close all;

load fig1data.mat; 

niter = 5;

for iter=1:niter
    
    [iter niter]
    
    kernelprs = meddistance(yobs)^2.*[0.1 0.5 1 2 4 8];
    
    for kkk=1:length(kernelprs)
        
        kernelparams = kernelprs(kkk);
        
        %% run our ABC code
        
        % sample theta M times
        M = 500;
        howmanytheta = length(theta);
        theta_samps = zeros(M, howmanytheta);
        
        % these will vary later
        % kernelparams = meddistance(yobs)^2;
        % kernelparams = meddistance(yobs)^2*2;
        howmanyepsilon = 5;
        epsilon = logspace(-4, 2, howmanyepsilon);
        % epsilon = 1000;
        
        muhat = zeros(howmanyepsilon,howmanytheta);
        
        ker = KGaussian(kernelparams);
        
        prior_var = 10*eye(howmanytheta);
        
        %%
        for count = 1:howmanyepsilon
            
            % we sample y L times, where each y consists of Ns samples
            L = 100;
            Ns = 200;
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
                
                for l = 1:L
                    
                    [iter count j l]
                    
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
                    
                    k(j, l) = exp(-mmd(y, yobs, ker)^2/epsilon(count));
                    
                end
                
            end
            
            %% (5) compute w_j which gives us posterior mean of theta
            
            wj_numerator = sum(k, 2)/L;
            wj_denominator = sum(sum(k))/L;
            
            %     muhat(count, :) = sum(wj_numerator.*theta_samps)/wj_denominator;
            muhat(count, :) = sum(bsxfun(@times, wj_numerator, theta_samps))./wj_denominator;
            
            %     [theta';  muhat(count, :)]
            
        end
        
        FN = strcat('results_fig1_ourmethod','_kernelparam',num2str(kkk),'_thIter',num2str(iter),'.mat');
        save(FN, 'muhat', 'yobs', 'theta', 'howmanytheta', 'howmanyepsilon', 'epsilon');
        
    end
end


%% (6) compute f(sigma, epsilon) = squared distance between theta_mean and theta_true

% load results_fig1_ourmethod_kernelparam7.mat;
%
% kernelprs = meddistance(yobs)^2.*[0.1 0.5 1 2 4 8 16];
%
% mse = @(a) sum(bsxfun(@minus, a, theta').^2, 2);
% % /howmanytheta;
%
% figure(2);
% % subplot(2,2,[3 4]); semilogx(epsilon, muhat, 'r.-', epsilon, repmat(theta', howmanyepsilon,1), 'k.'); ylabel('muhat'); title('fixed length scale = median(obs)');
% subplot(2,2,[3 4]); loglog(epsilon, mse(muhat), 'k.--'); xlabel('epsilon'); ylabel('mse'); hold on;
%
% % as epsilon gets larger, our estimate gets closer to prior mean.
% % as epsilon gets smaller, our estimate gets closer to observations.
%
%
%
% min(mse(muhat))