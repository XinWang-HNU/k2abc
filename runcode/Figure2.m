% to make Figure 2

load flydata.mat
n = length(flydata);

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

params_sl = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);

subplot(211); plot(1:180, flydata/1000, 'k', 1:180, simuldat_sl./1000, 'b-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])
subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat/1000) + 1])

% compute chosen summary statistics
s = ss_for_blowflydata(flydata);
% s_ours =  ss_for_blowflydata(simuldat_ours);
s_kabc = ss_for_blowflydata(simuldat);
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
% opts.obj = @(a) sqrt(sum((flydata'-a).^2)/n);
opts.obj = @(a) norm(s_true-ss_for_blowflydata(a));
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

subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

%%

opts_sl = opts; 
opts_sl.params = exp([ 3.76529501 -1.03266828  5.46587492 -0.40094812 -0.96334847  log(7) ]); 
% simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);
avg_loss_sl = compute_loss_for_epsilon_kernelparam(opts_sl); 
[min(min(avg_loss_mat)) avg_loss_sl]
