clear all;
clc;
clf;

load flydata.mat;

howmanyscalelength = 10; 
width2_list = meddistance(flydata)^2.*logspace(-2,2,howmanyscalelength); 
howmanyepsilon = 9; 
epsilon_list = logspace(-5, 0, howmanyepsilon);

eff_samp_size_mat = zeros(howmanyscalelength, howmanyepsilon);

whichmethod =  'ssf_kernel_abc';

effc_samp_size = @(a) 1/sum(a.^2);

for i=1:howmanyscalelength
    
    load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(i), '.mat'));    
    
    for j=1:howmanyepsilon
        [i j]
        
        norm_weights = results.R.norm_weights(:,j);
        norm_weights(norm_weights<1e-4) = 1e-4;
%         plot(norm_weights);
%         pause;
        
        
        eff_samp_size_mat(i,j) = effc_samp_size(norm_weights);
%         avg_loss_mat(i,j) = compute_loss_for_epsilon_kernelparam(opts); 
    end
    
end

%%
[minIdx1, minIdx2] = ind2sub([howmanyscalelength, howmanyepsilon], find(max(max(eff_samp_size_mat)) == eff_samp_size_mat,2));
subplot(211); plot(avg_loss_mat')

load(strcat('blowflydata: ', num2str(whichmethod), '_thIter', num2str(minIdx1), '.mat'));
params_ours = results.post_mean(minIdx2,:); 
simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

% load(strcat('blowflydata: ', num2str(whichmethod), '_medianHeuristic', '.mat'));
% params_ours = results.post_mean(2,:); 
% simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);

subplot(212); plot(1:180, flydata/1000, 'k', 1:180, simuldat_ours./1000, 'r-'); title('simulated data');
set(gca, 'ylim', [0 max(simuldat_ours/1000) + 1])

