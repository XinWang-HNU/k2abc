% to show mses on sufficient stats to compare mmds
% mijung wrote on oct 6, 2015

clear all;
clc;

oldRng = rng();
seed = 2;
rng(seed);


load flydata.mat
n = length(flydata);

num_rept_mse = 100;
how_many_methods = 2;
msemat = zeros(num_rept_mse, how_many_methods);
s = ss_for_blowflydata(flydata);
mse = @(a) norm(s-a);

for i=1:num_rept_mse
    
    %% k2 abc (quadratic mmd)
    
%     whichmethod = 'ssf_kernel_abc';
%     minIdx1= 11;
%     minIdx2 = 2;
%     load(strcat('blowflydata: ', num2str(whichmethod), 'fromXV', '.mat'), 'results');
%     params_ours = results.post_mean(minIdx2,:);
%     
%     simuldat_ours = gendata_pop_dyn_eqn(params_ours, n);
    
    %% ours  with different MMD estimators
    
    load opt_k2abc_rf;
    simuldat_ours_rf = gendata_pop_dyn_eqn(opt_k2abc_rf, n);
    
    load  opt_k2abc_lin;
    simuldat_ours_lin = gendata_pop_dyn_eqn(opt_k2abc_lin, n);
    
    %% synthetic likelihood abc
%     load thetas_sl_ep_point1.mat % accpt rate is 0.26
%     params_sl = mean(thetas);
%     params_sl = [exp(params_sl(1:5)) params_sl(end)];
%     simuldat_sl = gendata_pop_dyn_eqn(params_sl, n);
   
    %% compute chosen summary statistics
%     s_ours =  ss_for_blowflydata(simuldat_ours);
    s_ours_rf =  ss_for_blowflydata(simuldat_ours_rf);
    s_ours_lin =  ss_for_blowflydata(simuldat_ours_lin);
%     s_sl = ss_for_blowflydata(simuldat_sl);
    
    msemat(i,:) = [mse(s_ours_rf) mse(s_ours_lin)];
    
end

% mean(msemat)
% std(msemat)

%%

load mse_k2abc;
load mse_sl; 

msemat = [mse_k2abc msemat mse_sl];

%%

boxplot(msemat, {'k2', 'k2-rf', 'k2-lin','sl'});