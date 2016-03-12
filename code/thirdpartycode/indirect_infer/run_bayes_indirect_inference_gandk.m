
% Bayesian indirect inference g-and-k example

%%
% pdBIL with different number of replicate datasets

load('gandk_data');
cov_rw = [0.0001,0.0001,-0.00015,-9e-05;0.0001,0.0004,0.0003,-0.00048;-0.00015,0.0003,0.0025,0;-9e-05,-0.00048,0,0.0009];
M=1000000;
K=1;
n=10000;
numComp = 3;
tic;
[theta_10000 loglike_ind_10000] = bayes_ind_inf_gandk(x,M,n,cov_rw,numComp,K);
finaltime=toc;
save('mcmc_gandk_bii.mat','theta_10000','loglike_ind_10000','finaltime');


load('gandk_data');
cov_rw = [0.0001,0.0001,-0.00015,-9e-05;0.0001,0.0004,0.0003,-0.00048;-0.00015,0.0003,0.0025,0;-9e-05,-0.00048,0,0.0009];
M=500000;
n=40000;
K=1;
numComp = 3;
tic;
[theta_40000 loglike_ind_40000] = bayes_ind_inf_gandk(x,M,n,cov_rw,numComp,K);
finaltime=toc;
save('mcmc_gandk_bii_rep4.mat','theta_40000','loglike_ind_40000','finaltime');


load('gandk_data');
cov_rw = [0.00015,0.0002,-0.0002,-0.0001;0.0002,0.0007,0.0003,-0.0003;-0.0002,0.0003,0.002,-2e-05;-0.0001,-0.0003,-2e-05,0.0002];
M=100000;
n=200000;
K=1;
numComp = 3;
tic;
[theta_200000 loglike_ind_200000] = bayes_ind_inf_gandk(x,M,n,cov_rw,numComp,K);
finaltime=toc;
save('mcmc_gandk_bii_rep20.mat','theta_200000','loglike_ind_200000','finaltime');


load('gandk_data');
cov_rw = [0.00015,0.0002,-0.0002,-0.0001;0.0002,0.0007,0.0003,-0.0003;-0.0002,0.0003,0.002,-2e-05;-0.0001,-0.0003,-2e-05,0.0002];
M=75000;
n=500000;
K=1;
numComp = 3;
tic;
[theta_500000 loglike_ind_500000] = bayes_ind_inf_gandk(x,M,n,cov_rw,numComp,K);
finaltime=toc;
save('mcmc_gandk_bii_rep50.mat','theta_500000','loglike_ind_500000','finaltime');


%% ABC-IL2
load('gandk_data');
cov_rw = [0.00015,0.0002,-0.0002,-0.0001;0.0002,0.0007,0.0003,-0.0003;-0.0002,0.0003,0.002,-2e-05;-0.0001,-0.0003,-2e-05,0.0002];
M=100000;
abc_tol = 2.3;
n=10000;
tic;
[theta_10000 theta_d_samp abc_discs] = bayes_ind_inf_il2_gandk(x,M,n,cov_rw,abc_tol,3);
finaltime=toc;
save('mcmc_gandk_abcil2.mat','theta_10000','theta_d_samp','abc_discs','finaltime');

%% ABC-IS
load('gandk_data');
cov_rw = [0.00015,0.0002,-0.0002,-0.0001;0.0002,0.0007,0.0003,-0.0003;-0.0002,0.0003,0.002,-2e-05;-0.0001,-0.0003,-2e-05,0.0002];
M=7000000;
abc_tol = 3;
n=10000;
tic;
[theta_10000 summary_stats_samp abc_discs] = bayes_ind_inf_is_gandk(x,M,n,cov_rw,abc_tol,3);
finaltime=toc;
save('mcmc_gandk_abcis.mat','theta_10000', 'summary_stats_samp','abc_discs','finaltime');



%% ABC-IP
load('gandk_data');
cov_rw = [0.00015,0.0002,-0.0002,-0.0001;0.0002,0.0007,0.0003,-0.0003;-0.0002,0.0003,0.002,-2e-05;-0.0001,-0.0003,-2e-05,0.0002];
M=1000000;
abc_tol = 7;
n=10000;
tic;
[theta_10000 theta_d_samp abc_discs] = bayes_ind_inf_ip_gandk(x,M,n,cov_rw,abc_tol,3,0);
finaltime=toc;
save('mcmc_gandk_abcip.mat','theta_10000','theta_d_samp','abc_discs','finaltime');



