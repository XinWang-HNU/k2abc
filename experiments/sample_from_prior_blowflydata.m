function samps_in_log = sample_from_prior_blowflydata(n)

% mijung wrote on jan 24, 2015

% draw thetas from prior that Ted used in GPSABC paper

% input
% n: number of samps to draw

%%%%%%%%%%%%%%%%%%%%%%%%%
% remember: params are in this order
%%%%%%%%%%%%%%%%%%%%%%%%%
% P = params(1);
% delta = params(2);
% N0 = params(3);
% sig_d = params(4);
% sig_p = params(5);
% tau = parmas(6); 

log_P = 2 + 2.*randn(n,1);
log_delta = -1.8 + 0.4*randn(n,1); 
log_N0 = 6 + 0.5*randn(n,1);
log_sigd = -0.75 + randn(n,1);
log_sigp = -0.5 + randn(n, 1);
log_tau = 2.7 + 0.1*randn(n,1);

samps_in_log = [log_P log_delta log_N0 log_sigd log_sigp log_tau]';
