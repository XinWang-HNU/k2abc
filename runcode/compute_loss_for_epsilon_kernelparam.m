% this function computes a loss function for each set of epsilon and kernel parameter.
% mijung wrote on jan 27, 2015

function [avg_loss, std_loss] = compute_loss_for_epsilon_kernelparam(opts)

% input: 
% (1) opts.likelihood: to draw pseudo samples
% (2) opts.obj: which objective function to use
% (3) opts.num_samps: how many pseudo-data points to draw
% (4) opts.num_rep: compute avg loss across num_rep trials
% (5) opts.params: parameters to use to draw pseudo data

% output: mean(loss) and std(loss)

likelihood_func = opts.likelihood_func; 
loss = opts.obj;
num_rep = opts.num_rep;
num_samps =  opts.num_samps; 
params = opts.params;

lossmat = zeros(num_rep, 1);

for i=1:num_rep
    % (1) draw samples for pseudo data
    pseudo_i = likelihood_func(params, num_samps);
    lossmat(i) = loss(pseudo_i);
end

avg_loss = mean(lossmat);
std_loss = std(lossmat); 