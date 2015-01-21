function [ samples, probs ] = like_sigmoid_pw_const( params, n )
%LIKE_SIGMOID_PW_CONST Like like_piecewise_const but parametrise the probability 
%vector with a real-valued vectors. The real-valued vector is feed element-wise 
%to a sigmoid function. The normalized sigmoid values are used as probabilities.
%

assert(isnumeric(params));
sig = 1./(1+exp(-params));
norm_sig = sig/sum(sig);
probs = norm_sig;
samples = like_piecewise_const(norm_sig, n);

end

