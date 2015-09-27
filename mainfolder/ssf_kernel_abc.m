function [ results, op ] = ssf_kernel_abc( Obs, op )
%SSF_KERNEL_ABC Perform inference with K2ABC. See k2abc.
%   - Use k2abc instead. 
%   - This function is kept for legacy code.
% 
%@deprecated

[results, op] = k2abc(Obs, op);

end

