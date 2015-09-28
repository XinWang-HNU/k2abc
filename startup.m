% Matlab startup file.
% Include necessary dependencies
display('Running startup.m');

base = pwd();
fs = filesep();
addpath(pwd);

% folders to be added by genpath
gfolders = {'mainfolder', 'runcode', 'thirdpartycode/xunit/', 'experiments', ...
    'saved', 'demo', 'data', 'blowfly'};
for i=1:length(gfolders)
    fol = gfolders{i};
    p = [base , fs, fol];
    fprintf('add gen path: %s\n', p);
    addpath(genpath(p));
end

% addpath(genpath(fullfile(base, 'real')));
% addpath(genpath(fullfile(base, 'other')));

clear base fs gfolders i fol p 

