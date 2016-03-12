function fu = funcs_global()
    %FUNCS_GLOBAL Return a struct containing functions related to the global environment 
    %of the project.
    fu = struct();
    fu.getRootFolder = @getRootFolder;
    fu.getRuncodeFolder = @getRuncodeFolder;
    fu.expSavedFolder = @expSavedFolder;
    fu.expSavedFile = @expSavedFile;
    fu.runcodeSavedFile = @runcodeSavedFile;
    fu.runcodeSavedFolder = @runcodeSavedFolder;
    fu.dataFolder = @dataFolder;
    fu.inDataFolder = @inDataFolder;
end

function r=getRootFolder()
    [p,f,e]=fileparts(which('startup.m'));
    r=p;
end

function f=getRuncodeFolder()
    % return the folder containing directly (mostly) runnable scripts
    p=getRootFolder();
    f=fullfile(p, 'runcode');
    if ~exist(f, 'dir')
        mkdir(f);
    end
end

function f=getSavedFolder()
    % return the top folder for saving .mat files 

    p=getRootFolder();
    f=fullfile(p, 'saved');
    if ~exist(f, 'dir')
        mkdir(f);
    end
end


function fpath=expSavedFile(expNum, fname)
    % construct a full path the the file name fname in the experiment 
    % folder identified by expNum
    expNumFolder=expSavedFolder(expNum);
    fpath=fullfile(expNumFolder, fname);
end

function expNumFolder=expSavedFolder(expNum)
    % return full path to folder used for saving results of experiment 
    % identified by expNum. Create the folder if not existed. 
    assert(isscalar(expNum));
    assert(mod(expNum, 1)==0);
    root=getSavedFolder();

    fname=sprintf('ex%d', expNum);
    expNumFolder=fullfile(root, fname);
    if ~exist(expNumFolder, 'dir')
        mkdir(expNumFolder);
    end
end

function fpath=runcodeSavedFile(fname)
   savedFolder=runcodeSavedFolder();
   fpath=fullfile(savedFolder, fname);
end

function runcode=runcodeSavedFolder()
   % return full path to folder used for saving results of temporary scripts 
   saved=getSavedFolder();
   runcode=fullfile(saved, 'runcode');
   if ~exist(runcode, 'dir')
       mkdir(runcode);
   end

end

function data_folder = dataFolder()
    r = getRootFolder();
    data_folder = fullfile(r, 'data');
end

function fpath = inDataFolder(fname)
    % return a full path to a file in the data folder. The file needs not exist.
    fpath = fullfile(dataFolder(), fname);
end
