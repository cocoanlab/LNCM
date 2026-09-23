function set_path_env

gitdir = '/Users/jaejoong/github';
addpath(genpath(fullfile(gitdir, 'canlab')));
rmpath(genpath(fullfile(gitdir, 'canlab/MediationToolbox/geom2d')));
rmpath(genpath(fullfile(gitdir, 'canlab/CanlabPrivate/preprocess')));
addpath(genpath(fullfile(gitdir, 'cocoanlab')));

rscdir = '/Volumes/cocoanlab01/resources';
addpath(genpath(fullfile(rscdir, 'spm12')));
rmpath(genpath(fullfile(rscdir, 'spm12/external')));

end