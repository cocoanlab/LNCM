%%
basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
imgdir = '/Volumes/cocoanlab03/open_dataset/GSP1000';

subjlist = importdata(fullfile(imgdir, 'GSP1000.txt'));
n_subj = numel(subjlist);

%%

brainmask_img = fmri_mask_image(fullfile(datdir, 'brainmask_canlab_3mm.nii'));
n_node = size(brainmask_img.dat, 1); % 70098 voxels

%%

corr_z_sum = zeros(n_node, n_node, 'single');
corr_z_sqsum = zeros(n_node, n_node, 'single');

voxel_conn_z_t_file = fullfile(datdir, 'GSP_voxel_conn_group_z_t.mat');
voxel_conn_r_mean_file = fullfile(datdir, 'GSP_voxel_conn_group_r_mean.mat');

sj_start = 1;
rem_subjs = sj_start:n_subj;

%%

for sj_num = rem_subjs
    
    fprintf('Working on Subject %.3d  -  %s ... \n', sj_num, datetime);
    
    imglist = filenames(fullfile(imgdir, subjlist{sj_num}, 'func/sub-*_finalmask_3mm.nii'));
    concat_img_dat = [];
    
    for img_i = 1:numel(imglist)
        
        img_dat{img_i} = fmri_data(imglist{img_i}, brainmask_img);
        if any(all(img_dat{img_i}.dat' == 0)); system(sprintf('echo %s_run%d >> %s', subjlist{sj_num}, img_i, fullfile(basedir, 'Stroke_share/METAVCI_Network\ analysis/Analysis/scripts/EmptyVox_voxconn_list_GSP.txt'))); end
        img_dat{img_i}.dat = zscore(img_dat{img_i}.dat');
        concat_img_dat = [concat_img_dat; img_dat{img_i}.dat];
        
    end
    
    corr_z = atanh(corr(concat_img_dat));
    if any(isnan(corr_z), 1:2)
        disp('NaN corr found!');
        corr_z(isnan(corr_z)) = 0;
    end
    corr_z_sum = corr_z_sum + corr_z;
    corr_z_sqsum = corr_z_sqsum + corr_z.^2;
    
    fprintf('Done.\n');
    
end

clear corr_z;

%%

fprintf('Saving corr_z_t and corr_r_mean ...   \n');

corr_z_mean = corr_z_sum ./ n_subj;
clear corr_z_sum;
corr_z_var = (corr_z_sqsum ./ n_subj) - corr_z_mean.^2;
clear corr_z_sqsum;
corr_z_var = corr_z_var .* (n_subj / (n_subj - 1)); % unbiased est: default in var(x) Matlab function
corr_z_ste = (corr_z_var .^ 0.5) ./ (n_subj .^ 0.5);
clear corr_z_var;
corr_z_t = corr_z_mean ./ corr_z_ste;
corr_r_mean = tanh(corr_z_mean);

fid = fopen(voxel_conn_z_t_file, 'w');
fwrite(fid, corr_z_t, 'single');
fclose(fid);
fid = fopen(voxel_conn_r_mean_file, 'w');
fwrite(fid, corr_r_mean, 'single');
fclose(fid);

fprintf('Done.\n');
