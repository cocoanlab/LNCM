%%
basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
imgdir = '/Volumes/cocoanlab03/open_dataset/GSP1000';

subjlist = importdata(fullfile(imgdir, 'GSP1000.txt'));
n_subj = numel(subjlist);

%%

brainmask_img = fmri_mask_image(which('brainmask_canlab_2mm.nii'));
merged_lesion_dat = fmri_data({fullfile(datdir, 'Lesion_Maps/Hallym_lesion_2mm_merged.nii'), ...
    fullfile(datdir, 'Lesion_Maps/Bundang_lesion_2mm_merged.nii')}, brainmask_img);
merged_lesion_idx = logical(merged_lesion_dat.dat);
n_node = size(brainmask_img.dat, 1); % 242953 voxels
n_lesion = size(merged_lesion_idx, 2); % 1432 lesions

%%

corr_z_sum = zeros(n_node, n_lesion);
corr_z_sqsum = zeros(n_node, n_lesion);

lesion_conn_z_t_file = fullfile(datdir, 'GSP_lesion_conn_group_z_t.mat');
lesion_conn_r_mean_file = fullfile(datdir, 'GSP_lesion_conn_group_r_mean.mat');

sj_start = 1;
rem_subjs = sj_start:n_subj;

%%

for sj_num = rem_subjs
    
    fprintf('Working on Subject %.3d  -  %s ... \n', sj_num, datetime);
    
    imglist = filenames(fullfile(imgdir, subjlist{sj_num}, 'func/sub-*_finalmask.nii'));
    concat_img_dat = [];
    
    for img_i = 1:numel(imglist)
        
        img_dat{img_i} = fmri_data(imglist{img_i}, brainmask_img);
        if any(all(img_dat{img_i}.dat' == 0)); system(sprintf('echo %s_run%d >> %s', subjlist{sj_num}, img_i, fullfile(basedir, 'Stroke_share/METAVCI_Network\ analysis/Analysis/scripts/EmptyVox_list_GSP.txt'))); end
        img_dat{img_i}.dat = zscore(double(img_dat{img_i}.dat'));
        concat_img_dat = [concat_img_dat; img_dat{img_i}.dat];
        
    end
    
    seed_ts = zeros(size(concat_img_dat,1), n_lesion);
    
    for les_i = 1:n_lesion
        seed_ts(:,les_i) = mean(concat_img_dat(:, merged_lesion_idx(:,les_i)), 2);
    end
    
    corr_z = atanh(corr(concat_img_dat, seed_ts));
    if any(isnan(corr_z), 1:2)
        disp('NaN corr found!');
        corr_z(isnan(corr_z)) = 0;
    end
    corr_z_sum = corr_z_sum + corr_z;
    corr_z_sqsum = corr_z_sqsum + corr_z.^2;
    
    fprintf('Done.\n');
    
end

%%

fprintf('Saving corr_z_t and corr_r_mean ...   \n');

corr_z_mean = corr_z_sum ./ n_subj;
corr_z_var = (corr_z_sqsum ./ n_subj) - corr_z_mean.^2;
corr_z_var = corr_z_var .* (n_subj / (n_subj - 1)); % unbiased est: default in var(x) Matlab function
corr_z_ste = (corr_z_var .^ 0.5) ./ (n_subj .^ 0.5);
corr_z_t = corr_z_mean ./ corr_z_ste;
corr_r_mean = tanh(corr_z_mean);

save(lesion_conn_z_t_file, 'corr_z_t', '-v7.3');
save(lesion_conn_r_mean_file, 'corr_r_mean', '-v7.3');

fprintf('Done.\n');
