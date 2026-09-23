%%
basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
imgdir = '/Volumes/homeo/HCP_Aging/HCPAgingVer2Re/fmriresults01';

T = readtable(fullfile(datdir, 'HCA_LS_2.0_subject_completeness.csv'));
wh_unrel = strcmp(T.unrelated_subset, 'TRUE');
wh_full_scan = T.RS_fMRI_Count == 4 & T.RS_fMRI_PctCompl > 99;
subjlist = strcat(T.src_subject_id(wh_unrel & wh_full_scan), '_V1_MR');
n_subj = numel(subjlist);

%%

brainmask_img = fmri_mask_image(which('brainmask_canlab_2mm.nii'));
merged_lesion_dat = fmri_data({fullfile(datdir, 'Lesion_Maps/Hallym_lesion_2mm_merged.nii'), ...
    fullfile(datdir, 'Lesion_Maps/Bundang_lesion_2mm_merged.nii')}, brainmask_img);
merged_lesion_idx = logical(merged_lesion_dat.dat);
n_node = size(brainmask_img.dat, 1); % 242953 voxels
n_lesion = size(merged_lesion_idx, 2); % 1432 lesions
tr = 0.8;
sm_fwhm = 6;
sm_sig = sm_fwhm./(2*sqrt(2*log(2)));
freq_band = [0.009 0.08];
% freq_sig = 1./(2*sqrt(2*log(2)) * tr * freq_band);

%%

corr_z_sum = zeros(n_node, n_lesion);
corr_z_sqsum = zeros(n_node, n_lesion);

start_idx_file = fullfile(datdir, 'HCPA_lesion_conn_group_z_start_idx.mat');
lesion_conn_z_sum_file = fullfile(datdir, 'HCPA_lesion_conn_group_z_sum.mat');
lesion_conn_z_sqsum_file = fullfile(datdir, 'HCPA_lesion_conn_group_z_sqsum.mat');
lesion_conn_z_t_file = fullfile(datdir, 'HCPA_lesion_conn_group_z_t.mat');
lesion_conn_r_mean_file = fullfile(datdir, 'HCPA_lesion_conn_group_r_mean.mat');

if exist(start_idx_file) && exist(lesion_conn_z_sum_file) && exist(lesion_conn_z_sqsum_file)
    load(start_idx_file, 'sj_start');
    fprintf('Loading intermediate save file (Subject 1 - %d) ...   \n', sj_start-1);
    load(lesion_conn_z_sum_file, 'corr_z_sum');
    load(lesion_conn_z_sqsum_file, 'corr_z_sqsum');
else
    sj_start = 1;
end

rem_subjs = sj_start:n_subj;

%%

for sj_num = rem_subjs
    
    fprintf('Working on Subject %.3d  -  %s ... \n', sj_num, datetime);
    
    temp_working_dir = tempname;
    temp_working_dir = strrep(temp_working_dir, 'gz', '');
    mkdir(temp_working_dir);
    
    imglist = {fullfile(imgdir, subjlist{sj_num}, 'MNINonLinear/Results/rfMRI_REST1_AP/rfMRI_REST1_AP_hp0_clean.nii.gz'); ...
        fullfile(imgdir, subjlist{sj_num}, 'MNINonLinear/Results/rfMRI_REST1_PA/rfMRI_REST1_PA_hp0_clean.nii.gz'); ...
        fullfile(imgdir, subjlist{sj_num}, 'MNINonLinear/Results/rfMRI_REST2_AP/rfMRI_REST2_AP_hp0_clean.nii.gz'); ...
        fullfile(imgdir, subjlist{sj_num}, 'MNINonLinear/Results/rfMRI_REST2_PA/rfMRI_REST2_PA_hp0_clean.nii.gz')};
    
    for img_i = 1:numel(imglist)
        
        if ~exist(imglist{img_i}, 'file'); error('Image files do not exist!'); end
        
        unzip_img = fullfile(temp_working_dir, sprintf('img_%d.nii', img_i));
        system(sprintf('gunzip -c %s > %s', imglist{img_i}, unzip_img));
        
        img_dat{img_i} = fmri_data(unzip_img, brainmask_img);
        img_dat{img_i}.dat = double(img_dat{img_i}.dat);
        img_dat{img_i} = preprocess(img_dat{img_i}, 'smooth', sm_fwhm);
        img_dat{img_i} = canlab_connectivity_preproc(img_dat{img_i}, 'bpf', freq_band, tr, 'no_plots');
        img_dat{img_i}.covariates = mean(img_dat{img_i}.dat, 1).';
        img_dat{img_i} = preprocess(img_dat{img_i}, 'resid', 1);
        img_dat{img_i}.dat = img_dat{img_i}.dat.';
        if any(all(img_dat{img_i}.dat == 0)); system(sprintf('echo %s_run%d >> %s', subjlist{sj_num}, img_i, fullfile(basedir, 'Stroke_share/METAVCI_Network\ analysis/Analysis/scripts/EmptyVox_list_HCPA.txt'))); end
        img_dat{img_i}.dat = zscore(img_dat{img_i}.dat);
        
    end
    
    concat_img_dat = [img_dat{1}.dat; img_dat{2}.dat; img_dat{3}.dat; img_dat{4}.dat];
    seed_ts = zeros(size(concat_img_dat,1), n_lesion);
    
    for les_i = 1:n_lesion
        seed_ts(:,les_i) = mean(concat_img_dat(:, merged_lesion_idx(:,les_i)), 2);
    end
    
    corr_z = atanh(corr(concat_img_dat, seed_ts));
    if any(isnan(corr_z), 1:2); error('NaN corr found!'); end
    corr_z_sum = corr_z_sum + corr_z;
    corr_z_sqsum = corr_z_sqsum + corr_z.^2;
    
    system(['rm -r ' temp_working_dir]);
    
    if mod(sj_num, 100) == 0
        fprintf('Intermediate save...   ');
        save(lesion_conn_z_sum_file, 'corr_z_sum', '-v7.3');
        save(lesion_conn_z_sqsum_file, 'corr_z_sqsum', '-v7.3');
        sj_start = sj_num + 1;
        save(start_idx_file, 'sj_start');
    end
    
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

fprintf('Deleting intermediate corr_z_sum and corr_z_sqsum ...   \n');

delete(start_idx_file);
delete(lesion_conn_z_sum_file);
delete(lesion_conn_z_sqsum_file);

fprintf('Done.\n');
