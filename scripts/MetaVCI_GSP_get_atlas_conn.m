%%
basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
imgdir = '/Volumes/cocoanlab03/open_dataset/GSP1000';

subjlist = importdata(fullfile(imgdir, 'GSP1000.txt'));
n_subj = numel(subjlist);

%%

atlas_name = 'Schaefer100';
atlas_img = which('Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii');
atlas_dat = fmri_data(atlas_img, atlas_img);
u_atlas = unique(atlas_dat.dat(atlas_dat.dat~=0));

%%

norm_conn = 'GSP';
atlas_ts_file = fullfile(datdir, sprintf('%s_atlas_%s_ts.mat', norm_conn, atlas_name));
atlas_conn_file = fullfile(datdir, sprintf('%s_atlas_%s_group_r_z.mat', norm_conn, atlas_name));

%%

for sj_num = 1:n_subj
    
    fprintf('Working on Subject %.3d  -  %s ... \n', sj_num, datetime);
    
    imglist = filenames(fullfile(imgdir, subjlist{sj_num}, 'func/sub-*_finalmask.nii'));
    concat_img_dat = [];
    
    for img_i = 1:numel(imglist)
        
        img_dat{img_i} = fmri_data(imglist{img_i}, atlas_img);
        img_dat{img_i}.dat = zscore(double(img_dat{img_i}.dat'));
        concat_img_dat = [concat_img_dat; img_dat{img_i}.dat];
        
    end
    
    atlas_ts{sj_num} = zeros(size(concat_img_dat,1), numel(u_atlas));
    for reg_i = 1:numel(u_atlas)
        atlas_ts{sj_num}(:,reg_i) = mean(concat_img_dat(:, atlas_dat.dat == u_atlas(reg_i)), 2);
    end
    
    fprintf('Done.\n');
    
end

save(atlas_ts_file, 'atlas_ts');

%%

atlas_r = zeros(size(atlas_ts{1},2) * (size(atlas_ts{1},2)-1) / 2, n_subj);

for sj_num = 1:n_subj
    
    atlas_r(:,sj_num) = reformat_r_new(corr(atlas_ts{sj_num}), 'flatten');
    
end

atlas_z = atanh(atlas_r);
atlas_z_t = mean(atlas_z, 2) ./ (std(atlas_z, [], 2) ./ (n_subj .^ 0.5));
atlas_r_mean = tanh(mean(atlas_z, 2));

save(atlas_conn_file, 'atlas_r', 'atlas_z_t', 'atlas_r_mean');
