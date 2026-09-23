basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
figdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/figures');

brain_mask = fullfile(datdir, 'brainmask_canlab_3mm.nii');
brain_maskdat = fmri_data(brain_mask, brain_mask);
n_node = size(brain_maskdat.dat, 1); % 70098 voxels

vis_mask = which('keuken_2014_enhanced_for_underlay.img');

load(fullfile(datdir, 'colormap_jj.mat'));

%%

norm_conn = 'GSP';

voxel_conn_z_t_file = fullfile(datdir, sprintf('%s_voxel_conn_group_z_t.mat', norm_conn));
fid = fopen(voxel_conn_z_t_file, 'r');
corr_z_t = fread(fid, [n_node n_node], '*single');
fclose(fid);

%%

conn_thresh_t = 9;
n_thresh = 0.25;

%%

voxdeg_each = [sum(corr_z_t > conn_thresh_t, 2), ...
    sum(corr_z_t < -conn_thresh_t, 2)];
voxdeg_each = max(voxdeg_each,[],2) .* sign(voxdeg_each(:,1) - voxdeg_each(:,2));
voxdeg_each = voxdeg_each ./ n_node;
voxdeg_each = voxdeg_each .* double(abs(voxdeg_each) >= n_thresh);
voxdeg = voxdeg_each;

%%

cmaprange = [-0.4 -n_thresh n_thresh 0.4];
vox_net_voxdeg = brain_maskdat;
vox_net_voxdeg.dat = voxdeg;
r = region(apply_mask(vox_net_voxdeg, vis_mask));
brain_activations_display_jj(r, 'depth', 4, 'surface_only', 'surface_list', {'LL', 'LM'}, 'cmaprange', cmaprange, 'camlight_off');
set(gcf, 'Position', [1         649        781          360]);
savename = fullfile(figdir, sprintf('MetaVCI_voxel_conn_voxdeg_surf_%s_thr_t%d_n%d.pdf', norm_conn, conn_thresh_t, n_thresh*100));
pagesetup(gcf); saveas(gcf, savename); close all;
