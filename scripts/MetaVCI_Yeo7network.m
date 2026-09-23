basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
figdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/figures');

brain_mask = which('brainmask_canlab_2mm.nii');
brain_maskdat = fmri_data(brain_mask, brain_mask);

vis_mask = which('keuken_2014_enhanced_for_underlay.img');

load(fullfile(datdir, 'colormap_jj.mat'));

%%

Yeo_7net = fmri_data(which('Yeo_10networks_2mm.nii'), brain_mask);
Yeo_7net.dat(Yeo_7net.dat >= 8) = 0;
Yeo_7net_col = [120    18   134
    70   130   180
     0   118    14
   196    58   250
   220   248   164
   230   148    34
   205    62    78] ./ 255;
r = region(apply_mask(Yeo_7net, vis_mask), 'unique_mask_values');
brain_activations_display_jj(r, 'depth', 3, 'surface_only', 'surface_list', {'LL', 'LM'}, 'region_color', Yeo_7net_col, 'camlight_off');
set(gcf, 'Position', [1         649        781          360]);
savename = fullfile(figdir, 'MetaVCI_Yeo7network.pdf');
pagesetup(gcf); saveas(gcf, savename); close all;
