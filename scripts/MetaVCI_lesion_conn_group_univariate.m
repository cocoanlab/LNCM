basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
figdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/figures');

brain_mask = which('brainmask_canlab_2mm.nii');
brain_maskdat = fmri_data(brain_mask, brain_mask);

vis_mask = which('keuken_2014_enhanced_for_underlay.img');

load(fullfile(datdir, 'colormap_jj.mat'));

load(fullfile(datdir, 'Lesion_Maps/Hallym_lesion_2mm_merged_subject_id.mat'), 'lesion_HAL_id');
load(fullfile(datdir, 'Lesion_Maps/Bundang_lesion_2mm_merged_subject_id.mat'), 'lesion_BUN_id');
merged_id = [lesion_HAL_id; lesion_BUN_id];

T_orig = readtable(fullfile(datdir, 'domain_scores_calculated.csv'));
[~, wh_Tid, wh_Did] = intersect(T_orig.ID, merged_id);
T = T_orig(wh_Tid, :);

%%

fx_zscore = @(x) {x <= -2; x > -1};

wh_targsymp = [fx_zscore(T.attexec), ...
    fx_zscore(T.infospeed), ...
    fx_zscore(T.lang), ...
    fx_zscore(T.vermem), ...
    fx_zscore(T.visuospatial)];
targsymp = {'ATTEXEC', 'INFOSPEED', 'LANG', 'VERBMEM', 'VISSPA'};
targsymp = [strcat(targsymp, '_imp'); strcat(targsymp, '_not')];

disp(array2table(cell2mat(cellfun(@sum, wh_targsymp, 'UniformOutput', false)'), ...
    'VariableNames', {'Imp', 'Not'}, 'RowNames', strrep(targsymp(1,:), '_imp', '')));

%%

norm_conn = 'GSP';

load(fullfile(datdir, sprintf('%s_lesion_conn_group_z_t.mat', norm_conn)), 'corr_z_t');
les_net_t = brain_maskdat;
les_net_t.dat = corr_z_t(:,wh_Did);

%%

conn_thresh_t = 9;
n_thresh = 0.5;

%%

overlap = NaN(size(les_net_t.dat,1), size(wh_targsymp,1), size(wh_targsymp,2));

for symp_i = 1:numel(targsymp)
    
    overlap_each = [sum(les_net_t.dat(:, wh_targsymp{symp_i}) > conn_thresh_t, 2), ...
        sum(les_net_t.dat(:, wh_targsymp{symp_i}) < -conn_thresh_t, 2)];
    overlap_each = max(overlap_each,[],2) .* sign(overlap_each(:,1) - overlap_each(:,2));
    overlap_each = overlap_each ./ sum(wh_targsymp{symp_i});
    overlap_each = overlap_each .* double(abs(overlap_each) >= n_thresh);
    overlap(:,symp_i) = overlap_each;
    
end

overlap_merged = cat(3, sum(sign(overlap) > 0, 3), sum(sign(overlap) < 0, 3));
overlap_merged = max(overlap_merged,[],3) .* sign(overlap_merged(:,:,1) - overlap_merged(:,:,2));

disp(round([diag(corr(squeeze(overlap(:,1,:)), squeeze(overlap(:,2,:)))); corr(overlap_merged(:,1), overlap_merged(:,2))], 2));

%%

for symp_i = 1:numel(targsymp)
    
    cmaprange = [-0.8 -n_thresh n_thresh 0.8];
    les_net_overlap = les_net_t;
    les_net_overlap.dat = overlap(:,symp_i);
    r = region(apply_mask(les_net_overlap, vis_mask));
    brain_activations_display_jj(r, 'depth', 3, 'surface_only', 'surface_list', {'LL', 'LM'}, 'cmaprange', cmaprange, 'camlight_off');
    set(gcf, 'Position', [1         649        781          360]);
    savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_overlap_surf_%s_%s_thr_t%d_n%d.pdf', norm_conn, targsymp{symp_i}, conn_thresh_t, n_thresh*100));
    pagesetup(gcf); saveas(gcf, savename); close all;
    brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'cmaprange', cmaprange, 'camlight_off');
    set(gcf, 'Position', [1         734        1920         251]);
    savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_overlap_%s_%s_thr_t%d_n%d.pdf', norm_conn, targsymp{symp_i}, conn_thresh_t, n_thresh*100));
    pagesetup(gcf); saveas(gcf, savename); close all;

end

%%

poscm = [255 222 224
    255 190 193
    255 153 158
    255 112 120
    245  60  75]./255;

negcm = [ 70 143 208
    121 171 226
    161 196 241
    195 219 253
    225 238 255]./255;

%%

vis_colorbar(poscm);
savename = fullfile(figdir, 'MetaVCI_colorbar6.pdf');
pagesetup(gcf); saveas(gcf, savename); close all;

vis_colorbar(negcm);
savename = fullfile(figdir, 'MetaVCI_colorbar7.pdf');
pagesetup(gcf); saveas(gcf, savename); close all;

%%

merged_all = {'merged_imp', 'merged_not'};

for imp_i = 1:numel(merged_all)
    
    cmaprange = [-1 1] .* size(wh_targsymp,2);
    les_net_overlap = les_net_t;
    les_net_overlap.dat = overlap_merged(:,imp_i);
    r = region(apply_mask(les_net_overlap, vis_mask), 'unique_mask_values');
    brain_activations_display_jj(r, 'depth', 3, 'surface_only', 'surface_list', {'LL', 'LM'}, 'poscm', poscm, 'negcm', negcm, 'cmaprange', cmaprange, 'camlight_off', 'prioritize_last');
    set(gcf, 'Position', [1         649        781          360]);
    savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_overlap_surf_%s_%s_thr_t%d_n%d.pdf', norm_conn, merged_all{imp_i}, conn_thresh_t, n_thresh*100));
    pagesetup(gcf); saveas(gcf, savename); close all;
    brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'poscm', poscm, 'negcm', negcm, 'cmaprange', cmaprange, 'camlight_off', 'prioritize_last');
    set(gcf, 'Position', [1         734        1920         251]);
    savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_overlap_%s_%s_thr_t%d_n%d.pdf', norm_conn, merged_all{imp_i}, conn_thresh_t, n_thresh*100));
    pagesetup(gcf); saveas(gcf, savename); close all;

end

%%

norm_conn = 'GSP';

load(fullfile(datdir, sprintf('%s_lesion_conn_group_r_mean.mat', norm_conn)), 'corr_r_mean');
les_net_z = brain_maskdat;
les_net_z.dat = atanh(corr_r_mean(:,wh_Did));

%%

% sig_labels = {'unthr', 'unc05', 'fdr05', 'fdr01'};
sig_labels = {'fdr01'};
cmaprange = [3 9 -9 -3];

for cont_i = 1:size(targsymp,2)
    
    [~, t_p, ~, t_stat] = ttest2(les_net_z.dat(:, wh_targsymp{1,cont_i})', ...
        les_net_z.dat(:, wh_targsymp{2,cont_i})');
    t_p = t_p.';
    t_t = t_stat.tstat.';
    
    for sig_i = 1:numel(sig_labels)
        
        les_net_contrast = brain_maskdat;
        switch sig_labels{sig_i}
            case 'unthr'
                les_net_contrast.dat = t_t;
            case 'unc05'
                les_net_contrast.dat = t_t .* double(t_p <= 0.05);
            case 'fdr05'
                les_net_contrast.dat = t_t .* double(t_p <= FDR(t_p, 0.05));
            case 'fdr01'
                les_net_contrast.dat = t_t .* double(t_p <= FDR(t_p, 0.01));
        end
        
        if any(les_net_contrast.dat~=0)
            r = region(apply_mask(les_net_contrast, vis_mask));
            brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'cmaprange', cmaprange, 'camlight_off');
            set(gcf, 'Position', [1         734        1920         251]);
            savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_contrast_%s_%s_vs_%s_thr_%s.pdf', norm_conn, targsymp{1,cont_i}, targsymp{2,cont_i}, sig_labels{sig_i}));
            pagesetup(gcf); saveas(gcf, savename); close all;
            les_net_contrast.fullpath = strrep(strrep(savename, figdir, datdir), '.pdf', '.nii');
            write(les_net_contrast, 'overwrite');
            switch targsymp{1,cont_i}
                case 'VERBMEM_imp'
                    r = region(apply_mask(les_net_contrast, vis_mask));
                    cluster_surf(r, 3, which('surf_spm2_hipp.mat'), 'heatmap', cmaprange);
%                     h = addbrain('amygdala'); set(h, 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.1);
%                     h = addbrain('caudate'); set(h, 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.1);
%                     h = addbrain('put'); set(h, 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.1);
                    pagesetup(gcf); saveas(gcf, strrep(savename, '.pdf', '_hippo.pdf')); close all;
            end
        end
        
    end
    
end
