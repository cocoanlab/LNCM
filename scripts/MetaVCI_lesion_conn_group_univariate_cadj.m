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

load(fullfile(datdir, sprintf('%s_lesion_conn_group_r_mean.mat', norm_conn)), 'corr_r_mean');
les_net_z = brain_maskdat;
les_net_z.dat = atanh(corr_r_mean(:,wh_Did));

%%

C_all = [T.interval_onset_to_NP double(startsWith(T.ID, "H"))];
C_all = [ones(size(C_all,1),1) C_all C_all(:,1).*C_all(:,2)];

%%

% sig_labels = {'unthr', 'unc05', 'fdr05', 'fdr01'};
sig_labels = {'fdr01'};
cmaprange = [3 9 -9 -3];

for cont_i = 1:size(targsymp,2)

    wh_both = wh_targsymp{1,cont_i} | wh_targsymp{2,cont_i};
    g = double(wh_targsymp{1,cont_i}(wh_both));
    X = [g, C_all(wh_both, :)];
    Y = les_net_z.dat(:, wh_both).';

    n = size(X,1); p = size(X,2);
    b = X \ Y;
    s2 = sum((Y - X*b) .^ 2, 1) ./ (n - p);
    XtXinv = inv(X.' * X);

    t_t = (b(1,:) ./ sqrt(XtXinv(1,1) .* s2)).';
    t_p = 2 * tcdf(-abs(t_t), n - p);
    
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
            savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_contrast_%s_%s_vs_%s_thr_%s_cadj.pdf', norm_conn, targsymp{1,cont_i}, targsymp{2,cont_i}, sig_labels{sig_i}));
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
