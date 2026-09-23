basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
figdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/figures');

brain_mask = which('brainmask_canlab_2mm.nii');
brain_maskdat = fmri_data(brain_mask, brain_mask);

vis_mask = which('keuken_2014_enhanced_for_underlay.img');

load(fullfile(datdir, 'Lesion_Maps/Hallym_lesion_2mm_merged_subject_id.mat'), 'lesion_HAL_id');
load(fullfile(datdir, 'Lesion_Maps/Bundang_lesion_2mm_merged_subject_id.mat'), 'lesion_BUN_id');
merged_id = [lesion_HAL_id; lesion_BUN_id];

T_orig = readtable(fullfile(datdir, 'domain_scores_calculated.csv'));
[~, wh_Tid, wh_Did] = intersect(T_orig.ID, merged_id);
T = T_orig(wh_Tid, :);

merged_lesion_dat = fmri_data({fullfile(datdir, 'Lesion_Maps/Hallym_lesion_2mm_merged.nii'), ...
    fullfile(datdir, 'Lesion_Maps/Bundang_lesion_2mm_merged.nii')}, brain_mask);
merged_lesion_idx = logical(merged_lesion_dat.dat);
merged_lesion_idx = merged_lesion_idx(:,wh_Did);

n_node = size(merged_lesion_idx, 1); % 242953 voxels
n_lesion = size(merged_lesion_idx, 2); % 1432 lesions

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

poscm = [255,237,160
    254,217,118
    254,178,76
    253,141,60
    252,78,42
    227,26,28
    189,0,38
    128,0,38] ./ 255;
poscm = interp1(1:size(poscm,1), poscm, 1 + (1 - flip(linspace(0,1,100) .^ 2)).*(size(poscm,1)-1) );
vis_colorbar(poscm);
savename = fullfile(figdir, 'MetaVCI_colorbar1.pdf');
pagesetup(gcf); saveas(gcf, savename); close all;

poscm = colormap_tor([0.96 0.41 0], [1 1 0]);  % warm
negcm = colormap_tor([.23 1 1], [0.11 0.46 1]);  % cools
vis_colorbar(poscm);
savename = fullfile(figdir, 'MetaVCI_colorbar2.pdf');
pagesetup(gcf); saveas(gcf, savename); close all;
vis_colorbar(negcm);
savename = fullfile(figdir, 'MetaVCI_colorbar3.pdf');
pagesetup(gcf); saveas(gcf, savename); close all;

%%

thr_les = 3;
% sig_labels = {'unthr', 'unc05', 'fdr05', 'fdr01'};
sig_labels = {'fdr01'};

cmaprange = [eps 6 -6 -eps];

for cont_i = 1:size(targsymp,2)
    
    wh_anysymp = wh_targsymp{1,cont_i} | wh_targsymp{2,cont_i};
    
    X = double(merged_lesion_idx(:, wh_anysymp)).';
    Y = double(wh_targsymp{1,cont_i}(wh_anysymp));
    
    wh_lesion = sum(X) >= thr_les;
    
    cont_tab = [sum(X&Y); sum(X&~Y); sum(~X&Y); sum(~X&~Y)].';
    cont_tab = cont_tab(wh_lesion, :);
    
    fisher_OR = cont_tab(:,1) .* cont_tab(:,4) ./ cont_tab(:,2) ./ cont_tab(:,3);
    wh_errOR = isinf(fisher_OR) | fisher_OR == 0;
    fisher_OR(wh_errOR) = (cont_tab(wh_errOR,1) + 0.5) .* (cont_tab(wh_errOR,4) + 0.5) ...
        ./ (cont_tab(wh_errOR,2) + 0.5) ./ (cont_tab(wh_errOR,3) + 0.5); % Haldane correction
    fisher_p = NaN(sum(wh_lesion), 1);
    for vox_i = 1:sum(wh_lesion)
        [~, fisher_p(vox_i)] = fishertest([cont_tab(vox_i, 1:2); cont_tab(vox_i, 3:4)]);
    end
    
    for sig_i = 1:numel(sig_labels)
        
        les_symp_oddsratio = brain_maskdat;
        les_symp_oddsratio.dat = zeros(size(les_symp_oddsratio.dat,1), 1);
        switch sig_labels{sig_i}
            case 'unthr'
                les_symp_oddsratio.dat(wh_lesion) = log(fisher_OR);
            case 'unc05'
                les_symp_oddsratio.dat(wh_lesion) = log(fisher_OR) .* double(fisher_p <= 0.05);
            case 'fdr05'
                les_symp_oddsratio.dat(wh_lesion) = log(fisher_OR) .* double(fisher_p <= FDR(fisher_p, 0.05));
            case 'fdr01'
                les_symp_oddsratio.dat(wh_lesion) = log(fisher_OR) .* double(fisher_p <= FDR(fisher_p, 0.01));
        end
        
        if any(les_symp_oddsratio.dat~=0)
            r = region(apply_mask(les_symp_oddsratio, vis_mask));
            out = brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'cmaprange', cmaprange, 'camlight_off');
            set(gcf, 'Position', [1         734        1920         251]);
            savename = fullfile(figdir, sprintf('MetaVCI_lesion_symp_oddsratio_%s_vs_%s_thr_%s.pdf', targsymp{1,cont_i}, targsymp{2,cont_i}, sig_labels{sig_i}));
            pagesetup(gcf); saveas(gcf, savename); close all;
        end
    end
    
end
