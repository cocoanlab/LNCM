%%
basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
figdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/figures');

brain_mask = which('brainmask_canlab_2mm.nii');
brain_maskdat = fmri_data(brain_mask, brain_mask);

vis_mask = which('keuken_2014_enhanced_for_underlay.img');

norm_conn = 'GSP';
atlas_name = 'Schaefer100';
atlas_img = which('Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii');
atlas_dat = fmri_data(atlas_img, brain_mask);
atlas_r = region(apply_mask(atlas_dat, vis_mask), 'unique_mask_values');
atlas_conn_file = fullfile(datdir, sprintf('%s_atlas_%s_group_r_z.mat', norm_conn, atlas_name));
load(atlas_conn_file, 'atlas_z_t');

load(which('Schaefer_Net_Labels_r100.mat'), 'Schaefer_Net_Labels');
atlas_netnames = {'VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPN', 'DMN'};

%%

conn_thresh_t = 9;
n_thresh = 0.50;
les_n = 20;
boot_n = 10000;
do_rep = true;
p_hetero = round(0:0.05:1, 2);
Ci = Schaefer_Net_Labels.dat(:,2);

cols_pos = [255   247   243
   253   224   221
   252   197   192
   250   159   181
   247   104   161
   234    78   156
   221    52   151
   198    27   139
   174     1   126
   148     1   123
   122     1   119
    98     1   113
    73     0   106] ./ 255;
cols_neg = [255   247   251
   236   231   242
   208   209   230
   166   189   219
   116   169   207
    85   157   200
    54   144   192
    30   128   184
     5   112   176
     5   101   159
     4    90   141
     3    73   115
     2    56    88] ./ 255;

% cols_pos = [255 255 255
%     232 205 222
%     222 169 204
%     211 133 187
%     199  93 170] ./ 255;
% cols_neg = [255 255 255
%     186 218 220
%     125 197 199
%     0 176 179
%     0 155 159] ./ 255;

cols_pos = interp1(1:size(cols_pos,1), cols_pos, 1 + linspace(0,1,100).^0.8.*(size(cols_pos,1)-1) );
cols_neg = interp1(1:size(cols_neg,1), cols_neg, 1 + linspace(0,1,100).^0.8.*(size(cols_neg,1)-1) );

% cols = interp1(1:size(cols,1), cols, 1 + (1 - flip(linspace(0,1,100) .^ 3)).*(size(cols,1)-1) );
% cols = interp1(1:size(cols,1), cols, 1 + (linspace(0,1,100) .^ 3).*(size(cols,1)-1) );

net_cols = [122 18 136;
    72 130 183;
    2 118 17;
    198 58 252;
    222 248 166;
    233 148 36;
    207 62 80] ./ 255;
    

%%

vis_colorbar(cols_pos);
savename = fullfile(figdir, 'MetaVCI_colorbar4.pdf');
pagesetup(gcf);
saveas(gcf, savename);
close all;

vis_colorbar(cols_neg);
savename = fullfile(figdir, 'MetaVCI_colorbar5.pdf');
pagesetup(gcf);
saveas(gcf, savename);
close all;

%%

A = reformat_r_new(atlas_z_t, 'reconstruct');
A = double(abs(A) > conn_thresh_t) .* sign(A);
A_pos = A .* double(A > 0);
A_neg = A .* double(A < 0);

%%

h = vis_network(A_pos, 'gravity', 'group', Ci, 'groupcolor', net_cols, 'line_pos_color', [0.5 0.5 0.5 0.3], 'manual_node_size', sum(A_pos).*5);
set(gcf, 'Position', [441   232   867   700]);
savename = fullfile(figdir, sprintf('MetaVCI_simulate_whole_conn_network_%s_%s_t%d_pos.pdf', norm_conn, atlas_name, conn_thresh_t));
pagesetup(gcf);
saveas(gcf, savename);
close all;

h = vis_network(-A_neg, 'gravity', 'group', Ci, 'groupcolor', net_cols, 'line_pos_color', [0.5 0.5 0.5 0.3], 'manual_node_size', sum(-A_neg).*5);
set(gcf, 'Position', [441   232   867   700]);
savename = fullfile(figdir, sprintf('MetaVCI_simulate_whole_conn_network_%s_%s_t%d_neg.pdf', norm_conn, atlas_name, conn_thresh_t));
pagesetup(gcf);
saveas(gcf, savename);
close all;

%%

[les_net_overlap_pos_mean, les_net_overlap_neg_mean] = deal(zeros(size(A,1), numel(p_hetero), numel(unique(Ci))));

for net_i = unique(Ci).'
    
    les_net_overlap_pos = zeros(size(A,1), numel(p_hetero), boot_n);
    les_net_overlap_neg = zeros(size(A,1), numel(p_hetero), boot_n);

    for p_i = 1:numel(p_hetero)
        rng('default');
        for boot_i = 1:boot_n
            hetero_idx = rand(les_n, 1);
            les_idx = [randsample(find(Ci == net_i), sum(hetero_idx > p_hetero(p_i)), do_rep); ...
                randsample((1:numel(Ci)).', sum(hetero_idx <= p_hetero(p_i)), do_rep)];
            les_net_overlap_pos(:, p_i, boot_i) = sum(A_pos(les_idx,:)) ./ les_n;
            les_net_overlap_neg(:, p_i, boot_i) = sum(-A_neg(les_idx,:)) ./ les_n;
        end
    end
    
    les_net_overlap_pos_mean(:, :, net_i) = mean(les_net_overlap_pos > n_thresh, 3);
    les_net_overlap_neg_mean(:, :, net_i) = mean(les_net_overlap_neg > n_thresh, 3);
    
end

%%

for net_i = [1 2 7]
    
    for p_i = find(ismember(p_hetero, [0 0.7 0.8 0.95]))

        cols_pos_each = interp1(linspace(0, 1, size(cols_pos,1)), cols_pos, les_net_overlap_pos_mean(:,p_i,net_i));
        cols_neg_each = interp1(linspace(0, 1, size(cols_neg,1)), cols_neg, les_net_overlap_neg_mean(:,p_i,net_i));
        [~, cols_all_each_idx] = max([les_net_overlap_pos_mean(:,p_i,net_i) les_net_overlap_neg_mean(:,p_i,net_i)], [], 2);
        cols_all_each = NaN(size(cols_pos_each));
        cols_all_each(cols_all_each_idx==1,:) = cols_pos_each(cols_all_each_idx==1,:);
        cols_all_each(cols_all_each_idx==2,:) = cols_neg_each(cols_all_each_idx==2,:);

        brain_activations_display_jj(atlas_r, 'depth', 3, 'surface_only', 'surface_list', {'LL', 'LM'}, 'region_color', cols_pos_each, 'camlight_off');
        set(gcf, 'Position', [1         649        781          360]);
        savename = fullfile(figdir, sprintf('MetaVCI_simulate_lesion_conn_overlap_pos_%s_%s_t%d_les%d_n%d_init%s_change%.3d.pdf', norm_conn, atlas_name, conn_thresh_t, les_n, n_thresh*100, atlas_netnames{net_i}, p_hetero(p_i)*100));
        pagesetup(gcf);
        saveas(gcf, savename);
        close all;
        
%         brain_activations_display_jj(atlas_r, 'depth', 3, 'surface_only', 'surface_list', {'LL', 'LM'}, 'region_color', cols_neg_each, 'camlight_off');
%         set(gcf, 'Position', [1         649        781          360]);
%         savename = fullfile(figdir, sprintf('MetaVCI_simulate_lesion_conn_overlap_neg_%s_%s_t%d_les%d_n%d_init%s_change%.3d.pdf', norm_conn, atlas_name, conn_thresh_t, les_n, n_thresh*100, atlas_netnames{net_i}, p_hetero(p_i)*100));
%         pagesetup(gcf);
%         saveas(gcf, savename);
%         close all;
%         
%         brain_activations_display_jj(atlas_r, 'depth', 3, 'surface_only', 'surface_list', {'LL', 'LM'}, 'region_color', cols_all_each, 'camlight_off');
%         set(gcf, 'Position', [1         649        781          360]);
%         savename = fullfile(figdir, sprintf('MetaVCI_simulate_lesion_conn_overlap_all_%s_%s_t%d_les%d_n%d_init%s_change%.3d.pdf', norm_conn, atlas_name, conn_thresh_t, les_n, n_thresh*100, atlas_netnames{net_i}, p_hetero(p_i)*100));
%         pagesetup(gcf);
%         saveas(gcf, savename);
%         close all;

    end
    
end


%%

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

sim_overlap = NaN(size(atlas_dat.dat,1), numel(p_hetero), numel(unique(Ci)));
for net_i = unique(Ci).'
    for p_i = 1:numel(p_hetero)
        for r_i = 1:numel(Ci)
            sim_overlap(atlas_dat.dat == r_i, p_i, net_i) = les_net_overlap_pos_mean(r_i, p_i, net_i);
        end
    end
end

load(fullfile(datdir, sprintf('%s_lesion_conn_group_z_t.mat', norm_conn)), 'corr_z_t');
les_net_t = brain_maskdat;
les_net_t.dat = corr_z_t(:,wh_Did);

conn_thresh_t = 9;
n_thresh = 0.5;

overlap = NaN(size(les_net_t.dat,1), size(wh_targsymp,1), size(wh_targsymp,2));
overlap_pos = NaN(size(les_net_t.dat,1), size(wh_targsymp,1), size(wh_targsymp,2));

for symp_i = 1:numel(targsymp)
    
    overlap_each = [sum(les_net_t.dat(:, wh_targsymp{symp_i}) > conn_thresh_t, 2), ...
        sum(les_net_t.dat(:, wh_targsymp{symp_i}) < -conn_thresh_t, 2)];
    overlap_each = max(overlap_each,[],2) .* sign(overlap_each(:,1) - overlap_each(:,2));
    overlap_each = overlap_each ./ sum(wh_targsymp{symp_i});
    overlap_each = overlap_each .* double(abs(overlap_each) >= n_thresh);
    overlap(:,symp_i) = overlap_each;
        
    overlap_each = sum(les_net_t.dat(:, wh_targsymp{symp_i}) > conn_thresh_t, 2);
    overlap_each = overlap_each ./ sum(wh_targsymp{symp_i});
    overlap_each = overlap_each .* double(abs(overlap_each) >= n_thresh);
    overlap_pos(:,symp_i) = overlap_each;
    
end

overlap_merged = cat(3, sum(sign(overlap) > 0, 3), sum(sign(overlap) < 0, 3));
overlap_merged = max(overlap_merged,[],3) .* sign(overlap_merged(:,:,1) - overlap_merged(:,:,2));

overlap_merged_pos = sum(sign(overlap) > 0, 3);

wh_nonan = ~any(isnan(sim_overlap), 2:3);

sim_overlap_corr = corr(reshape(sim_overlap(wh_nonan,:,:), sum(wh_nonan), []), overlap_merged_pos(wh_nonan,1));
sim_overlap_corr = reshape(sim_overlap_corr, size(sim_overlap,2), size(sim_overlap,3));

% sim_overlap_corrup = tanh(atanh(sim_overlap_corr) + norminv(0.975) / (sum(wh_nonan)-3).^0.5);
% sim_overlap_corrlo = tanh(atanh(sim_overlap_corr) - norminv(0.975) / (sum(wh_nonan)-3).^0.5);
% figure; hold on;
% for net_i = unique(Ci).'
%     h = errorbar(1:size(sim_overlap_corr,1), sim_overlap_corr(:,net_i), ...
%         sim_overlap_corr(:,net_i)-sim_overlap_corrlo(:,net_i), ...
%         sim_overlap_corrup(:,net_i)-sim_overlap_corr(:,net_i), ...
%         '-o', 'CapSize', 0, 'Color', net_cols(net_i,:), 'LineWidth', 2);
% end

plot(sim_overlap_corr, 'LineWidth', 2);
colororder(gca, net_cols);
xlabel('{\itP}_{Hetero}');
ylabel('Overlap similarity ({\itr})');
set(gca, 'XTick', 1:4:21, 'XTickLabels', p_hetero(1:4:21), 'XLim', [0 22], 'FontSize', 18, 'Box', 'off', 'TickDir', 'out');
% set(gcf, 'color', 'w', 'position', [704   491   406   465]);
set(gcf, 'color', 'w', 'position', [704   491   506   465]);
savename = fullfile(figdir, sprintf('MetaVCI_simulate_lesion_conn_overlap_corrwithreal_all_pos_%s_%s_t%d_les%d_n%d.pdf', norm_conn, atlas_name, conn_thresh_t, les_n, n_thresh*100));
pagesetup(gcf); saveas(gcf, savename); close all;

for net_i = [1 2 7]
    plot(sim_overlap_corr(:,net_i), 'LineWidth', 2, 'Color', net_cols(net_i,:));
    xlabel('{\itP}_{Hetero}');
    ylabel('Overlap similarity ({\itr})');
    set(gca, 'XTick', 1:4:21, 'XTickLabels', p_hetero(1:4:21), 'XLim', [0 22], 'FontSize', 18, 'Box', 'off', 'TickDir', 'out');
%     set(gcf, 'color', 'w', 'position', [704   491   406   465]);
    set(gcf, 'color', 'w', 'position', [704   491   506   465]);
    savename = fullfile(figdir, sprintf('MetaVCI_simulate_lesion_conn_overlap_corrwithreal_%s_pos_%s_%s_t%d_les%d_n%d.pdf', atlas_netnames{net_i}, norm_conn, atlas_name, conn_thresh_t, les_n, n_thresh*100));
    pagesetup(gcf); saveas(gcf, savename); close all;
end
