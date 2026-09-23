basedir = '/Volumes/cocoanlab02/projects';
datdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/data');
figdir = fullfile(basedir, 'Stroke_share/METAVCI_Network analysis/Analysis/figures');

brain_mask = which('brainmask_canlab_2mm.nii');
brain_maskdat = fmri_data(brain_mask, brain_mask);

load(fullfile(datdir, 'Lesion_Maps/Hallym_lesion_2mm_merged_subject_id.mat'), 'lesion_HAL_id');
load(fullfile(datdir, 'Lesion_Maps/Bundang_lesion_2mm_merged_subject_id.mat'), 'lesion_BUN_id');
merged_id = [lesion_HAL_id; lesion_BUN_id];

T_orig = readtable(fullfile(datdir, 'domain_scores_calculated.csv'));
[~, wh_Tid, wh_Did] = intersect(T_orig.ID, merged_id);
T = T_orig(wh_Tid, :);

fx_zscore = @(x) {x <= -2; x > -1};
wh_targsymp = [fx_zscore(T.attexec), fx_zscore(T.infospeed), fx_zscore(T.lang), ...
    fx_zscore(T.vermem), fx_zscore(T.visuospatial)];
targtest = {'ATTEXEC', 'INFOSPEED', 'LANG', 'VERBMEM', 'VISSPA'};

% same nuisance design as the prediction and univariate cadj scripts
C_all = [T.interval_onset_to_NP - mean(T.interval_onset_to_NP), double(startsWith(T.ID, "H"))];
C_all = [ones(size(C_all,1),1) C_all C_all(:,1).*C_all(:,2)];

n_dom = numel(targtest);
n_vox = size(brain_maskdat.dat, 1);

%% Recompute raw LNCM t-maps

% only the FDR-thresholded contrasts were written to .nii, so the
% unthresholded maps are recomputed here from each connectome

t_GSP = NaN(n_vox, n_dom); t_cadj = NaN(n_vox, n_dom); t_HCPA = NaN(n_vox, n_dom);

for norm_conn = {'GSP', 'HCPA'}

    fprintf('Loading %s connectome ... \n', norm_conn{1});
    load(fullfile(datdir, sprintf('%s_lesion_conn_group_r_mean.mat', norm_conn{1})), 'corr_r_mean');
    les_net_z = atanh(corr_r_mean(:, wh_Did));
    clear corr_r_mean

    for cont_i = 1:n_dom

        g1 = wh_targsymp{1, cont_i};    % impaired
        g2 = wh_targsymp{2, cont_i};    % not impaired

        % published contrast: two-sample t, equal variance
        [~, ~, ~, st] = ttest2(les_net_z(:, g1).', les_net_z(:, g2).');
        if strcmp(norm_conn{1}, 'GSP')
            t_GSP(:, cont_i) = st.tstat.';
        else
            t_HCPA(:, cont_i) = st.tstat.';
        end

        % covariate-adjusted contrast, GSP only
        if strcmp(norm_conn{1}, 'GSP')
            wh_both = g1 | g2;
            g = double(g1(wh_both));
            X = [g, C_all(wh_both, :)];
            Y = les_net_z(:, wh_both).';
            n = size(X,1); p = size(X,2);
            b = X \ Y;
            s2 = sum((Y - X*b) .^ 2, 1) ./ (n - p);
            XtXinv = inv(X.' * X);
            t_cadj(:, cont_i) = (b(1,:) ./ sqrt(XtXinv(1,1) .* s2)).';
            clear Y b
        end

    end

    clear les_net_z

end

%% LNCM similarity against the published GSP contrast

V1 = NaN(n_dom, 6);
for cont_i = 1:n_dom

    f_GSP  = fullfile(datdir, sprintf('MetaVCI_lesion_conn_contrast_GSP_%s_imp_vs_%s_not_thr_fdr01.nii', ...
        targtest{cont_i}, targtest{cont_i}));
    a = fmri_data(f_GSP, brain_mask);
    b = fmri_data(strrep(f_GSP, '.nii', '_cadj.nii'), brain_mask);
    c = fmri_data(strrep(f_GSP, '_GSP_', '_HCPA_'), brain_mask);

    V1(cont_i,:) = [corr(t_GSP(:,cont_i), t_cadj(:,cont_i)), corr(a.dat, b.dat), sum(b.dat~=0), ...
                    corr(t_GSP(:,cont_i), t_HCPA(:,cont_i)), corr(a.dat, c.dat), sum(c.dat~=0)];
    if cont_i == 1, n_GSP = NaN(n_dom,1); end
    n_GSP(cont_i) = sum(a.dat~=0);

end
T_lncm = array2table([V1(:,1:3), n_GSP, V1(:,4:6)], 'RowNames', targtest, 'VariableNames', ...
    {'cadj_r_raw','cadj_r_thr','cadj_nvox','GSP_nvox', ...
     'HCPA_r_raw','HCPA_r_thr','HCPA_nvox'});

disp('--- LNCM contrast vs published GSP ---');
fprintf('    r_raw = unthresholded t maps; r_thr = FDR .01 maps over all voxels\n');
fprintf('    (r_thr matches the w_les_r convention in the published nestcv script)\n');
fprintf('    nvox = suprathreshold count, given because r_thr is sensitive to map size\n');
disp(T_lncm);

%% Ridge-PCR prediction accuracy

G = load(fullfile(datdir, 'GSP_lesion_conn_group_predict_model_nestcv.mat'), 'mdl');
H = load(fullfile(datdir, 'HCPA_lesion_conn_group_predict_model_nestcv.mat'), 'mdl');

V2 = NaN(n_dom, 6);
for cont_i = 1:n_dom
    V2(cont_i,:) = [G.mdl(cont_i).optall.r2, H.mdl(cont_i).optall.r2, ...
        G.mdl(cont_i).optall.ocv_best_ncomp, H.mdl(cont_i).optall.ocv_best_ncomp, ...
        G.mdl(cont_i).optall.ocv_best_lambda, H.mdl(cont_i).optall.ocv_best_lambda];
end
T_r2 = array2table(V2, 'RowNames', targtest, 'VariableNames', ...
    {'GSP_R2','HCPA_R2','GSP_ncomp','HCPA_ncomp','GSP_lambda','HCPA_lambda'});

disp('--- Ridge-PCR cross-validated R2: GSP vs HCP-Aging connectome ---');
fprintf('    ncomp / lambda are the outer-CV selected values; large differences mean the\n');
fprintf('    two models differ in complexity as well as in connectome\n');
disp(T_r2);