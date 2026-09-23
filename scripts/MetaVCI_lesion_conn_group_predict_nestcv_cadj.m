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

targtest = {'ATTEXEC', 'INFOSPEED', 'LANG', 'VERBMEM', 'VISSPA'};
targscore = {T.attexec, ...
    T.infospeed, ...
    T.lang, ...
    T.vermem, ...
    T.visuospatial};
wh_targscore = cellfun(@(x) ~isnan(x), targscore, 'UniformOutput', false);

disp(array2table(cell2mat(cellfun(@sum, wh_targscore, 'UniformOutput', false)'), ...
    'VariableNames', {'All'}, 'RowNames', targtest));

fx_stratify = @(x) double(x > -1).*1 + double(x <= -1 & x > -2).*2 + double(x <= -2).*3;

targstratify = [fx_stratify(T.attexec), ...
    fx_stratify(T.infospeed), ...
    fx_stratify(T.lang), ...
    fx_stratify(T.vermem), ...
    fx_stratify(T.visuospatial)];

%%

norm_conn = 'GSP';

load(fullfile(datdir, sprintf('%s_lesion_conn_group_r_mean.mat', norm_conn)), 'corr_r_mean');
les_net_z = brain_maskdat;
les_net_z.dat = atanh(corr_r_mean(:,wh_Did));

%%

C_all = [T.interval_onset_to_NP - mean(T.interval_onset_to_NP) double(startsWith(T.ID, "H"))];
C_all = [ones(size(C_all,1),1) C_all C_all(:,1).*C_all(:,2)];

mdlname = fullfile(datdir, sprintf('%s_lesion_conn_group_predict_model_nestcv.mat', norm_conn));
load(mdlname, 'mdl');

%% Get CV and nested-CV Yfit

thr_lvl = 95;
thr_idx = mdl(1).pc_thr==thr_lvl;

mdlc = [];

for pred_i = 1:numel(targscore)
    
    fprintf('Regression: %s (N = %d) ... \n', targtest{pred_i}, sum(wh_targscore{pred_i}));
    
    X = les_net_z.dat(:, wh_targscore{pred_i}).';
    Y = mdl(pred_i).Y;
    C = C_all(wh_targscore{pred_i}, :);
    cv = mdl(pred_i).cv;
    
    [Yfit_C, Yfit_Xopt, Yfit_X95, Yfit_CXopt, Yfit_CX95] = deal(NaN(numel(Y), 1));

    for cv_i = 1:cv.NumTestSets
        
        fprintf('Outer loop: CV %d / %d ... \n', cv_i, cv.NumTestSets);
        
        otr_X = X(training(cv, cv_i), :);
        otr_Y = Y(training(cv, cv_i));
        ote_X = X(test(cv, cv_i), :);
        otr_C = C(training(cv, cv_i), :);
        ote_C = C(test(cv, cv_i), :);

        [otr_X_pc, ~, ~, ~, otr_X_explained, ~] = pca(otr_X);
        otr_X_explained_cum = cumsum(otr_X_explained);

        % Yfit_C
        
        mdlc_b_C = otr_C \ otr_Y;
        Yfit_C(test(cv, cv_i)) = ote_C * mdlc_b_C;

        % Yfit_Xopt & Yfit_CXopt
        
        ncomp_thr = sum(otr_X_explained_cum < mdl(pred_i).optall.icv_best_pc_thr(cv_i)) + 1;
        otr_X_pc_thr = otr_X_pc(:, 1:ncomp_thr);
        otr_X_sc_thr = otr_X * otr_X_pc_thr;

        mdlc_b_X = ridge(otr_Y, otr_X_sc_thr, mdl(pred_i).optall.icv_best_lambda(cv_i), 0);

        Yfit_Xopt(test(cv, cv_i)) = ote_X * otr_X_pc_thr * mdlc_b_X(2:end) + mdlc_b_X(1);
        
        otr_Y_Cres = otr_Y - otr_C * mdlc_b_C;
        otr_X_sc_thr_Cres = otr_X_sc_thr - otr_C * (otr_C \ otr_X_sc_thr);

        mdlc_b_CX_X = ridge(otr_Y_Cres, otr_X_sc_thr_Cres, mdl(pred_i).optall.icv_best_lambda(cv_i), 0);
        mdlc_b_CX_C = otr_C \ (otr_Y - otr_X_sc_thr * mdlc_b_CX_X(2:end));

        Yfit_CXopt(test(cv, cv_i)) = ote_C * mdlc_b_CX_C + ote_X * otr_X_pc_thr * mdlc_b_CX_X(2:end);

        % Yfit_X95 & Yfit_CX95

        ncomp_thr = sum(otr_X_explained_cum < thr_lvl) + 1;
        otr_X_pc_thr = otr_X_pc(:, 1:ncomp_thr);
        otr_X_sc_thr = otr_X * otr_X_pc_thr;

        mdlc_b_X = ridge(otr_Y, otr_X_sc_thr, mdl(pred_i).optlambda.icv_best_lambda(thr_idx, cv_i), 0);

        Yfit_X95(test(cv, cv_i)) = ote_X * otr_X_pc_thr * mdlc_b_X(2:end) + mdlc_b_X(1);
        
        otr_Y_Cres = otr_Y - otr_C * mdlc_b_C;
        otr_X_sc_thr_Cres = otr_X_sc_thr - otr_C * (otr_C \ otr_X_sc_thr);

        mdlc_b_CX_X = ridge(otr_Y_Cres, otr_X_sc_thr_Cres, mdl(pred_i).optlambda.icv_best_lambda(thr_idx, cv_i), 0);
        mdlc_b_CX_C = otr_C \ (otr_Y - otr_X_sc_thr * mdlc_b_CX_X(2:end));

        Yfit_CX95(test(cv, cv_i)) = ote_C * mdlc_b_CX_C + ote_X * otr_X_pc_thr * mdlc_b_CX_X(2:end);

    end

    assert(max(abs(Yfit_Xopt - mdl(pred_i).optall.ocv_besticv_Yfit)) < 1e-9);
    assert(max(abs(Yfit_X95 - mdl(pred_i).optlambda.ocv_besticv_Yfit(:,thr_idx))) < 1e-9);
    
    mdlc(pred_i).Cadj.Yfit = Yfit_C;
    mdlc(pred_i).PCopt.Yfit = Yfit_Xopt;
    mdlc(pred_i).PCoptCadj.Yfit = Yfit_CXopt;
    mdlc(pred_i).PC95.Yfit = Yfit_X95;
    mdlc(pred_i).PC95Cadj.Yfit = Yfit_CX95;
    
end


%% Get performance and final model weigths

n_boot = 10000;

for pred_i = 1:numel(mdlc)
    
    fprintf('Regression: %s (N = %d) ... \n', targtest{pred_i}, sum(wh_targscore{pred_i}));
    
    X = les_net_z.dat(:, wh_targscore{pred_i}).';
    Y = mdl(pred_i).Y;
    C = C_all(wh_targscore{pred_i}, :);
    cv = mdl(pred_i).cv;

    rng('default');
    boot_idx = randi(numel(Y), numel(Y), n_boot);
    
    X_pc = pca(X);

    for des = ["Cadj", "PCopt", "PCoptCadj", "PC95", "PC95Cadj"]

        mdlc(pred_i).(des).mse = mean((Y - mdlc(pred_i).(des).Yfit) .^ 2);
        mdlc(pred_i).(des).r2 = 1 - (sum((Y - mdlc(pred_i).(des).Yfit) .^ 2) ./ sum((Y - mean(Y)) .^ 2));
        mdlc(pred_i).(des).r2_boot = ...
            1 - (sum((Y(boot_idx) - reshape(mdlc(pred_i).(des).Yfit(boot_idx),numel(Y),n_boot)) .^ 2) ./ sum((Y(boot_idx) - mean(Y(boot_idx))) .^ 2));
        [mdlc(pred_i).(des).r2_ci, mdlc(pred_i).(des).r2_p] = boot2cip(mdlc(pred_i).(des).r2_boot);
        [mdlc(pred_i).(des).r, mdlc(pred_i).(des).r_p] = corr(Y, mdlc(pred_i).(des).Yfit);

    end

    mdlc_b_C = C \ Y;

    mdlc(pred_i).Cadj.int = mdlc_b_C(1);
    mdlc(pred_i).Cadj.cw = mdlc_b_C(2:end);

    n = size(C,1); p = size(C,2);
    s2 = sum((Y - C * mdlc_b_C) .^ 2) ./ (n - p);
    se = sqrt(diag(inv(C.' * C)) .* s2);
    t  = mdlc_b_C ./ se;
    mdlc(pred_i).Cadj.cw_t = t(2:end);
    mdlc(pred_i).Cadj.cw_p = 2 * tcdf(-abs(t(2:end)), n - p);
    mdlc(pred_i).Cadj.df = n - p;

    X_pc_thr = X_pc(:, 1:mdl(pred_i).optall.ocv_best_ncomp);
    X_sc_thr = X * X_pc_thr;

    mdlc_b_X = ridge(Y, X_sc_thr, mdl(pred_i).optall.ocv_best_lambda, 0);

    mdlc(pred_i).PCopt.int = mdlc_b_X(1);
    mdlc(pred_i).PCopt.pcw = mdlc_b_X(2:end);
    mdlc(pred_i).PCopt.voxw = X_pc_thr * mdlc_b_X(2:end);

    Y_Cres = Y - C * mdlc_b_C;
    X_sc_thr_Cres = X_sc_thr - C * (C \ X_sc_thr);

    mdlc_b_CX_X = ridge(Y_Cres, X_sc_thr_Cres, mdl(pred_i).optall.ocv_best_lambda, 0);
    mdlc_b_CX_C = C \ (Y - X_sc_thr * mdlc_b_CX_X(2:end));

    mdlc(pred_i).PCoptCadj.int = mdlc_b_CX_C(1);
    mdlc(pred_i).PCoptCadj.cw = mdlc_b_CX_C(2:end);
    mdlc(pred_i).PCoptCadj.pcw = mdlc_b_CX_X(2:end);
    mdlc(pred_i).PCoptCadj.voxw = X_pc_thr * mdlc_b_CX_X(2:end);

    X_pc_thr = X_pc(:, 1:mdl(pred_i).ncomp(thr_idx));
    X_sc_thr = X * X_pc_thr;

    mdlc_b_X = ridge(Y, X_sc_thr, mdl(pred_i).optlambda.ocv_best_lambda(thr_idx), 0);

    mdlc(pred_i).PC95.int = mdlc_b_X(1);
    mdlc(pred_i).PC95.pcw = mdlc_b_X(2:end);
    mdlc(pred_i).PC95.voxw = X_pc_thr * mdlc_b_X(2:end);

    Y_Cres = Y - C * mdlc_b_C;
    X_sc_thr_Cres = X_sc_thr - C * (C \ X_sc_thr);

    mdlc_b_CX_X = ridge(Y_Cres, X_sc_thr_Cres, mdl(pred_i).optlambda.ocv_best_lambda(thr_idx), 0);
    mdlc_b_CX_C = C \ (Y - X_sc_thr * mdlc_b_CX_X(2:end));

    mdlc(pred_i).PC95Cadj.int = mdlc_b_CX_C(1);
    mdlc(pred_i).PC95Cadj.cw = mdlc_b_CX_C(2:end);
    mdlc(pred_i).PC95Cadj.pcw = mdlc_b_CX_X(2:end);
    mdlc(pred_i).PC95Cadj.voxw = X_pc_thr * mdlc_b_CX_X(2:end);
    
end

savename = fullfile(datdir, sprintf('%s_lesion_conn_group_predict_model_nestcv_cadj.mat', norm_conn));
save(savename, 'mdlc');

%%

T_res = [];
for pred_i = 1:numel(targtest)
    T_res(pred_i,:) = [numel(mdlc(pred_i).PC95Cadj.Yfit), ...
        round(mdlc(pred_i).Cadj.r2,2), ...
        round(mdlc(pred_i).PC95.r2,2),  round(mdlc(pred_i).PC95Cadj.r2,2), ...
        round(mdlc(pred_i).PCopt.r2,2), round(mdlc(pred_i).PCoptCadj.r2,2)];
end
T_res = array2table(T_res, 'RowNames', targtest, 'VariableNames', ...
    {'N', 'R2_covonly', 'PC95_R2', 'PC95_R2cov', 'PCopt_R2', 'PCopt_R2cov'});
disp(T_res);

iv  = T.interval_onset_to_NP;
bin = 1*(iv <= 14) + 2*(iv > 14 & iv <= 90) + 3*(iv > 90 & iv <= 180) + 4*(iv > 180);

binname  = ["0-14", "15-90", "91-180", "181-365", "Total"];
sitename = ["Hallym", "Bundang"];
min_cell = 5;   % below this the median is reported but the quartiles are not

binmask  = [arrayfun(@(k) bin == k, 1:4, 'UniformOutput', false), {true(size(bin))}];
sitemask = {startsWith(T.ID, "H"), ~startsWith(T.ID, "H")};

fx_desc = @(v) [numel(v), median(v), prctile(v, 25), prctile(v, 75)];

[bin_i, site_i] = ndgrid(1:numel(binname), 1:numel(sitename));
rowlab = sitename(site_i(:)) + "_" + binname(bin_i(:));

T_res_desc = NaN(numel(rowlab), 4 * numel(targtest));
for r = 1:numel(rowlab)
    for pred_i = 1:numel(targtest)
        y = targscore{pred_i};
        d = fx_desc(y(~isnan(y) & binmask{bin_i(r)} & sitemask{site_i(r)}));
        if d(1) < min_cell, d(3:4) = NaN; end
        T_res_desc(r, (pred_i-1)*4 + (1:4)) = d;
    end
end

T_res_desc = array2table(T_res_desc, 'RowNames', cellstr(rowlab), 'VariableNames', ...
    cellstr(string(targtest) + ["_n"; "_median"; "_q1"; "_q3"]));
disp(T_res_desc);

T_res_cw = [];
for pred_i = 1:numel(targtest)
    T_res_cw(pred_i,:) = reshape([mdlc(pred_i).Cadj.cw .* [30; 1; 30] mdlc(pred_i).Cadj.cw_t mdlc(pred_i).Cadj.cw_p].', 1, []);
end
T_res_cw = array2table(T_res_cw, 'RowNames', targtest, 'VariableNames', cellstr(["interval", "site", "interaction"] + ["_b", "_t", "_p"].'));
disp(T_res_cw);
