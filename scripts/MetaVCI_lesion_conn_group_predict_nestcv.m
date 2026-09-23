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

%% Get CV and nested-CV Yfit

lambda = logspace(-5, 5, 100);
pc_thr = [80:99 99.1:0.1:99.9];
n_fold = 10;

mdl = [];

for pred_i = 1:numel(targscore)
    
    fprintf('Regression: %s (N = %d) ... \n', targtest{pred_i}, sum(wh_targscore{pred_i}));
    
    X = les_net_z.dat(:, wh_targscore{pred_i}).';
    Y = targscore{pred_i}(wh_targscore{pred_i});
    rng('default');
    cv = cvpartition(targstratify(wh_targscore{pred_i}, pred_i), 'kFold', n_fold, 'Stratify', true);
    
    ocv_Yfit = NaN(numel(Y), numel(lambda), numel(pc_thr));
    icv_Yfit = NaN(numel(Y), numel(lambda), numel(pc_thr), n_fold);

    for cv_i = 1:cv.NumTestSets
        
        fprintf('Outer loop: CV %d / %d ... \n', cv_i, cv.NumTestSets);
        
        otr_X = X(training(cv, cv_i), :);
        otr_Y = Y(training(cv, cv_i));
        ote_X = X(test(cv, cv_i), :);
        
        [otr_X_pc, ~, ~, ~, otr_X_explained, ~] = pca(otr_X);
        otr_X_explained_cum = cumsum(otr_X_explained);
        
        for thr_i = 1:numel(pc_thr)
            
            ncomp_thr = sum(otr_X_explained_cum < pc_thr(thr_i)) + 1;
            otr_X_pc_thr = otr_X_pc(:, 1:ncomp_thr);
            otr_X_sc_thr = otr_X * otr_X_pc_thr;
            
            mdl_b = ridge(otr_Y, otr_X_sc_thr, lambda, 0);
            
            ocv_Yfit(test(cv, cv_i), :, thr_i) = ote_X * otr_X_pc_thr * mdl_b(2:end,:) + mdl_b(1,:);
            
        end
        
        for cv_j = setdiff(1:cv.NumTestSets, cv_i)
            
            cv_j_count = find(cv_j == setdiff(1:cv.NumTestSets, cv_i));
            fprintf('Inner loop: CV %d / %d ... \n', cv_j_count, cv.NumTestSets-1);
            
            itr_X = X(training(cv, cv_i) & training(cv, cv_j), :);
            itr_Y = Y(training(cv, cv_i) & training(cv, cv_j));
            ite_X = X(test(cv, cv_j), :);
            
            [itr_X_pc, ~, ~, ~, itr_X_explained, ~] = pca(itr_X);
            itr_X_explained_cum = cumsum(itr_X_explained);
            
            for thr_i = 1:numel(pc_thr)
                
                ncomp_thr = sum(itr_X_explained_cum < pc_thr(thr_i)) + 1;
                itr_X_pc_thr = itr_X_pc(:, 1:ncomp_thr);
                itr_X_sc_thr = itr_X * itr_X_pc_thr;
                
                mdl_b = ridge(itr_Y, itr_X_sc_thr, lambda, 0);
                
                icv_Yfit(test(cv, cv_j), :, thr_i, cv_i) = ite_X * itr_X_pc_thr * mdl_b(2:end,:) + mdl_b(1,:);
                
            end
            
        end
        
    end
    
    mdl(pred_i).targtest = targtest{pred_i};
    mdl(pred_i).Y = Y;
    mdl(pred_i).cv = cv;
    mdl(pred_i).lambda = lambda;
    mdl(pred_i).pc_thr = pc_thr;
    mdl(pred_i).ocv_Yfit = ocv_Yfit;
    mdl(pred_i).icv_Yfit = icv_Yfit;
    
end

savename = fullfile(datdir, sprintf('%s_lesion_conn_group_predict_model_nestcv.mat', norm_conn));
save(savename, 'mdl');

%% Get performance and final model weigths

n_perm = 10000;
n_boot = 10000;

thr_lvl = 95;

for pred_i = 1:numel(mdl)
    
    fprintf('Regression: %s (N = %d) ... \n', targtest{pred_i}, sum(wh_targscore{pred_i}));
    
    X = les_net_z.dat(:, wh_targscore{pred_i}).';
    Y = mdl(pred_i).Y;
    cv = mdl(pred_i).cv;
    
    rng('default');
    perm_idx = NaN(numel(Y), n_perm);
    for perm_i = 1:n_perm
        perm_idx(:,perm_i) = randperm(numel(Y));
    end

    rng('default');
    boot_idx = randi(numel(Y), numel(Y), n_boot);
    
    [X_pc, ~, ~, ~, X_explained, ~] = pca(X);
    X_explained_cum = cumsum(X_explained);

    mdl(pred_i).ncomp = sum(X_explained_cum < mdl(pred_i).pc_thr) + 1;

    icv_mse = squeeze(mean((mdl(pred_i).Y - mdl(pred_i).icv_Yfit) .^ 2, 1, 'omitnan'));
    ocv_mse = squeeze(mean((mdl(pred_i).Y - mdl(pred_i).ocv_Yfit) .^ 2, 1, 'omitnan'));

    % optall

    fprintf('Optimize Lambda and PCs ...\n');

    [~, icv_wh_best] = min(icv_mse, [], 1:2, 'linear');
    [icv_wh_best_lambda, icv_wh_best_pc_thr, ~] = ind2sub(size(icv_mse), icv_wh_best);
    icv_wh_best_lambda = squeeze(icv_wh_best_lambda);
    icv_wh_best_pc_thr = squeeze(icv_wh_best_pc_thr);

    mdl(pred_i).optall.icv_best_lambda = mdl(pred_i).lambda(icv_wh_best_lambda);
    mdl(pred_i).optall.icv_best_pc_thr = mdl(pred_i).pc_thr(icv_wh_best_pc_thr);
    mdl(pred_i).optall.icv_best_ncomp = sum(X_explained_cum < mdl(pred_i).optall.icv_best_pc_thr) + 1;

    ocv_besticv_Yfit = NaN(numel(Y), 1);
    for cv_i = 1:cv.NumTestSets
        ocv_besticv_Yfit(test(cv, cv_i)) = mdl(pred_i).ocv_Yfit(test(cv, cv_i), ...
            icv_wh_best_lambda(cv_i), icv_wh_best_pc_thr(cv_i));
    end

    mdl(pred_i).optall.ocv_besticv_Yfit = ocv_besticv_Yfit;
    mdl(pred_i).optall.mse = mean((Y - ocv_besticv_Yfit) .^ 2);
    mdl(pred_i).optall.r2 = 1 - (sum((Y - ocv_besticv_Yfit) .^ 2) ./ sum((Y - mean(Y)) .^ 2));
    mdl(pred_i).optall.r2_boot = 1 - (sum((Y(boot_idx) - reshape(ocv_besticv_Yfit(boot_idx),numel(Y),n_boot)) .^ 2) ./ sum((Y(boot_idx) - mean(Y(boot_idx))) .^ 2));
    [mdl(pred_i).optall.r2_ci, mdl(pred_i).optall.r2_p] = boot2cip(mdl(pred_i).optall.r2_boot);
    [mdl(pred_i).optall.r, mdl(pred_i).optall.r_p] = corr(Y, ocv_besticv_Yfit);
    
    [~, ocv_wh_best] = min(ocv_mse, [], 1:2, 'linear');
    [ocv_wh_best_lambda, ocv_wh_best_pc_thr] = ind2sub(size(ocv_mse), ocv_wh_best);

    mdl(pred_i).optall.ocv_best_lambda = mdl(pred_i).lambda(ocv_wh_best_lambda);
    mdl(pred_i).optall.ocv_best_pc_thr = mdl(pred_i).pc_thr(ocv_wh_best_pc_thr);
    mdl(pred_i).optall.ocv_best_ncomp = sum(X_explained_cum < mdl(pred_i).optall.ocv_best_pc_thr) + 1;

    X_pc_thr = X_pc(:, 1:mdl(pred_i).optall.ocv_best_ncomp);
    X_sc_thr = X * X_pc_thr;

    mdl_b = ridge(Y, X_sc_thr, mdl(pred_i).optall.ocv_best_lambda, 0);

    mdl(pred_i).optall.int = mdl_b(1);
    mdl(pred_i).optall.pcw = mdl_b(2:end);
    mdl(pred_i).optall.voxw = X_pc_thr * mdl_b(2:end);

    mdl_b_perm = NaN(mdl(pred_i).optall.ocv_best_ncomp+1, n_perm);
    for perm_i = 1:n_perm
        mdl_b_perm(:,perm_i) = ridge(Y(perm_idx(:,perm_i)), X_sc_thr, mdl(pred_i).optall.ocv_best_lambda, 0);
    end
    mdl_b_perm_pcw = mdl_b_perm(2:end, :);
    mdl_b_perm_voxw = X_pc_thr * mdl_b_perm(2:end, :);

    mdl(pred_i).optall.pcw_p = min([(sum(mdl(pred_i).optall.pcw < mdl_b_perm_pcw, 2) + 1) ./ (n_perm + 1), ...
        (sum(mdl(pred_i).optall.pcw > mdl_b_perm_pcw, 2) + 1) ./ (n_perm + 1)], [], 2) .* 2;
    mdl(pred_i).optall.voxw_p = min([(sum(mdl(pred_i).optall.voxw < mdl_b_perm_voxw, 2) + 1) ./ (n_perm + 1), ...
        (sum(mdl(pred_i).optall.voxw > mdl_b_perm_voxw, 2) + 1) ./ (n_perm + 1)], [], 2) .* 2;

    ncomp_95 = mdl(pred_i).ncomp(mdl(pred_i).pc_thr==thr_lvl);
    mdl(pred_i).optall.voxw_lt95 = X_pc_thr(:,1:ncomp_95) * mdl_b(2:ncomp_95+1);
    mdl(pred_i).optall.voxw_gt95 = X_pc_thr(:,ncomp_95+1:end) * mdl_b(ncomp_95+2:end);
    mdl_b_perm_voxw_lt95 = X_pc_thr(:,1:ncomp_95) * mdl_b_perm(2:ncomp_95+1, :);
    mdl_b_perm_voxw_gt95 = X_pc_thr(:,ncomp_95+1:end) * mdl_b_perm(ncomp_95+2:end, :);
    mdl(pred_i).optall.voxw_lt95_p = min([(sum(mdl(pred_i).optall.voxw_lt95 < mdl_b_perm_voxw_lt95, 2) + 1) ./ (n_perm + 1), ...
        (sum(mdl(pred_i).optall.voxw_lt95 > mdl_b_perm_voxw_lt95, 2) + 1) ./ (n_perm + 1)], [], 2) .* 2;
    mdl(pred_i).optall.voxw_gt95_p = min([(sum(mdl(pred_i).optall.voxw_gt95 < mdl_b_perm_voxw_gt95, 2) + 1) ./ (n_perm + 1), ...
        (sum(mdl(pred_i).optall.voxw_gt95 > mdl_b_perm_voxw_gt95, 2) + 1) ./ (n_perm + 1)], [], 2) .* 2;

    % optlambda

    fprintf('Optimize Lambda ...\n');

    [~, icv_wh_best] = min(icv_mse, [], 1, 'linear');
    [icv_wh_best_lambda, ~, ~] = ind2sub(size(icv_mse), icv_wh_best);
    icv_wh_best_lambda = squeeze(icv_wh_best_lambda);

    mdl(pred_i).optlambda.icv_best_lambda = mdl(pred_i).lambda(icv_wh_best_lambda);

    ocv_besticv_Yfit = NaN(numel(Y), numel(mdl(pred_i).pc_thr));
    for cv_i = 1:cv.NumTestSets
        for thr_i = 1:numel(mdl(pred_i).pc_thr)
            ocv_besticv_Yfit(test(cv, cv_i), thr_i) = mdl(pred_i).ocv_Yfit(test(cv, cv_i), ...
                icv_wh_best_lambda(thr_i, cv_i), thr_i);
        end
    end

    mdl(pred_i).optlambda.ocv_besticv_Yfit = ocv_besticv_Yfit;
    mdl(pred_i).optlambda.mse = mean((Y - ocv_besticv_Yfit) .^ 2);
    mdl(pred_i).optlambda.r2 = 1 - (sum((Y - ocv_besticv_Yfit) .^ 2) ./ sum((Y - mean(Y)) .^ 2));
    [mdl(pred_i).optlambda.r2_boot, mdl(pred_i).optlambda.r2_ci, mdl(pred_i).optlambda.r2_p] = ...
        deal(NaN(n_boot,numel(mdl(pred_i).pc_thr)), NaN(2,numel(mdl(pred_i).pc_thr)), NaN(1,numel(mdl(pred_i).pc_thr)));
    for thr_i = 1:numel(mdl(pred_i).pc_thr)
        mdl(pred_i).optlambda.r2_boot(:,thr_i) = 1 - (sum((Y(boot_idx) - reshape(ocv_besticv_Yfit(boot_idx,thr_i),numel(Y),n_boot)) .^ 2) ./ sum((Y(boot_idx) - mean(Y(boot_idx))) .^ 2));
        [mdl(pred_i).optlambda.r2_ci(:,thr_i), mdl(pred_i).optlambda.r2_p(thr_i)] = boot2cip(mdl(pred_i).optlambda.r2_boot(:,thr_i));
    end
    [mdl(pred_i).optlambda.r, mdl(pred_i).optlambda.r_p] = corr(Y, ocv_besticv_Yfit);

    [~, ocv_wh_best] = min(ocv_mse, [], 1, 'linear');
    [ocv_wh_best_lambda, ~] = ind2sub(size(ocv_mse), ocv_wh_best);

    mdl(pred_i).optlambda.ocv_best_lambda = mdl(pred_i).lambda(ocv_wh_best_lambda);

    mdl(pred_i).optlambda.int = NaN(1, numel(mdl(pred_i).pc_thr));
    mdl(pred_i).optlambda.pcw = NaN(max(mdl(pred_i).ncomp), numel(mdl(pred_i).pc_thr));
    mdl(pred_i).optlambda.voxw = NaN(size(X,2), numel(mdl(pred_i).pc_thr));
    mdl(pred_i).optlambda.pcw_p = NaN(size(mdl(pred_i).optlambda.pcw));
    mdl(pred_i).optlambda.voxw_p = NaN(size(mdl(pred_i).optlambda.voxw));

    for thr_i = 1:numel(mdl(pred_i).pc_thr)

        fprintf('PC threshold %.1f%% ...\n', mdl(pred_i).pc_thr(thr_i));

        X_pc_thr = X_pc(:, 1:mdl(pred_i).ncomp(thr_i));
        X_sc_thr = X * X_pc_thr;

        mdl_b = ridge(Y, X_sc_thr, mdl(pred_i).optlambda.ocv_best_lambda(thr_i), 0);

        mdl(pred_i).optlambda.int(thr_i) = mdl_b(1);
        mdl(pred_i).optlambda.pcw(1:size(mdl_b,1)-1, thr_i) = mdl_b(2:end);
        mdl(pred_i).optlambda.voxw(:, thr_i) = X_pc_thr * mdl_b(2:end);

        mdl_b_perm = NaN(mdl(pred_i).ncomp(thr_i)+1, n_perm);
        for perm_i = 1:n_perm
            mdl_b_perm(:,perm_i) = ridge(Y(perm_idx(:,perm_i)), X_sc_thr, mdl(pred_i).optlambda.ocv_best_lambda(thr_i), 0);
        end
        mdl_b_perm_pcw = mdl_b_perm(2:end, :);
        mdl_b_perm_voxw = X_pc_thr * mdl_b_perm(2:end, :);

        mdl(pred_i).optlambda.pcw_p(1:size(mdl_b,1)-1, thr_i) = ...
            min([(sum(mdl(pred_i).optlambda.pcw(1:size(mdl_b,1)-1, thr_i) < mdl_b_perm_pcw, 2) + 1) ./ (n_perm + 1), ...
            (sum(mdl(pred_i).optlambda.pcw(1:size(mdl_b,1)-1, thr_i) > mdl_b_perm_pcw, 2) + 1) ./ (n_perm + 1)], [], 2) .* 2;
        mdl(pred_i).optlambda.voxw_p(:, thr_i) = ...
            min([(sum(mdl(pred_i).optlambda.voxw(:, thr_i) < mdl_b_perm_voxw, 2) + 1) ./ (n_perm + 1), ...
            (sum(mdl(pred_i).optlambda.voxw(:, thr_i) > mdl_b_perm_voxw, 2) + 1) ./ (n_perm + 1)], [], 2) .* 2;
    end
    
end

savename = fullfile(datdir, sprintf('%s_lesion_conn_group_predict_model_nestcv.mat', norm_conn));
save(savename, 'mdl', '-v7.3');

%%

savename = fullfile(datdir, sprintf('%s_lesion_conn_group_predict_model_nestcv.mat', norm_conn));
load(savename, 'mdl');

%%

thr_lvl = 95;
thr_idx = mdl(1).pc_thr==thr_lvl;

T_res = [];
for pred_i = 1:numel(targtest)
    T_res(pred_i,:) = [numel(mdl(pred_i).Y), ...
        mdl(pred_i).ncomp(thr_idx), ...
        round(mdl(pred_i).optlambda.ocv_best_lambda(thr_idx),1), ...
        round(mdl(pred_i).optlambda.r2(thr_idx),2), ...
        mdl(pred_i).optall.ocv_best_pc_thr, ...
        mdl(pred_i).optall.ocv_best_ncomp, ...
        round(mdl(pred_i).optall.ocv_best_lambda,1), ...
        round(mdl(pred_i).optall.r2,2)];
end
T_res = array2table(T_res, 'RowNames', targtest, 'VariableNames', ...
    {'N', 'ol_ncomp', 'ol_lambda', 'ol_R2', ...
    'oa_pc_thr', 'oa_ncomp', 'oa_lambda', 'oa_R2'});
disp(T_res);

%%

cols = [225 106 134
    170 144   0
      0 170  90
      0 166 202
    182 117 224]./255;

figure; hold on;
for i = 1:5
    % fill([mdl(i).pc_thr fliplr(mdl(i).pc_thr)], [mdl(i).optlambda.r2_ci(2,:) fliplr(mdl(i).optlambda.r2_ci(1,:))], cols(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.15);
    fill([mdl(i).pc_thr fliplr(mdl(i).pc_thr)], [mdl(i).optlambda.r2 + std(mdl(i).optlambda.r2_boot) fliplr(mdl(i).optlambda.r2 - std(mdl(i).optlambda.r2_boot))], cols(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.15);
    plot([mdl(i).pc_thr], [mdl(i).optlambda.r2], 'LineWidth', 2, 'Color', cols(i,:));
end
ylim_range = get(gca, 'ylim');
line([95 95], ylim_range, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
line([99 99], ylim_range, 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
xlabel('Explained variance of PCs');
ylabel('R^2');
xticks([80:5:95 99]);
set(gca, 'FontSize', 14, 'ylim', ylim_range);
set(gcf, 'Position', [540   373   373   257], 'Color', 'w');
savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_pred_r2_%s_%s.pdf', norm_conn, 'Specific'));
pagesetup(gcf);
saveas(gcf, savename);
close all;

%%

sig_labels = {'unthr', 'unc01'};

for pred_i = 1:numel(targtest)
    
    for sig_i = 1:numel(sig_labels)
        
        fprintf('Visualization of regression weights: %s, %s ... \n', targtest{pred_i}, sig_labels{sig_i});
        
        w = -mdl(pred_i).optall.voxw;
        switch sig_labels{sig_i}
            case 'unthr'
            case 'unc01'
                w = w .* double(mdl(pred_i).optall.voxw_p < 0.01);
        end
        
        % cmaprange = prctile(w, [0.2 99.8]); disp(cmaprange);
        cmaprange = max(abs(prctile(w, [0.2 99.8]))) .* [-1 1]; disp(cmaprange);
        w_reg = brain_maskdat;
        w_reg.dat = w;
        r = region(apply_mask(w_reg, vis_mask));
        brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'poscm', col_diff_map3(65:end,:), 'negcm', col_diff_map3(1:65,:), 'cmaprange', cmaprange, 'camlight_off');
        set(gcf, 'Position', [1         734        1920         251]);
        savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_pred_weight_%s_%s_permsig_%s_nestcv_optall.pdf', norm_conn, targtest{pred_i}, sig_labels{sig_i}));
        pagesetup(gcf); saveas(gcf, savename); close all;
        
    end
    
end

%%

thr_lvl = 95;
thr_idx = mdl(1).pc_thr==thr_lvl;

for pred_i = 1:numel(targtest)
    
    for sig_label = {'unthr', 'unc01'}
        
        fprintf('Visualization of regression weights: %s, %s... \n', targtest{pred_i}, sig_label{1});
        
        w = -mdl(pred_i).optlambda.voxw(:, thr_idx);
        switch sig_label{1}
            case 'unthr'
            case 'unc01'
                w = w .* double(mdl(pred_i).optlambda.voxw_p(:, thr_idx) < 0.01);
        end
        
        % cmaprange = prctile(w, [0.2 99.8]); disp(cmaprange);
        cmaprange = max(abs(prctile(w, [0.2 99.8]))) .* [-1 1]; disp(cmaprange);
        w_reg = brain_maskdat;
        w_reg.dat = w;
        r = region(apply_mask(w_reg, vis_mask));
        brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'poscm', col_diff_map3(65:end,:), 'negcm', col_diff_map3(1:65,:), 'cmaprange', cmaprange, 'camlight_off');
        set(gcf, 'Position', [1         734        1920         251]);
        savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_pred_weight_%s_%s_permsig_%s_nestcv_optlambda_PC%d.pdf', norm_conn, targtest{pred_i}, sig_label{1}, thr_lvl));
        pagesetup(gcf); saveas(gcf, savename); close all;
        
    end
    
end


%%

delete(gcp('nocreate')); pool = parpool(28);

thr_lvl = 95;
thr_idx = mdl(1).pc_thr==thr_lvl;

thr_les = 3;

sig_labels = {'unthr', 'unc01'};
split_labels = {'lt95', 'gt95'};

[w_les_r, w_les_r_p] = deal(NaN(numel(split_labels), 2, numel(targtest), numel(sig_labels)));

for pred_i = 1:numel(targtest)
    
    % Lesion-network contrast

    [~, t_p, ~, t_stat] = ttest2(les_net_z.dat(:, wh_targsymp{1,pred_i})', ...
        les_net_z.dat(:, wh_targsymp{2,pred_i})');
    t_p = t_p.';
    t_t = t_stat.tstat.';

    % Lesion-symptom mapping

    wh_anysymp = wh_targsymp{1,pred_i} | wh_targsymp{2,pred_i};

    X = double(merged_lesion_idx(:, wh_anysymp)).';
    Y = double(wh_targsymp{1,pred_i}(wh_anysymp));

    wh_lesion = sum(X) >= thr_les;

    cont_tab = [sum(X&Y); sum(X&~Y); sum(~X&Y); sum(~X&~Y)].';
    cont_tab = cont_tab(wh_lesion, :);

    fisher_OR = cont_tab(:,1) .* cont_tab(:,4) ./ cont_tab(:,2) ./ cont_tab(:,3);
    wh_errOR = isinf(fisher_OR) | fisher_OR == 0;
    fisher_OR(wh_errOR) = (cont_tab(wh_errOR,1) + 0.5) .* (cont_tab(wh_errOR,4) + 0.5) ...
        ./ (cont_tab(wh_errOR,2) + 0.5) ./ (cont_tab(wh_errOR,3) + 0.5); % Haldane correction
    fisher_p = NaN(sum(wh_lesion), 1);
    parfor vox_i = 1:sum(wh_lesion)
        [~, fisher_p(vox_i)] = fishertest([cont_tab(vox_i, 1:2); cont_tab(vox_i, 3:4)]);
    end

    for sig_i = 1:numel(sig_labels)

        for split_i = 1:numel(split_labels)
        
            les_symp_oddsratio = wh_lesion.';
            switch sig_labels{sig_i}
                case 'unthr'
                    les_symp_ttest = t_t;
                    les_symp_oddsratio(les_symp_oddsratio) = log(fisher_OR);
                case 'unc01' % use FDR 0.01 for consistency and to match overall voxel numbers
                    les_symp_ttest = t_t .* double(t_p <= FDR(t_p, 0.01));
                    les_symp_oddsratio(les_symp_oddsratio) = log(fisher_OR) .* double(fisher_p <= FDR(fisher_p, 0.01));
            end

            fprintf('Visualization of regression weights: %s, %s, %s... \n', targtest{pred_i}, sig_labels{sig_i}, split_labels{split_i});

            w = -mdl(pred_i).optall.("voxw_"+split_labels{split_i});
            switch sig_labels{sig_i}
                case 'unthr'
                case 'unc01'
                    w = w .* double(mdl(pred_i).optall.("voxw_"+split_labels{split_i}+"_p") < 0.01);
            end
            % cmaprange = prctile(w, [0.2 99.8]); disp(cmaprange);
            cmaprange = max(abs(prctile(w, [0.2 99.8]))) .* [-1 1]; disp(cmaprange);
            w_reg = brain_maskdat;
            w_reg.dat = w;
            r = region(apply_mask(w_reg, vis_mask));
            brain_activations_display_jj(r, 'depth', 3, 'all2', 'all2_xyz', [-2 2 -20 20 -20:10:30], 'poscm', col_diff_map3(65:end,:), 'negcm', col_diff_map3(1:65,:), 'cmaprange', cmaprange, 'camlight_off');
            set(gcf, 'Position', [1         734        1920         251]);
            savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_pred_weight_%s_%s_pcsplit_%s_permsig_%s_nestcv_optall.pdf', norm_conn, targtest{pred_i}, split_labels{split_i}, sig_labels{sig_i}));
            pagesetup(gcf); saveas(gcf, savename); close all;

            [w_les_r(split_i,:,pred_i,sig_i), w_les_r_p(split_i,:,pred_i,sig_i)] = corr(w, [les_symp_ttest, les_symp_oddsratio]);

        end
        
    end
    
end

%%

col_two = [255   166    48
     0   167   255] ./ 255;

for pred_i = 1:numel(targtest)

    hold on;
    plot(1:2, w_les_r(1,:,pred_i,2), 'LineWidth', 2, 'MarkerSize', 6, 'Color', col_two(1,:));
    scatter(1:2, w_les_r(1,:,pred_i,2), 100, col_two(1,:), 'filled');
    plot(1:2, w_les_r(2,:,pred_i,2), '-o', 'LineWidth', 2, 'MarkerSize', 6, 'Color', col_two(2,:));
    scatter(1:2, w_les_r(2,:,pred_i,2), 100, col_two(2,:), 'filled');
    yl = [min(floor(min(w_les_r(:,:,pred_i,2),[],'all')*10)/10, 0) ceil(max(w_les_r(:,:,pred_i,2),[],'all')*10)/10];
    set(gca, 'LineWidth', 1.5, 'XLim', [0.5 2.5], 'YLim', yl, 'XTick', 1:2, 'XTickLabel', '', ...
        'TickLength', [0.03 0.03], 'Tickdir', 'out', 'Box', 'off', 'FontSize', 16);
    set(gcf, 'Position', [1   734   201   261], 'color', 'w');
    savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_corr_weight_les_%s_%s_nestcv_optall.pdf', norm_conn, targtest{pred_i}));
    pagesetup(gcf); saveas(gcf, savename); close all;

    hold on;
    b = bar(1:2, w_les_r(:,:,pred_i,2)', 'grouped', 'BarWidth', 0.8);
    b(1).FaceColor = col_two(1,:);
    b(2).FaceColor = col_two(2,:);
    b(1).EdgeColor = 'none';
    b(2).EdgeColor = 'none';
    yl = [min(floor(min(w_les_r(:,:,pred_i,2),[],'all')*20)/20, 0) ceil(max(w_les_r(:,:,pred_i,2),[],'all')*10)/10];
    set(gca, 'LineWidth', 1.5, 'XLim', [0.5 2.5], 'YLim', yl, 'XTick', [], ...
        'TickLength', [0.03 0.03], 'Tickdir', 'out', 'Box', 'off', 'FontSize', 16);
    set(gcf, 'Position', [1   734   201   261], 'color', 'w');
    savename = fullfile(figdir, sprintf('MetaVCI_lesion_conn_corr_weight_les_%s_%s_nestcv_optall.pdf', norm_conn, targtest{pred_i}));
    pagesetup(gcf); saveas(gcf, savename); close all;

end