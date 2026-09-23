# LNCM — Lesion-Network Contrast Mapping

Analysis code for **"Dissociating post-stroke cognitive deficits via lesion-network contrast mapping and high-dimensional network modeling"** (Lim et al., under review).

Conventional lesion-network mapping thresholds each patient's lesion-connectivity map and overlaps the results, which tends to converge on the same functional hubs regardless of which cognitive domain is studied. This repository implements three analyses that address that problem in a cohort of 1,431 patients with acute ischemic stroke from the Hallym and Bundang VCI cohorts (Meta VCI Map consortium):

1. **Lesion-network overlap** — the conventional approach, reproduced here to characterize its hub convergence, together with virtual lesion simulations that show the convergence is a property of the method rather than of the data.
2. **Lesion-network contrast mapping (LNCM)** — a voxel-wise contrast of *unthresholded* lesion-connectivity maps between patients impaired and unimpaired in a given domain.
3. **Ridge principal-component regression** — prediction of continuous domain scores from high-dimensional lesion-connectivity features, with nested cross-validation over the number of components and the ridge penalty.

## Requirements

| | |
|---|---|
| MATLAB | R2023b, with the Statistics and Machine Learning Toolbox (the Parallel Computing Toolbox is optional; without it `parfor` runs serially) |
| [CanlabCore](https://github.com/canlab/CanlabCore) | `fmri_data`, `region`, display and statistics utilities |
| [cocoanCORE](https://github.com/cocoanlab/cocoanCORE) | brain display (`brain_activations_display_jj`), figure export (`pagesetup`) and standard-space images |
| [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) | image I/O and rendering (used via CanlabCore) |

Standard-space images are resolved through the MATLAB path (`which`): `brainmask_canlab_2mm.nii`, `Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii`, `Schaefer_Net_Labels_r100.mat` and `Yeo_10networks_2mm.nii` from cocoanCORE; `keuken_2014_enhanced_for_underlay.img` and `surf_spm2_hipp.mat` from CanlabCore.

### Inputs prepared upstream

This repository begins at the point where the imaging inputs are already in standard space. Lesion masks were segmented and registered to MNI-152 space by the Meta VCI Map consortium using [RegLSM](https://github.com/Meta-VCI-Map/RegLSM), which calls elastix v4.8 and SPM12; the registered masks were then resampled with FSL to 2 mm and concatenated into one image per cohort. The 3 mm brain mask and the 3 mm normative images used for the voxel-wise connectome were likewise prepared beforehand. Those preparation steps are not included here.

### Paths

`set_path_env.m` and the `basedir` / `datdir` / `figdir` / `imgdir` assignments at the head of each script contain absolute paths specific to the machine the analyses were run on. **Edit these before running anything.** Scripts write statistical maps (`.nii`) and fitted models (`.mat`) to `datdir`, where later scripts read them back, and figures to `figdir`. The scripts expect:

```
<datdir>/
  Lesion_Maps/
    {Hallym,Bundang}_lesion_2mm_merged.nii            # lesion masks, MNI-152, 2 mm, one file per cohort
    {Hallym,Bundang}_lesion_2mm_merged_subject_id.mat # subject order within each merged image
  domain_scores_calculated.csv                        # cognitive domain z scores, one row per patient
  HCA_LS_2.0_subject_completeness.csv                 # HCP-Aging subject list
  brainmask_canlab_3mm.nii, colormap_jj.mat
  {GSP,HCPA}_lesion_conn_group_*.mat                  # written by the connectome scripts
  GSP_voxel_conn_group_*.mat, GSP_atlas_*.mat         # written by the connectome scripts
  MetaVCI_lesion_conn_contrast_*.nii                  # LNCM maps, written by the univariate scripts
  {GSP,HCPA}_lesion_conn_group_predict_model_*.mat    # written by the prediction scripts
```

Normative resting-state data are read from `imgdir`: the Brain Genomics Superstruct Project (GSP1000; https://doi.org/10.7910/DVN/ILXIKS) for the main analyses and HCP-Aging (Lifespan 2.0 release, available through the NIMH Data Archive under a data use agreement) for the sensitivity analysis. Neither is redistributed here.

## Data availability

Individual-level patient data are not included in this repository. The data come from the health records of clinic patients who have not consented to data sharing and are therefore not readily sharable.

## Pipeline

Run in order; each stage depends on the outputs of the one above.

**1 — Build normative connectomes** (long-running)

```
MetaVCI_GSP_lesion_conn_group_z.m      # main: lesion-seed connectivity, GSP1000
MetaVCI_GSP_get_atlas_conn.m           # Schaefer-100 atlas connectome, for the simulations
MetaVCI_GSP_voxel_conn_group_z.m       # voxel-wise connectome at 3 mm, for hub degree
MetaVCI_HCPA_lesion_conn_group_z.m     # sensitivity analysis: HCP-Aging in place of GSP
```

The GSP data are used as distributed, in preprocessed form. The HCP-Aging ICA-FIX runs are processed in `MetaVCI_HCPA_lesion_conn_group_z.m` to match them: 6 mm FWHM smoothing, 0.009–0.08 Hz band-pass filtering and global signal regression.

**2 — Analyses**

```
MetaVCI_lesion_symp_group_univariate.m       # voxel-wise lesion-symptom mapping
MetaVCI_lesion_conn_group_univariate.m       # lesion-network overlap and LNCM contrast
MetaVCI_voxel_conn_group_univariate.m        # hub degree of the normative connectome
MetaVCI_simulate_lesion_overlap.m            # virtual lesion simulations
MetaVCI_lesion_conn_group_predict_nestcv.m   # ridge-PCR with nested cross-validation (parfor)
MetaVCI_Yeo7network.m                        # Yeo seven-network reference figure
```

**3 — Sensitivity analyses**

The older-adult connectome analysis is produced by re-running `MetaVCI_lesion_conn_group_univariate.m` and `MetaVCI_lesion_conn_group_predict_nestcv.m` with `norm_conn = 'HCPA'` in place of `'GSP'` (the scripts are set to GSP as deposited). `MetaVCI_map_similarity.m` requires the outputs of both runs.

```
MetaVCI_lesion_conn_group_univariate_cadj.m       # LNCM adjusted for assessment interval x site
MetaVCI_lesion_conn_group_predict_nestcv_cadj.m   # ridge-PCR with the same covariates
MetaVCI_map_similarity.m                          # spatial r across map variants; tabulates saved R2
```

The covariate-adjusted scripts fit the covariate block unpenalized, profiling it out of the ridge fit within each cross-validation fold, so that only the lesion-connectivity features are shrunk.

## Which script produces which result

| Script | Output |
|---|---|
| `MetaVCI_lesion_symp_group_univariate.m` | Figure 2 |
| `MetaVCI_lesion_conn_group_univariate.m` | Figures 3A–B and 4; Supplementary Figures 3 and 4 |
| `MetaVCI_voxel_conn_group_univariate.m` | Figure 3C |
| `MetaVCI_Yeo7network.m` | Figure 3D |
| `MetaVCI_simulate_lesion_overlap.m` | Figures 3E–F; Supplementary Figure 1 |
| `MetaVCI_lesion_conn_group_predict_nestcv.m` | Figures 5 and 6 |
| `MetaVCI_lesion_conn_group_univariate_cadj.m` | covariate-adjusted LNCM maps (sensitivity analysis) |
| `MetaVCI_lesion_conn_group_predict_nestcv_cadj.m` | covariate-adjusted prediction accuracy; domain scores by cohort and assessment interval (sensitivity analysis) |
| `MetaVCI_lesion_conn_group_univariate.m`, `MetaVCI_lesion_conn_group_predict_nestcv.m` with `norm_conn = 'HCPA'` | HCP-Aging LNCM maps and prediction accuracy (sensitivity analysis) |
| `MetaVCI_map_similarity.m` | spatial correlations between the main and sensitivity-analysis maps; tabulates the saved GSP vs HCP-Aging R² |

Figure 1 is a schematic and has no code. Supplementary Figure 2 (cumulative lesion maps) is a lesion-frequency overview produced by the Meta VCI Map consortium.

`boot2cip.m` converts bootstrap distributions to confidence intervals and two-tailed p-values and is called by the analysis scripts.

## Citation

```
Lim J-S, Lee J-J, Kim GH, Kim BJ, Kim JY, Kim DY, Lee M, Lee B-C, Woo C-W,
Oh MS, Yu K-H, Biessels G, Biesbroek JM, Weaver N, Bae H-J.
Dissociating post-stroke cognitive deficits via lesion-network contrast
mapping and high-dimensional network modeling. Under review.
```

Co-corresponding authors: Hee-Joon Bae (braindoc@snu.ac.kr) and Choong-Wan Woo (waniwoo@skku.edu).


## License

MIT — see [LICENSE](LICENSE).
