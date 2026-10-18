# Tick-Borne Pathogen Prevalence and Quantification Analysis (*Borrelia* spp. and *Babesia* spp.)

This repository consists of the data of each individual tick analysed in the RåFäst project that was detached from roe deer in the Örebro area, Sweden, and the statistical analysis pipeline accompanying the manuscript:

**Pathogen-specific associations between feeding duration and detection of Borrelia and Babesia in Ixodes ricinus collected from roe deer**
*Authors:* 

[Peter Wilhelmsson 1,2*†

John Bergqvist 1,5,6*

Petter Kjellander 3

Hans Fredlund 4

Madeleine Christensson 3

Per-Eric Lindgren 1,2

†Corresponding author

*These authors contributed equally to this work.

1 Division of Inflammation and Infection, Department of Biomedical and Clinical Sciences, Linköping University, Linköping, Sweden

2 Division of Clinical Microbiology, Department of Laboratory Medicine, Region Jönköping County, Jönköping, Sweden

3 Grimö Wildlife Research Station, Department of Ecology, Swedish University of Agricultural Sciences, Riddarhyttan, Sweden

4 Department of Laboratory Medicine, Faculty of Medicine and Health, Örebro University, Örebro, Sweden

5 Present address: MRC Laboratory of Medical Sciences, London, UK 

6 Present address: Institute of Clinical Sciences, Imperial College London, London, UK]


## Overview
The script (`tick_pathogen_analysis.py`) performs data quality control, filtering, and statistical modeling to evaluate the relationship between *Ixodes ricinus* nymphal blood-feeding duration and pathogen prevalence/burden while accounting for host-level clustering across individual roe deer (*Capreolus capreolus*):
1. **Pathogen Prevalence:** Univariable logistic regression (`statsmodels`) and binomial Generalized Linear Mixed Models (`lme4::glmer` via `rpy2`) with roe deer identity as a random intercept and latent-scale ICC calculation.
2. **Pathogen Quantification:** Ordinary Least Squares (OLS) linear regression on $\log_{10}(x+1)$-transformed gene copy numbers, Breusch-Pagan homoscedasticity diagnostics, Spearman's rank-order correlation (`SciPy`), and Linear Mixed Models (`statsmodels` REML with L-BFGS optimization).
3. **Co-infection Analysis:** Uncorrected Pearson's $\chi^2$ test of independence and Fisher's exact test (`SciPy`).

## Environment & Dependencies
* **Python:** 3.11.16 (`pandas` 3.0.6, `numpy` 2.4.6, `statsmodels` 0.15.0, `scipy` 1.17.1, `rpy2` 3.6.8, `matplotlib` 3.11.2, `seaborn` 0.13.2)
* **R:** 4.5.1 (`lme4` 2.0.6)

