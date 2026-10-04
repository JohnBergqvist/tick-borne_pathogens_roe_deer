# tick-borne_pathgoens_roe_deer
This repository consists of the data of each individual tick analysed in the RåFäst project that was detached from roe deer in the Örebro area, Sweden. 

Each tick has information on the estimated feeding time, whether they are positive for Borrelia spp. and/or Babesia spp., qPCR quantification of each pathogen, and pathgeon species identity. 

The script performs statistical analyses on the the effect of feeding time on the prevalence and quantification of the pathogens. 

To test if there are host-specific confounders in that specific roe deer individuals could affect the prevalence, we performed both Generalised Estimating Equation (GEE), that adjusts the standard errors of our population-level estimate to account for the fact that ticks on the same deer are not independent, and Generalised Linear Mixed Model (GLMM) explicitly models the variance caused by the host. The `(1 | roe_deer_id)` syntax calculates a "random intercept", meaning it allows the baseline probability of the pathogen to shift up or down for each individual deer. To perform the GLMM in a python script, we execute raw R code via rpy2 to access R's `lme4` package. 

To test if there are host-specific confounders that could affect the quantification of the pathogens, we performed Linear Mixed Model (LMM) with roe deer id as a random intercept and a Conjugate Gradient (CG) optimiser to successfully converge the data. The default LMM optimiser did not converge the data, likely due to small sample size. 

