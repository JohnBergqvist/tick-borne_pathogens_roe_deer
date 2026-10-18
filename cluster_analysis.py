'''
(C) John Bergqvist 2026

Affiliation: 
Division of Inflammation and Infection, Department of Biomedical and Clinical Sciences, Linköping University, Linköping, Sweden
MRC Laboratory of Medical Sciences, London, UK
Institute of Clinical Sciences, Imperial College London, London, UK

This script is designed to analyse the pathogen prevalence in ticks, and the quantification of the pathogen load, taking into account the roe deer ID that the ticks were taken from. 
The analysis is performed to test if there are host-specific confounders affecting the accosiation between feeding time and pathogen prevlanece. 

The statistical models used for analysing the prevalence of pathogens in ticks are the simple logistic regression analysis and the Generalised Linear Mixed Model (GLMM).
The statistical models used for analysing the log10 tranformed data quantification of pathogens in ticks is OLS linear regression, Spearman's rank correlation, and the Linear Mixed Model (LMM)  with L-BFGS optimisation to converge.

'''

#%%
# import necessary libraries
from pathlib import Path
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from scipy import stats as sp_stats


# rpy2 imports for bridging Python and R directly - ignore the 'Error importing in API mode, Trying to import in ABI mode'. 
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter



#%%

# Directories 
# define the HOME_DIR as the place where the scirpt is located, and print it to the console
print(f'HOME_DIR: {Path(__file__).resolve().parent}')
HOME_DIR = Path(__file__).resolve().parent

borrelia_data_dir = HOME_DIR / 'borrelia_data.csv'
babesia_data_dir = HOME_DIR / 'babesia_data.csv'
roedeer_dir = HOME_DIR / 'roedeer_info.csv'

# load the data
borrelia_df = pd.read_csv(borrelia_data_dir)
babesia_df = pd.read_csv(babesia_data_dir)
roedeer_df = pd.read_csv(roedeer_dir)


####### ========== DATA QUALITY CONTROL CHECKS ==========
print(f'========== DATA QUALITY CONTROL CHECKS ==========')

# Make a new column in both dfs called 'Test tube marking' that is the all characters before the second '-' of the column 'ID.nr.'
borrelia_df['Test tube marking'] = borrelia_df['ID.nr.'].str.split('-').str[:2].apply('-'.join)
babesia_df['Test tube marking'] = babesia_df['ID.nr.'].str.split('-').str[:2].apply('-'.join)

# Merge the column 'Collected from animal ID.1' from the roedeer_df into both borrelia_df and babesia_df based on the 'Test tube marking' column
borrelia_df = borrelia_df.merge(roedeer_df[['Test tube marking', 'Collected from animal ID.1']], on='Test tube marking', how='left')
babesia_df = babesia_df.merge(roedeer_df[['Test tube marking', 'Collected from animal ID.1']], on='Test tube marking', how='left')

# Rename the column 'Collected from animal ID.1' to 'roe_deer_id' in both dataframes
borrelia_df.rename(columns={'Collected from animal ID.1': 'roe_deer_id'}, inplace=True)
babesia_df.rename(columns={'Collected from animal ID.1': 'roe_deer_id'}, inplace=True)

# Ensure that the same tick 'ID.nr' values are present in both dataframes
borrelia_df = borrelia_df.sort_values(by='ID.nr.').reset_index(drop=True)
babesia_df = babesia_df.sort_values(by='ID.nr.').reset_index(drop=True)

# Check that the 'ID.nr' column contains the same values in both dataframes, and print the values that are different if there are any differences
for df1, df2, name1, name2 in zip([borrelia_df], [babesia_df], ['Borrelia'], ['Babesia']):
    if not df1['ID.nr.'].equals(df2['ID.nr.']):
        print(f'The ID.nr. columns in the {name1} and {name2} dataframes are not equal.')
        print(f'Values in {name1} but not in {name2}: {set(df1["ID.nr."]) - set(df2["ID.nr."])}')
        print(f'Values in {name2} but not in {name1}: {set(df2["ID.nr."]) - set(df1["ID.nr."])}')
    else:
        print(f'The ID.nr. columns in the {name1} and {name2} dataframes are equal.')

# Checks to ensure the two dataframes contain the same roe deer and tick information, and that the only difference is the pathogen prevalence information.
# This loop iterates through the two dataframes, and for each dataframe, prints the length of the dataframe, 
#   the number of unique roe deer IDs, and the number of unique roe deer IDs that have the 'Developmental stage' column as 'A' (adult), 'N' (nymph), and 'L' (larva)

for df, name in zip([borrelia_df, babesia_df], ['Borrelia', 'Babesia']):
    print(f'================== {name} dataframe information =================')
    print(f'Length of the {name} dataframe: {len(df)}')
    print(f'Number of ticks with feeding time determined (not "?" or NaN): {df[df["feeding_time"].isin(["?", np.nan]) == False].shape[0]}')
    print(f'Number of unique roe deer IDs: {df["roe_deer_id"].nunique()}')
    # make a list of unique roe deer IDs that have the 'Developmental stage' column as 'A' (adult)
    adult_roe_deer_ids = df[df['developmental_stage'] == 'A']
    print(f'Number of unique roe deer IDs with adult ticks: {adult_roe_deer_ids[adult_roe_deer_ids["developmental_stage"] == "A"]["roe_deer_id"].nunique()}')
    # make a list of unique roe deer IDs that have the 'Developmental stage' column as 'N' (nymph)
    nymph_roe_deer_ids = df[df['developmental_stage'] == 'N']
    print(f'Number of unique roe deer IDs with nymph ticks: {nymph_roe_deer_ids[nymph_roe_deer_ids["developmental_stage"] == "N"]["roe_deer_id"].nunique()}')
    # make a list of unique roe deer IDs that have the 'Developmental stage' column as 'L' (larva)
    larva_roe_deer_ids = df[df['developmental_stage'] == 'L']
    print(f'Number of unique roe deer IDs with larva ticks: {larva_roe_deer_ids[larva_roe_deer_ids["developmental_stage"] == "L"]["roe_deer_id"].nunique()}')
    # print the number and the unique roe_deer_ids that overlap between the adult and nymph ticks
    overlap_roe_deer_ids = set(adult_roe_deer_ids['roe_deer_id']).intersection(set(nymph_roe_deer_ids['roe_deer_id']))
    print(f'Number of unique roe deer IDs with both adult and nymph ticks: {len(overlap_roe_deer_ids)}')
    print(f'Unique roe deer IDs with both adult and nymph ticks: {overlap_roe_deer_ids}')

    # print the number and the unique roe_deer_ids that overlap between the nymph and larva ticks
    overlap_roe_deer_ids = set(nymph_roe_deer_ids['roe_deer_id']).intersection(set(larva_roe_deer_ids['roe_deer_id']))
    print(f'Number of unique roe deer IDs with both nymph and larva ticks: {len(overlap_roe_deer_ids)}')
    print(f'Unique roe deer IDs with both nymph and larva ticks: {overlap_roe_deer_ids}')
    # print the number of unique row deer IDs that have a '?' or 'NaN' in 'feeding_time' column
    print(f'Number of unique tick ID "ID.nr." with unknown feeding time: {df[df["feeding_time"].isin(["?", np.nan])]["ID.nr."].nunique()}')
    # change the 'pathogen_status' column to 'positive' for '+' and 'negative' for blank spaces in both dataframes
    print(f'================== {name} pathogen status =================')
    df['pathogen_status'] = np.where(df['pathogen_status'] == '+', 'positive', 'negative')
    print(f'Number of positive ticks in the {name} dataframe: {df[df["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe: {df[df["pathogen_status"] == "negative"].shape[0]}')

        # save the filtered dataframe to a new variable
    if name == 'Borrelia':
        borrelia_df = df
    else:
        babesia_df = df


#%%

###============================ PATHOGEN PREVALENCE ANALYSIS ============================###
# This analsis is only performed on the nymphs.

# define the colors to be used in the plots - grey for negative ticks and red for positive ticks
colors = {'negative': 'grey', 'positive': 'red'}

# 1. A loop that filters the two dataframes to only contain rows with the letter 'N' in the 'developmental_stage' column (nymphs),
## remove the rows with a '?' in the 'feeding_time' column (undetermined feeding times), and remove the rows with missing values 
## in the 'feeding_time' column (undetermined feeding times).
borrelia_df_filtered = borrelia_df.copy()
babesia_df_filtered = babesia_df.copy()

for df, name in zip([borrelia_df_filtered, babesia_df_filtered], ['Borrelia', 'Babesia']):
    print(f'================== {name} Key Values For Prevalence Analysis=================')
    print(f'Number of ticks in the {name} dataframe before filtering: {len(df)}')
    df_filtered = df[df['developmental_stage'] == 'N'] # keep only nymphs
    df_filtered = df_filtered[df_filtered['feeding_time'] != '?'] # remove rows with undetermined feeding times
    df_filtered = df_filtered.dropna(subset=['feeding_time']) # remove rows with missing values in feeding time (undetermined feeding times)
    print(f'Number of ticks in the {name} dataframe after filtering (only nymps, no unknown feeding times): {len(df_filtered)}')
    print(f'Number of ticks removed during filtering: {len(df) - len(df_filtered)}')
    print(f'Number of unique roe deer IDs in the {name} dataframe before filtering: {df["roe_deer_id"].nunique()}')
    print(f'Number of unique roe deer IDs in the {name} dataframe after filtering: {df_filtered["roe_deer_id"].nunique()}')
    print(f'Number of roe deers removed during filtering: {df["roe_deer_id"].nunique() - df_filtered["roe_deer_id"].nunique()}')
    print(f'Number of ticks with {name} detected before filtering: {df[df["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of nymph ticks with {name} without determined feeding time (before filtering): {df[(df["pathogen_status"] == "positive") & (df["developmental_stage"] == "N") & (df["feeding_time"].isin(["?", np.nan]))].shape[0]}')
    print(f'Number of ticks with {name} detected after filtering: {df_filtered[df_filtered["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of adult ticks with {name} detected before filtering: {df[(df["pathogen_status"] == "positive") & (df["developmental_stage"] == "A")].shape[0]}')
    print(f'Number of unique roe deer IDs with {name} detected after filtering: {df_filtered[df_filtered["pathogen_status"] == "positive"]["roe_deer_id"].nunique()}')
    
    # 2. Transform the 'pathgoen_status' column to a binary variable, where 'positive' is 1 and 'negative' is 0 for both dataframes
    df_filtered['pathogen_status'] = df_filtered['pathogen_status'].map({'positive': 1, 'negative': 0})
    print(f'================== {name} pathogen status after transformation (1=positive, 0=negative) =================')
    print(f'Check the unique values in the {name} dataframe after transformation: {df_filtered["pathogen_status"].unique()}')
    print(f'Number of positive ticks in the {name} dataframe after transformation: {df_filtered[df_filtered["pathogen_status"] == 1].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe after transformation: {df_filtered[df_filtered["pathogen_status"] == 0].shape[0]}')
    
    
    # save the filtered dataframe to a new variable
    if name == 'Borrelia':
        borrelia_df_filtered = df_filtered
    else:
        babesia_df_filtered = df_filtered



# 3. Perform Simple logistic regression for both dataframes to assess the association between feeding time and pathogen prevalence, while accounting for the clustering effect of roe deer ID.
    print(f'================== {name} Simple logistic regression =================')
    
    # Fit standard, unpenalized logistic regression
    model = smf.logit('pathogen_status ~ feeding_time', data=df_filtered).fit()
    
    # 1. Print the full summary table (includes coef, SE, z, P>|z|, and 95% CIs)
    print(model.summary())
    
    # 2. Extract specific values directly
    intercept = model.params['Intercept']
    intercept_se = model.bse['Intercept']
    slope = model.params['feeding_time']
    slope_se = model.bse['feeding_time']
    p_value = model.pvalues['feeding_time']
    
    print(f"\nIntercept: {intercept:.4f} (SE = {intercept_se:.4f})")
    print(f"Slope (feeding_time): {slope:.4f} (SE = {slope_se:.4f}, p = {p_value:.4e})\n")
    
    # 3. Generate predictions for the logistic curve
    feeding_time_range = np.linspace(df_filtered['feeding_time'].min(), df_filtered['feeding_time'].max(), 100)
    pred_df = pd.DataFrame({'feeding_time': feeding_time_range})
    predicted_probabilities = model.predict(pred_df)
    
    # 4. Initialize plot and draw observed data points
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='feeding_time',
        y='pathogen_status',
        data=df_filtered,
        hue='pathogen_status',
        palette={0: colors['negative'], 1: colors['positive']},
        alpha=0.6
    )
    # Plot the logistic regression curve
    plt.plot(feeding_time_range, predicted_probabilities, color='black', linewidth=2, label='Logistic Regression Curve')
    plt.title(f'Logistic Regression of {name} Prevalence vs Feeding Time', fontsize=18)
    plt.xlabel('Feeding Time (h)', fontsize=16)
    plt.ylabel('Probability of Positive Tick', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # define the position of the legend to be in the center right corner of the plot
    plt.legend(title='', loc='center right')
    plt.tight_layout()
    plt.show()
    plt.close()

# 4. Plot the number of positive and negative ticks for each roe deer ID in a stacked bar chart for both dataframes
    print(f'================== {name} stacked bar chart =================')
    # count the number of positive and negative ticks for each roe deer ID
    tick_counts_by_deer = df_filtered.groupby(['roe_deer_id', 'pathogen_status']).size().unstack(fill_value=0).reset_index()
    tick_counts_by_deer.columns = ['roe_deer_id', 'negative', 'positive']
    # plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    tick_counts_by_deer.set_index('roe_deer_id')[['negative', 'positive']].plot(kind='bar', stacked=True, ax=plt.gca(), color=[colors['negative'], colors['positive']])
    plt.title(f'Count of Positive and Negative Ticks by Roe Deer ID - {name}', fontsize=18)
    plt.xlabel('Roe Deer ID', fontsize=16)
    plt.ylabel('Count of Ticks', fontsize=16)
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close()



#%%
#### ============================ CLUSTER ANALYSIS ============================ ####

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================

# Explicitly cast data types to ensure the models treat them correctly:
# - Predictor must be numeric for a continuous effect.
# - Cluster ID must be a string/category so the model treats it as discrete groups,
#   not a continuous mathematical variable.

for df, name in zip([borrelia_df_filtered, babesia_df_filtered], ['Borrelia', 'Babesia']):
    print(f'================== {name} data preparation =================')
    df['feeding_time'] = pd.to_numeric(df['feeding_time'])
    df['roe_deer_id'] = df['roe_deer_id'].astype(str)



# ==============================================================================
# 2. GENERALISED LINEAR MIXED MODEL (GLMM)
# ==============================================================================
    print("\n======================================================")
    print(f" 2. GENERALISED LINEAR MIXED MODEL (GLMM) - pure rpy2 bridge - {name}")
    print("======================================================")

# WHY WE DO THIS: Unlike the GEE, the GLMM explicitly models the variance caused 
# by the host. The `(1 | roe_deer_id)` syntax calculates a "random intercept", 
# meaning it allows the baseline probability of pathogen to shift up or down 
# for each individual deer.
#
# WHY RPY2: Python's native ecosystem struggles with Generalised (logistic) mixed 
# models. By executing raw R code via rpy2, we access R's `lme4` package 
# and completely bypass translation bugs associated with wrapper libraries.

# Step A: Create a context manager that translates Pandas dataframes and their 
# underlying NumPy arrays into native R dataframes.
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        # Push the Python dataframe directly into R's global memory environment
        ro.globalenv['r_df'] = ro.conversion.py2rpy(df)

# Step B: Write the exact R script as a raw text string.
# Because this is evaluated natively inside R, it avoids any Python-to-R 
# class inheritance bugs regarding the `family = binomial()` argument.
    r_script = """
    # Load the mixed modeling library natively in R
    library(lme4)

    # Fit the logistic mixed model
    glmm_model <- glmer(pathogen_status ~ feeding_time + (1 | roe_deer_id), 
                        data = r_df, 
                        family = binomial(link="logit"))

    # Return the standard R summary output
    summary(glmm_model)
    """

    # Step C: Execute the raw R script and print the result back into Python
    glmm_result = ro.r(r_script)
    print(f"GLMM Result - {name}: {glmm_result}")
    
    # Extract random effect variance using VarCorr() and calculate latent-scale ICC, which is the proportion of variance explained by the random effect (roe_deer_id)
    icc_script = """
    re_var <- as.numeric(VarCorr(glmm_model)$roe_deer_id)
    icc_val <- re_var / (re_var + (pi^2 / 3))
    icc_val
    """
    icc = ro.r(icc_script)[0]  # Extract the float from the R vector
    print(f"ICC - {name}: {icc:.4f}")






# %%

#####==================== PATHOGEN QUANTIFICATION ANALYSIS====================#####

# Checking the range of quantification values for both pathogens, and the number of positive and negative ticks in each dataframe.
q_borrelia_df = borrelia_df.copy()
q_babesia_df = babesia_df.copy()

for q_df, name in zip([q_borrelia_df, q_babesia_df], ['Borrelia', 'Babesia']):
    #q_df['pathogen_status'] = np.where(q_df['pathogen_status'] == '+', 'positive', 'negative')
    print(f'Number of positive ticks in the {name} dataframe: {q_df[q_df["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe: {q_df[q_df["pathogen_status"] == "negative"].shape[0]}')
    # keep only the positive ticks to analyse the quantity of pathogen in the ticks
    q_df = q_df[q_df['pathogen_status'] == 'positive']
    print(f'Before any filtering, the {name} dataframe has {len(q_df)} rows ({name}-positive ticks).')
    print(f"Range of quantification values for {name}: min={q_df['quantification'].min():.2e}, max={q_df['quantification'].max():.2e}, median={q_df['quantification'].median():.2e}")
    adults = q_df[q_df['developmental_stage'] == 'A']
    print(f'Range of quantification values for {name} in adult ticks: number of observations={len(adults)}, min={adults["quantification"].min():.2e}, max={adults["quantification"].max():.2e}, median={adults["quantification"].median():.2e}')
    nymphs = q_df[q_df['developmental_stage'] == 'N']
    print(f'Range of quantification values for {name} in nymph ticks: number of observations={len(nymphs)}, min={nymphs["quantification"].min():.2e}, max={nymphs["quantification"].max():.2e}, median={nymphs["quantification"].median():.2e}')
    if name == 'Borrelia':
        # remove the 'B. miyamotoi' positive ticks from the quantification dataset of borrelia
        nymphs_no_Bm = nymphs[nymphs['borrelia_species'] != 'B. miyamotoi']
        print(f'After removing B. miyamotoi positive ticks, the {name} dataframe has {len(nymphs_no_Bm)} rows ({name}-positive ticks).')
        print(f"Range of quantification values for {name} after removing B. miyamotoi: min={nymphs_no_Bm['quantification'].min():.2e}, max={nymphs_no_Bm['quantification'].max():.2e}, median={nymphs_no_Bm['quantification'].median():.2e}")
    else:
        continue

# Re-load the dataframes 
q_borrelia_df = borrelia_df.copy()
q_babesia_df = babesia_df.copy()
## Remove 'B. miyamotoi' positive ticks from the quantification dataset of borrelia
q_borrelia_df = q_borrelia_df[q_borrelia_df['borrelia_species'] != 'B. miyamotoi']

for q_df, name in zip([q_borrelia_df, q_babesia_df], ['Borrelia', 'Babesia']):
    print(f'================== {name} quantification analysis =================')
    print(f'Length of the dataframe before filtering: {len(q_df)}')
    q_df = q_df[q_df['developmental_stage'].isin(['N'])]
    print(f'Length of the dataframe after filtering for developmental stage: {len(q_df)}')
    q_df = q_df[q_df['feeding_time'] != '?']
    print(f'Length of the dataframe after filtering for feeding time: {len(q_df)}')
    q_df = q_df.dropna(subset=['feeding_time'])
    print(f'Length of the dataframe after removing missing values in feeding time: {len(q_df)}')
    # transform the 'feeding_time' column to numeric
    q_df['feeding_time'] = pd.to_numeric(q_df['feeding_time'])
    # Ensure the 'roe_deer_id' column is treated as a string (categorical variable)
    q_df['roe_deer_id'] = q_df['roe_deer_id'].astype(str)
    # transformt the 'pathogen_status' column to 'positive' for '+' and 'negative' for blank spaces
    #q_df['pathogen_status'] = np.where(q_df['pathogen_status'] == '+', 'positive', 'negative')
    print(f'Number of positive ticks in the {name} dataframe: {q_df[q_df["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe: {q_df[q_df["pathogen_status"] == "negative"].shape[0]}')
    # keep only the positive ticks to analyse the quantity of pathogen in the ticks
    q_df = q_df[q_df['pathogen_status'] == 'positive']
    print(f'Length of the dataframe with only positive ticks: {len(q_df)}')
    # transform the 'quantification' column to numeric
    q_df['quantification'] = pd.to_numeric(q_df['quantification'], errors='coerce')
    # transform the 'quantification' column to log10 scale, and add a small constant to avoid log(0)
    q_df['log_quantification_per_tick'] = np.log10(q_df['quantification'] + 1)

    # apply the edits to the df
    if name == 'Borrelia':
        q_df_borrelia = q_df.copy()
    else:
        q_df_babesia = q_df.copy()

### ====================== Without clustering of roe deer ID====================== ###
    print("\n======================================================")
    print(f" LINEAR REGRESSION (OLS) - statsmodels - {name} ")
    print("======================================================")
    # Fit standard, unpenalized linear regression
    model = smf.ols('log_quantification_per_tick ~ feeding_time', data=q_df).fit()
    
    # 1. Print the full summary table (includes coef, SE, t, P>|t|, and 95% CIs)
    print(model.summary())
    
    # 2. Extract specific values directly:
    intercept = model.params['Intercept']
    intercept_se = model.bse['Intercept']
    slope = model.params['feeding_time']
    slope_se = model.bse['feeding_time']
    p_value = model.pvalues['feeding_time']

    print(f"\nIntercept: {intercept:.4f} (SE = {intercept_se:.4f})")
    print(f"Slope (feeding_time): {slope:.4f} (SE = {slope_se:.4f}, p = {p_value:.4e})\n")
    from statsmodels.stats.diagnostic import het_breuschpagan

    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print(
         f'Breusch-Pagan p-value: {bp_test[1]:.4f}'
    )  # p > 0.05 means constant variance holds!
    

    # Also perform Spearman’s Rank-Order Correlation on this data to assess the monotonic relationship between feeding time and pathogen load in ticks, without assuming a linear relationship.
    # Spearman Rank-Order Correlation (via scipy.stats)
    print("\n======================================================")
    print(f" SPEARMAN RANK-ORDER CORRELATION - statsmodels - {name} ")
    print("======================================================")
    rho, spearman_p = spearmanr(
        q_df['feeding_time'], q_df['log_quantification_per_tick']
    )
    print(
        f"Spearman's rank correlation (rho): {rho:.4f} (p = {spearman_p:.4e})\n"
    )

### ====================== With clustering of roe deer ID====================== ###

    # Perform the Linear Mixed Model (LMM) analysis using statsmodels for the log-transformed quantification of 
    # the pathogen in ticks, while accounting for the clustering effect of roe deer ID.
    print("\n======================================================")
    print(f" LINEAR MIXED MODEL (LMM) - statsmodels - {name} ")
    print("======================================================")
    # Fit the Linear Mixed Model
    # smf.mixedlm automatically applies a linear (Gaussian) family 
    # and sets the random intercept based on the 'groups' argument.
    lmm_model = smf.mixedlm("log_quantification_per_tick ~ feeding_time", 
                            q_df, 
                            groups=q_df["roe_deer_id"])
    lmm_result = lmm_model.fit(method='lbfgs')  # Use L-BFGS optimizer for convergence
    print(f"LMM Result with L-BFGS optimisation - {name}:")
    print(lmm_result.summary())


# The default lmm optimiser did not converge the data, likely due to small sample sizes.
# The L-BFGS optimiser successfully converged both pathogen data, but the overall output remains the same - 
# we cannot determine a significant effect of feeding time on the quantity of either pathogen in ticks, likely due to the small sample size and high variability in the data.



# =============== Plotting the relationship between feeding time and log10-transformed quantification of pathogens in ticks ===============
plt.figure(figsize=(6, 4))
sns.regplot(data=q_df_borrelia, x='feeding_time', y='log_quantification_per_tick', ci=95)
plt.title(f"Borrelia Load vs Feeding Time (N={len(q_df_borrelia)})")
plt.ylabel("Log10 Quantification per Tick", fontsize=12)
plt.xlabel("Feeding Time (hours)", fontsize=12)
plt.show()

# The 'ÖCT90-14-47' ID.nr. is an outlier at feeding time 92hrs. Removing this datapoint brings the p-value of the LMM from 0.028 to 0.052 for Babesia spp.
#q_df_babesia = q_df_babesia[q_df_babesia['ID.nr.'] != 'ÖCT90-14-47']

plt.figure(figsize=(6, 4))
sns.regplot(data=q_df_babesia, x='feeding_time', y='log_quantification_per_tick', ci=95)
plt.title(f"Babesia Load vs Feeding Time (N={len(q_df_babesia)})")
plt.ylabel("Log10 Quantification per Tick", fontsize=12)
plt.xlabel("Feeding Time (hours)", fontsize=12)
plt.show()


# %%
# ============== CHI SQUARE TEST FOR THE CO-OCCURRENCE OF BORRELIA AND BABESIA IN THE TICKS ==============


# load the data fresh without any of the filtering applied for the analyses above, as we want to assess the co-occurrence of the two pathogens in all ticks,
# irrespective of tick stage and feeding time.

borrelia_chi_df = borrelia_df.copy()
babesia_chi_df = babesia_df.copy()


# create a suffix on the 'pathogen_staus' column in both dataframes to distinguish between the two pathogens
borrelia_chi_df.rename(columns={'pathogen_status': 'pathogen_status_borrelia'}, inplace=True)
babesia_chi_df.rename(columns={'pathogen_status': 'pathogen_status_babesia'}, inplace=True)

# merge the two dataframes on the 'ID.nr', keeping only 'ID.nr.', 'pathogen_status_borrelia', and 'pathogen_status_babesia' columns
merged_df = pd.merge(borrelia_chi_df[['ID.nr.', 'pathogen_status_borrelia']], babesia_chi_df[['ID.nr.', 'pathogen_status_babesia']], on='ID.nr.', how='outer')
print(f'Length of the merged dataframe: {len(merged_df)}')

# Print the number and percentage of ticks positive for Borrelia, Babesia, and both pathogens (irrespective of tick stage and feeiding time)
print(f'Number of ticks positive for Borrelia: {merged_df[merged_df["pathogen_status_borrelia"] == "positive"].shape[0]}')
print(f'Percentage of ticks positive for Borrelia out of all ticks: {merged_df[merged_df["pathogen_status_borrelia"] == "positive"].shape[0] / len(merged_df) * 100:.2f}%')
print(f'Number of ticks positive for Babesia: {merged_df[merged_df["pathogen_status_babesia"] == "positive"].shape[0]}')
print(f'Percentage of ticks positive for Babesia out of all ticks: {merged_df[merged_df["pathogen_status_babesia"] == "positive"].shape[0] / len(merged_df) * 100:.2f}%')
print(f'Number of ticks positive for both Borrelia and Babesia: {merged_df[(merged_df["pathogen_status_borrelia"] == "positive") & (merged_df["pathogen_status_babesia"] == "positive")].shape[0]}')
print(f'Percentage of ticks positive for both Borrelia and Babesia out of all ticks: {merged_df[(merged_df["pathogen_status_borrelia"] == "positive") & (merged_df["pathogen_status_babesia"] == "positive")].shape[0] / len(merged_df) * 100:.2f}%')



# Calculate the anticipated co-occurrence of Borrelia and Babesia in ticks
print('================== Chi-square test for co-occurrence of Borrelia and Babesia in ticks =================')

# Create a contingency table for the Chi-square test
contingency_table = pd.crosstab(merged_df['pathogen_status_borrelia'], merged_df['pathogen_status_babesia'])
print(f'Contingency table:\n{contingency_table}')

# Perform the Chi-square test using scipy.stats (sp_stats)
chi2, p_value, dof, expected = sp_stats.chi2_contingency(contingency_table)
print(f'Chi-square statistic: {chi2:.2f}')
print(f'P-value: {p_value:.4f}')
print(f'Degrees of freedom: {dof}')
print(f'Expected frequencies:\n{expected}')

# %%
import sys
import matplotlib
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scipy
import seaborn as sns
import statsmodels

print(f'Matplotlib: {matplotlib.__version__}') # only for plotting, not for analysis
print(f'Seaborn: {sns.__version__}') # only for plotting, not for analysis

print(f'Python version: {sys.version.split()[0]}')
print(f'pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'statsmodels: {statsmodels.__version__}')
print(f'SciPy: {scipy.__version__}')

r_version = ro.r('R.version.string')[0]
lme4_version = ro.r('as.character(packageVersion("lme4"))')[0]

print(f"rpy2: {version('rpy2')}")
print(f'R version: {r_version}')
print(f'lme4 version: {lme4_version}')

# %%
