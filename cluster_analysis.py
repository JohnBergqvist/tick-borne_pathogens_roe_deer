'''
(C) John Bergqvist 2026
This script is designed to analyse the pathogen prevalence in ticks, and the quantification of the pathogen load, taking into account the roe deer ID that the ticks were taken from. 
The analysis is performed to test if there are host-specific confounders affecting the accosiation between feeding time and pathogen prevlanece. 

The statistical models used for analysing the prevalence of pathogens in ticks are the Generalised Estimating Equation (GEE) and the Generalised Linear Mixed Model (GLMM).
The statistical model used for analysing the quantification of pathogens in ticks is the Linear Mixed Model (LMM).

'''

#%%
# import necessary libraries
from pathlib import Path
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# rpy2 imports for bridging Python and R directly
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
#%%

# Directories 
# define the HOME_DIR as the place where the scirpt is located, and print it to the console

HOME_DIR = Path(__file__).resolve().parent

borrelia_data_dir = HOME_DIR / 'borrelia_data.csv'
babesia_data_dir = HOME_DIR / 'babesia_data.csv'

# load the data
borrelia_df = pd.read_csv(borrelia_data_dir)
babesia_df = pd.read_csv(babesia_data_dir)



# Make a new column in both dfs called 'roe_deer_id' that is the first characters before the first '-' of the column 'ID.nr.'
borrelia_df['roe_deer_id'] = borrelia_df['ID.nr.'].str.split('-').str[0]
babesia_df['roe_deer_id'] = babesia_df['ID.nr.'].str.split('-').str[0]

# Checks to ensure the two dataframes contain the same roe deer and tick information, and that the only difference is the pathogen prevalence information.
# Create a loop to iterate through the two dataframes, and for each dataframe, print the length of the dataframe, 
# the number of unique roe deer IDs, and the number of unique roe deer IDs that have the 'Developmental stage' column as 'A' (adult), 'N' (nymph), and 'L' (larva)

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


#%%

###============================ PATHOGEN PREVALENCE ANALYSIS ============================###
# This analsis is only performed on the ticks that are nymphs.

# create a loop to filter the two dataframes to only contain rows with the letter 'N' in the 'developmental_stage' column
# to filter only for nymphs, remove the rows with a '?' in the 'feeding_time' column, and remove the rows with missing values in the 'feeding_time' column

for df, name in zip([borrelia_df, babesia_df], ['Borrelia', 'Babesia']):
    print(f'================== {name} Key Values For Prevalence Analysis=================')
    print(f'Number of ticks in the {name} dataframe before filtering: {len(df)}')
    df_filtered = df[df['developmental_stage'] == 'N']
    df_filtered = df_filtered[df_filtered['feeding_time'] != '?']
    df_filtered = df_filtered.dropna(subset=['feeding_time'])
    print(f'Number of ticks in the {name} dataframe after filtering (only nymps, no unknown feeding times): {len(df_filtered)}')
    print(f'Difference in number of ticks: {len(df) - len(df_filtered)}')
    print(f'Number of unique roe deer IDs in the {name} dataframe before filtering: {df["roe_deer_id"].nunique()}')
    print(f'Number of unique roe deer IDs in the {name} dataframe after filtering: {df_filtered["roe_deer_id"].nunique()}')
    print(f'Difference in the number of unique roe deer IDs: {df["roe_deer_id"].nunique() - df_filtered["roe_deer_id"].nunique()}')
    print(f'Number of ticks with {name} detected before filtering: {df[df["pathogen_status"] == "+"].shape[0]}')
    print(f'Number of ticks with {name} detected after filtering: {df_filtered[df_filtered["pathogen_status"] == "+"].shape[0]}')
    print(f'Number of adult ticks with {name} detected before filtering: {df[(df["pathogen_status"] == "+") & (df["developmental_stage"] == "A")].shape[0]}')
    print(f'Number of unique roe deer IDs with {name} detected after filtering: {df_filtered[df_filtered["pathogen_status"] == "+"]["roe_deer_id"].nunique()}')
    # save the filtered dataframe to a new variable
    if name == 'Borrelia':
        borrelia_df_filtered = df_filtered
    else:
        babesia_df_filtered = df_filtered

# populate the 'pathogen_status' column with 'positive' for '+' and 'negative' for blank spaces for both dataframes
for df, name in zip([borrelia_df_filtered, babesia_df_filtered], ['Borrelia', 'Babesia']):
    print(f'================== {name} pathogen status =================')
    df['pathogen_status'] = np.where(df['pathogen_status'] == '+', 'positive', 'negative')
    print(f'Number of positive ticks in the {name} dataframe: {df[df["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe: {df[df["pathogen_status"] == "negative"].shape[0]}')


# Transform the 'pathgoen_status' column to a binary variable, where 'positive' is 1 and 'negative' is 0 for both dataframes
for df, name in zip([borrelia_df_filtered, babesia_df_filtered], ['Borrelia', 'Babesia']):
    df['pathogen_status'] = df['pathogen_status'].map({'positive': 1, 'negative': 0})
    print(f'================== {name} pathogen status after transformation (1=positive, 0=negative) =================')
    print(f'Check the unique values in the {name} dataframe after transformation: {df["pathogen_status"].unique()}')
    print(f'Number of positive ticks in the {name} dataframe after transformation: {df[df["pathogen_status"] == 1].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe after transformation: {df[df["pathogen_status"] == 0].shape[0]}')

# define the colors to be used in the plots - grey for negative ticks and red for positive ticks
colors = {'negative': 'grey', 'positive': 'red'}

# Perform logistic regression for both dataframes to assess the association between feeding time and pathogen prevalence, while accounting for the clustering effect of roe deer ID.
for df, name in zip([borrelia_df_filtered, babesia_df_filtered], ['Borrelia', 'Babesia']):
    print(f'================== {name} logistic regression =================')
    # Fit the logistic regression model
    X = df[['feeding_time']]
    y = df['pathogen_status']
    model = LogisticRegression()
    model.fit(X, y)
    # Print the coefficients and intercept
    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}')
    feeding_time_range = np.linspace(df['feeding_time'].min(), df['feeding_time'].max(), 100).reshape(-1, 1)
    predicted_probabilities = model.predict_proba(feeding_time_range)[:, 1]
    # Plot the logistic regression curve, where the colour of the points is determined by the roe deer ID
    plt.figure(figsize=(10, 6))
    # Scatter plot of the actual data points, colored by roe deer ID
    sns.scatterplot(
        x='feeding_time',
        y='pathogen_status',
        data=df,
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

# plot the number of positive and negative ticks for each roe deer ID in a stacked bar chart for both dataframes


for df, name in zip([borrelia_df_filtered, babesia_df_filtered], ['Borrelia', 'Babesia']):
    print(f'================== {name} stacked bar chart =================')
    # count the number of positive and negative ticks for each roe deer ID
    tick_counts_by_deer = df.groupby(['roe_deer_id', 'pathogen_status']).size().unstack(fill_value=0).reset_index()
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



# %%
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
# 2. GENERALISED ESTIMATING EQUATION (GEE)
# ==============================================================================
    print("======================================================")
    print(f" 1. GENERALISED ESTIMATING EQUATION (GEE) - statsmodels - {name}")
    print("======================================================")

# WHY WE DO THIS: The GEE adjusts the standard errors of our population-level 
# estimate to account for the fact that ticks on the same deer are not independent.
# 
# cov_struct.Exchangeable() assumes that any two ticks collected from the same 
# roe deer have the same baseline correlation in their probability of infection.
    gee_model = smf.gee("pathogen_status ~ feeding_time", 
                        groups=df['roe_deer_id'], 
                        data=df, 
                        family=sm.families.Binomial(), 
                        cov_struct=sm.cov_struct.Exchangeable())

    gee_result = gee_model.fit()
    print(gee_result.summary())


# ==============================================================================
# 3. GENERALISED LINEAR MIXED MODEL (GLMM)
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






# %%

#####==================== Quantification analysis ====================#####

q_borrelia_df = borrelia_df.copy()
q_babesia_df = babesia_df.copy()

# Filter the dataframes to only keep the nymphs, and remove the rows with a '?' or  missing values in the 'feeding_time' column.
for q_df, name in zip([q_borrelia_df, q_babesia_df], ['Borrelia', 'Babesia']):
    print(f'================== {name} quantification analysis =================')
    print(f'Length of the dataframe before filtering: {len(q_df)}')
    q_df = q_df[q_df['developmental_stage'].isin(['N', 'A'])]
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
    q_df['pathogen_status'] = np.where(q_df['pathogen_status'] == '+', 'positive', 'negative')
    print(f'Number of positive ticks in the {name} dataframe: {q_df[q_df["pathogen_status"] == "positive"].shape[0]}')
    print(f'Number of negative ticks in the {name} dataframe: {q_df[q_df["pathogen_status"] == "negative"].shape[0]}')
    # keep only the positive ticks to analyse the quantity of pathogen in the ticks
    q_df = q_df[q_df['pathogen_status'] == 'positive']
    print(f'Length of the dataframe with only positive ticks: {len(q_df)}')
    # transform the 'quantification' column to numeric
    q_df['quantification'] = pd.to_numeric(q_df['quantification'], errors='coerce')
    # transform the 'quantification' column to log10 scale, and add a small constant to avoid log(0)
    q_df['log_quantification_per_tick'] = np.log10(q_df['quantification'] + 1)

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

## Borrelia:
# p-value: 0.861, which is not statistically significant at the 0.05 level. 
# Coefficient of feeding duration: -0.002, indicating a very small negative effect of feeding duration on the bacterial load of an infected tick, but this effect is not statistically significant.
# Host Variance: The random intercept variance (Group Var) is 0.107, indicating that there is a small baseline difference in bacterial loads depending on the host deer, but this effect is not statistically significant.

## Babesia:
# p-value: 0.356, which is not statistically significant at the 0.05 level.
# Coefficient of feeding duration: -0.008, indicating a small negative effect of feeding duration on the parasite load of an infected tick, but this effect is not statistically significant.
# Host Variance: The random intercept variance (Group Var) is 0.026, indicating that there is a very small baseline difference in parasite loads depending on the host deer, but this effect is not statistically significant.

# %%
