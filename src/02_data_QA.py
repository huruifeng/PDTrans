import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

cohort = "pdbp"

meta_df = pd.read_csv("../results/processed/sample_meta_"+cohort+".csv",index_col=0,sep=",")
meta_df["age_at_visit"] = meta_df.age_at_baseline.astype(int) + meta_df.visit_month/12.0

## UPDRS III scores distribution: mds_updrs_part_iii_summary_score ,case_control_other_at_baseline/sex
plt.figure(figsize=(10,6))
sns.histplot(meta_df, x="mds_updrs_part_iii_summary_score", hue="sex",bins=50, kde=True)
plt.title("UPDRS III scores distribution")
plt.xlabel("UPDRS III score")
plt.ylabel("Frequency")
plt.savefig("../results/qa/"+cohort+"_updrs3_distribution_sex.png")
plt.show()

## Age distribution
plt.figure(figsize=(10,6))
sns.histplot(meta_df, x="age_at_visit", bins=50, hue="sex", kde=True)
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("../results/qa/"+cohort+"_age_distribution_sex.png")
plt.show()

