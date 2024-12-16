# %%
import pandas as pd
import numpy as np

# %%
## ==============================================
## Prepare a data table with columns: expression data, time period, UPDRS
DEG_df = pd.read_csv("../results/processed/DEG_genes.txt",index_col=0,sep="\t",header=0)
gene_ls = DEG_df.index.tolist()

expr_df = pd.read_csv("../results/processed/sample_expr_pdbp.csv", index_col=0, header=0)
meta_df = pd.read_csv("../results/processed/sample_meta_pdbp.csv", index_col=0, header=0)

patient_visit_df = pd.read_csv("../results/processed/PDBP_patient_visits.csv", index_col=0, header=0)
# PPMI viits: m0, m6, m12, m24, m36
# PDBP viits: m0, m6, m12, m18, m24

# %%
# construst a data table with columns: current expression data, current UPDRS, time period, Next UPDRS
visit_month = {"visit_m0":0,"visit_m6":6,"visit_m12":12,"visit_m18":18,"visit_m24":24,"visit_m36":36}
df = pd.DataFrame(columns=gene_ls+["current_updrs","time_period","next_updrs"])
for patient in patient_visit_df.index:
    patient_record = patient_visit_df.loc[patient,:].to_dict()

    ## remove missing visits
    patient_visit_ls = []
    for visit in patient_record.keys():
        if str(patient_record[visit]).strip() == "" or patient_record[visit]== np.nan or str(patient_record[visit]).strip() == "nan":
            print(patient_record[visit])
            continue
        else:
            patient_visit_ls.append([visit,patient_record[visit]])

    ## construct a data table
    for i in range(len(patient_visit_ls)-1):
        current_visit = patient_visit_ls[i][1]
        current_expr = list(expr_df.loc[gene_ls,current_visit].T.values)
        current_updrs = meta_df.loc[current_visit,"mds_updrs_part_iii_summary_score"]

        next_visit = patient_visit_ls[i + 1][1]
        # next_expr = expr_df.loc[gene_ls,next_visit].T
        next_updrs = meta_df.loc[next_visit,"mds_updrs_part_iii_summary_score"]

        time_period = visit_month[patient_visit_ls[i + 1][0]] - visit_month[patient_visit_ls[i][0]]

        temp_df = pd.DataFrame([current_expr+[current_updrs,time_period,next_updrs]],columns=gene_ls+["current_updrs","time_period","next_updrs"])
        temp_df.index = [current_visit]
        df = pd.concat([df,temp_df],axis=0)

print(df.shape)
df.to_csv("../results/training_testing/PDBP_data_current_next.csv",index=True)


# %%
## ==============================================
## Prepare a data table with columns: previous expression data, previous UPDRS, current expression data, current UPDRS, time period, next UPDRS
## only consider: time period = 12m
## PDBP viits: m0, m12 -> m24


gene_ls_previous = ["previous_"+i for i in gene_ls]
gene_ls_current = ["current_"+i for i in gene_ls]
df = pd.DataFrame(columns=gene_ls_previous + gene_ls_current+["previous_updrs","current_updrs","time_period","next_updrs"])

patient_visit_df_m0 = patient_visit_df.loc[:,["visit_m0","visit_m12","visit_m24"]]
## remove rows if one of the visit is missing
patient_visit_df_m0 = patient_visit_df_m0.dropna(axis=0)


# %%
patient_visit_ls = []
patient_record = patient_visit_df_m0.to_dict(orient="index")
for patient in patient_record.keys():
    patient_visit_ls.append(patient_record[patient].items())

for patient_visit in patient_visit_ls:
    patient_visit = list(patient_visit)
    previous_visit = patient_visit[0][1]
    previous_expr = list(expr_df.loc[gene_ls, previous_visit].T.values)
    previous_updrs = meta_df.loc[previous_visit, "mds_updrs_part_iii_summary_score"]

    current_visit = patient_visit[1][1]
    current_expr = list(expr_df.loc[gene_ls, current_visit].T.values)
    current_updrs = meta_df.loc[current_visit, "mds_updrs_part_iii_summary_score"]

    next_visit = patient_visit[2][1]
    next_updrs = meta_df.loc[next_visit, "mds_updrs_part_iii_summary_score"]

    time_period = visit_month[patient_visit[2][0]] - visit_month[patient_visit[1][0]]

    temp_df = pd.DataFrame(
        [previous_expr + current_expr + [previous_updrs, current_updrs, time_period, next_updrs]],
        columns=gene_ls_previous + gene_ls_current + ["previous_updrs", "current_updrs", "time_period", "next_updrs"])
    temp_df.index = [current_visit]
    df = pd.concat([df, temp_df], axis=0)

print(df.shape)
df.to_csv("../results/training_testing/PDBP_data_previous_current_next.csv", index=True)

