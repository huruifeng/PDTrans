import os.path
import pandas as pd

# %%

output_folder = "../results/processed"
if not os.path.exists(output_folder): os.makedirs(output_folder,exist_ok=True)

## Load data
expr_data = pd.read_csv("../data/gene_expr_matrix_tpm_row_genes.txt", index_col=0, header=0, sep="\t")
clin_data = pd.read_csv("../data/PPB_RNAseq_samples_BigTable_11202022.tsv", index_col=0, header=0, sep="\t")

col_list = ['participant_id','Study','sex','age_at_baseline',
            'visit_month','education_level',
            'case_control_other_at_baseline',
            'case_control_other_latest',
            'upsit_total_score',
            'mds_updrs_total',
            'mds_updrs_part_i_summary_score','mds_updrs_part_ii_summary_score',
            'mds_updrs_part_iii_summary_score','mds_updrs_part_iv_summary_score']
clin_data = clin_data.loc[:,col_list]
clin_data.info()
# clin_data = clin_data.loc[~clin_data.mds_updrs_part_ii_summary_score.isna(),:]

all_samples = [x_i for x_i in list(clin_data.index) if x_i in list(expr_data.columns)]

clin_data = clin_data.loc[all_samples,:]
expr_data = expr_data.loc[:,all_samples]
clin_data.to_csv(output_folder+"/sample_meta.csv")
expr_data.to_csv(output_folder+"/sample_expr.csv")

ppmi_samples = [x_i for x_i in all_samples if x_i.startswith("PP-")]
clin_data_ppmi = clin_data.loc[ppmi_samples,:]
expr_data_ppmi = expr_data.loc[:,ppmi_samples]
clin_data_ppmi.to_csv(output_folder+"/sample_meta_ppmi.csv")
expr_data_ppmi.to_csv(output_folder+"/sample_expr_ppmi.csv")


pdbp_samples = [x_i for x_i in all_samples if x_i.startswith("PD-")]
clin_data_pdbp = clin_data.loc[pdbp_samples,:]
expr_data_pdbp = expr_data.loc[:,pdbp_samples]
clin_data_pdbp.to_csv(output_folder+"/sample_meta_pdbp.csv")
expr_data_pdbp.to_csv(output_folder+"/sample_expr_pdbp.csv")

# condition = (clin_data.case_control_other_latest==clin_data.case_control_other_at_baseline)
# clin_data = clin_data.loc[condition & (clin_data.case_control_other_latest.isin(["Case","Control"])),:]

print(clin_data_ppmi.visit_month.value_counts() )
print(clin_data_pdbp.visit_month.value_counts())

# %%
patient_visit_count = clin_data.groupby("participant_id").size().reset_index(name='counts')
patient_visit_count.to_csv(output_folder+"/patient_visit_count.csv")
patient_visit_count.counts.value_counts()

## PPMI
visit_dict = {}
for sample_i in clin_data_ppmi.index:
    patient_id = clin_data_ppmi.loc[sample_i,'participant_id']
    visit_month = "visit_m"+str(int(clin_data_ppmi.loc[sample_i,'visit_month']))
    if patient_id in visit_dict:
        if visit_month in visit_dict[patient_id]:
            print(f"patient {patient_id} has multiple visits for {visit_month}: {visit_dict[patient_id][visit_month], sample_i}")
            visit_dict[patient_id][visit_month] = sample_i
        else:
            visit_dict[patient_id][visit_month] = sample_i
    else:
        visit_dict[patient_id] = {}
        visit_dict[patient_id][visit_month]=sample_i

ppmi_visit_df = pd.DataFrame.from_dict(visit_dict,orient='index')
## PP-65003: 5 visits, PP-92490: 4 vivits, PP-60170: 3 visits
print(visit_dict["PP-65003"])
x_df = clin_data_ppmi.loc[clin_data_ppmi.participant_id.isin(["PP-65003"]),:]
x_df =ppmi_visit_df.loc[["PP-65003","PP-92490","PP-60170"],:]

ppmi_visit_df.to_csv(output_folder+"/PPMI_patient_visits.csv")

## PDBP
visit_dict = {}
for sample_i in clin_data_pdbp.index:
    patient_id = clin_data_pdbp.loc[sample_i,'participant_id']
    visit_month = "visit_m"+str(int(clin_data_pdbp.loc[sample_i,'visit_month']))
    if patient_id in visit_dict:
        if visit_month in visit_dict[patient_id]:
            print(f"patient {patient_id} has multiple visits for {visit_month}: {visit_dict[patient_id][visit_month], sample_i}")
            visit_dict[patient_id][visit_month] = sample_i
        else:
            visit_dict[patient_id][visit_month] = sample_i
    else:
        visit_dict[patient_id] = {}
        visit_dict[patient_id][visit_month]=sample_i
pdbp_visit_df = pd.DataFrame.from_dict(visit_dict,orient='index')
pdbp_visit_df.to_csv(output_folder+"/PDBP_patient_visits.csv")


# %%
expr_gene_var = expr_data_ppmi.var(axis=1)
ppmi_m0 = ppmi_visit_df.visit_m0.dropna().drop_duplicates()
ppmi_m6 = ppmi_visit_df.visit_m0.dropna().drop_duplicates()
ppmi_m12 = ppmi_visit_df.visit_m0.dropna().drop_duplicates()
ppmi_m24 = ppmi_visit_df.visit_m0.dropna().drop_duplicates()
ppmi_m36 = ppmi_visit_df.visit_m0.dropna().drop_duplicates()
## PPMI viits
for (i,visit_x) in zip(["m0","m6","m12","m24","m36"],[ppmi_m0,ppmi_m6,ppmi_m12,ppmi_m24,ppmi_m36]):
    clin_data_ppmi = clin_data.loc[visit_x,:]
    expr_data_ppmi = expr_data.loc[expr_gene_var>1,visit_x]

    expr_data_ppmi.to_csv(output_folder+"/PPMI_expr_"+i+".csv")
    clin_data_ppmi.to_csv(output_folder+"/PPMI_meta_"+i+".csv")

## PDBP viits
expr_gene_var = expr_data_pdbp.var(axis=1)
pdbp_m0 = pdbp_visit_df.visit_m0.dropna().drop_duplicates()
pdbp_m6 = pdbp_visit_df.visit_m6.dropna().drop_duplicates()
pdbp_m12 = pdbp_visit_df.visit_m12.dropna().drop_duplicates()
pdbp_m18 = pdbp_visit_df.visit_m18.dropna().drop_duplicates()
pdbp_m24 = pdbp_visit_df.visit_m24.dropna().drop_duplicates()
## PPMI viits
for (i, visit_x) in zip(["m0", "m6", "m12", "m18", "m24"], [pdbp_m0, pdbp_m6, pdbp_m12, pdbp_m18, pdbp_m24]):
    clin_data_pdbp = clin_data.loc[visit_x, :]
    expr_data_pdbp = expr_data.loc[expr_gene_var > 1, visit_x]

    expr_data_pdbp.to_csv(output_folder + "/PDBP_expr_" + i + ".csv")
    clin_data_pdbp.to_csv(output_folder + "/PDBP_meta_" + i + ".csv")


