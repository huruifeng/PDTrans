import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %%
cohort = "ppmi"

expr_df = pd.read_csv("../results/processed/sample_expr_"+cohort+".csv",index_col=0,sep=",",header=0)

DEG_df = pd.read_csv("../data/DEresult.padj05.xls",index_col=0,sep="\t",header=0)


# %%
gene_df = DEG_df.loc[:,["symbol","geneType"]]
gene_df_protein_coding = gene_df.loc[gene_df.geneType=="protein_coding",:]


# %%
gene_df.to_csv("../results/processed/DEG_genes.txt",sep="\t",index=True)
gene_df_protein_coding.to_csv("../results/processed/DEG_genes_protein_coding.txt",sep="\t",index=True)

