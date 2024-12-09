import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

cohort = "ppmi"

expr_df = pd.read_csv("../results/processed/sample_expr_"+cohort+".csv",index_col=0,sep=",",header=0)




