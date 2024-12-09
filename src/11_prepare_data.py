## ==============================================
## This is a regression problem.
# INPUT:
#   -- gene expression data
#   -- time period until next visit
#   -- UPDRS score of current visit and previous visits
#   -- clinical data: sex, age, time period, etc.

# OUTPUT:
#   -- UPDRS III score

## ==============================================
## Data source:
## PPMI data:
# The gene expression data is from blood samples
# The visits: 0m, 6m, 12m, 24m, 36m

## PDBP data:
# The gene expression data is from blood samples
# The visits: 0m, 6m, 12m, 18m, 24m

## ==============================================
## Ax model design:
# AA. the input data is gene expression data of current visit, the output data is UPDRS of current visit
#       -- the model is to predict the UPDRS of current visit
#       -- current expression data -> current UPDRS

# AB. the input data is gene expression data of current visit, the output data is UPDRS of next visit (in 12 months)
#       -- the model is to predict the UPDRS of next visit
#       -- expression data -> next UPDRS in 12 months

# AC. the input data is gene expression data of current visit, time period until next visit, the output data is UPDRS of next visit
#       -- the model is to predict the UPDRS of next visit
#       -- expression data + time period -> next UPDRS
### AC is the most general model design for Ax model, and it can be used for both AA and AB by setting the time period as 0 for AA and 12 for AB
### Prepare a data table with columns: expression data, time period, UPDRS

##--------------------------
## Bx model design:
# BA. the input data is gene expression data of current visit and current UPDRS, the output data is UPDRS of next visit (in 12 months)
#       -- the model is to predict the UPDRS of next visit
#       -- current expression data + current UPDRS -> next UPDRS in 12 months

# BB. the input data is gene expression data of current visit, current UPDRS, time period until next visit, the output data is UPDRS of next visit
#       -- the model is to predict the UPDRS of next visit
#       -- current expression data + current UPDRS + time period -> next UPDRS
### BB is the most general model design for Bx model, and it can be used for both BA and BB by setting the time period as 0 for BA
### Prepare a data table with columns: expression data, current UPDRS, time period until next visit, UPDRS of next visit

##--------------------------
## Cx model design:
# CA. the input data is gene expression data of current visit, current UPDRS, 1 previous UPDRS, the output data is UPDRS of next visit (in 12 months)
#       -- the model is to predict the UPDRS of next visit
#       -- current expression data + current UPDRS + 1 previous UPDRS -> next UPDRS in 12 months

# CB. the input data is gene expression data of current visit, current UPDRS, 1 previous UPDRS, time period until next visit, the output data is UPDRS of next visit
#       -- the model is to predict the UPDRS of next visit
#       -- current expression data + current UPDRS + 1 previous UPDRS + time period -> next UPDRS
### CB is the most general model design for Cx model, and it can be used for both CA and CB by setting the time period as 0 for CA
### Prepare a data table with columns: expression data, current UPDRS, 1 previous UPDRS, time period until next visit, UPDRS of next visit

##--------------------------
## Dx model design:
# DA. the input data is gene expression data of current visit, current UPDRS, 2 previous UPDRS, the output data is UPDRS of next visit
#       -- the model is to predict the UPDRS of next visit
#       -- current expression data + current UPDRS + 2 previous UPDRS -> next UPDRS

# DB. the input data is gene expression data of current visit, current UPDRS, 2 previous UPDRS, time period until next visit, the output data is UPDRS of next visit
#       -- the model is to predict the UPDRS of next visit
#       -- current expression data + current UPDRS + 2 previous UPDRS + time period -> next UPDRS
### DB is the most general model design for Dx model, and it can be used for both DA and DB by setting the time period as 0 for DA
### Prepare a data table with columns: expression data, current UPDRS, 2 previous UPDRS, time period until next visit, UPDRS of next visit

import pandas as pd

## ==============================================
## Data preparation:
# 1. Load the gene expression data
expr_df = pd.read_csv('../results/processed_data/sample_expr_ppmi.csv', index_col=0, header=0)
print(expr_df.head())

meta_df = pd.read_csv('../results/processed_data/sample_meta_ppmi.csv', index_col=0, header=0)


# 2. Load the UPDRS data
# 3. Load the clinical data
# 4. Prepare the data table for the model design

