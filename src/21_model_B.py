'''
### Bx model design:
* BA. the input data is gene expression data of current visit and current UPDRS, the output data is UPDRS of next visit (in 12 months)
  * -- the model is to predict the UPDRS of next visit
  * -- current expression data + current UPDRS -> next UPDRS in 12 months

* BB. the input data is gene expression data of current visit, current UPDRS, time period until next visit, the output data is UPDRS of next visit
  * -- the model is to predict the UPDRS of next visit
  * -- current expression data + current UPDRS + time period -> next UPDRS
* <u>BB is the most general model design for Bx model, and it can be used for both BA and BB by setting the time period as 0 for BA</u>
  * Prepare a data table with columns: current expression data, current UPDRS, time period until next visit, UPDRS of next visit
'''