import pandas as pd
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


if __name__ == "__main__":
    data_folder = "UPDRS1"
    df = pd.read_csv(f"../results/training_testing/{data_folder}/PPMI_data_current_next.csv", index_col=0)
    ## remove rows if THERE IS MISSING DATA
    df = df.dropna(axis=0)
    df = df.loc[df["time_period"] == 12, :]

    val_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(val_df.index)

    x_train = train_df.loc[:, train_df.columns.str.startswith("ENSG") | train_df.columns.str.startswith("current_updrs")].values
    y_train = train_df.loc[:, train_df.columns.str.startswith("next_updrs")].values

    x_val = val_df.loc[:, val_df.columns.str.startswith("ENSG") | val_df.columns.str.startswith("current_updrs")].values
    y_val = val_df.loc[:, val_df.columns.str.startswith("next_updrs")].values

    ## original r2 and pcc
    r2 = r2_score(train_df["current_updrs"].values, train_df["next_updrs"].values)
    pcc = pearsonr(train_df["current_updrs"].values, train_df["next_updrs"].values)[0]
    print(f"Original train: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    r2 = r2_score(val_df["current_updrs"].values, val_df["next_updrs"].values)
    pcc = pearsonr(val_df["current_updrs"].values, val_df["next_updrs"].values)[0]
    print(f"Original val: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    # Define model
    model = SVR()
    model.fit(x_train, y_train)

    # Evaluate model
    y_train_pred = model.predict(x_train)
    r2 = r2_score(y_train, y_train_pred)
    pcc = pearsonr(y_train.ravel(), y_train_pred.ravel())[0]
    print(f"Training: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    y_val_pred = model.predict(x_val)
    r2 = r2_score(y_val, y_val_pred)
    pcc = pearsonr(y_val.ravel(), y_val_pred.ravel())[0]
    print(f"Validation: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

   ## predictions
    test_df = pd.read_csv(f"../results/training_testing/{data_folder}/PDBP_data_current_next.csv", index_col=0)
    test_df = test_df.dropna(axis=0)

    x_test = test_df.loc[:, test_df.columns.str.startswith("ENSG") | test_df.columns.str.startswith("current_updrs")].values
    y_test =test_df.loc[:, test_df.columns.str.startswith("next_updrs")].values

    r2 = r2_score(test_df["current_updrs"].values, test_df["next_updrs"].values)
    pcc = pearsonr(test_df["current_updrs"].values, test_df["next_updrs"].values)[0]
    print(f"Original test: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")

    y_test_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_test_pred)
    pcc = pearsonr(y_test.ravel(), y_test_pred.ravel())[0]
    print(f"Test: R^2 score: {r2:.4f}, PCC: {pcc:.4f}")
