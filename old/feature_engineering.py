import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder


def apply_feature_engineering(df):
    """
    Apply feature engineering as a function for
    standardization.
    
    Parameters:
    - df: Description of the parameter
    """
    print("Starting feature engineering...")
    start_time = time.time()

    print("Adding datetime features...")
    # Extract HospitalStayDays to remove 
    df.loc[:, "HospitalStayDays"] = (df["DischargeDt"] - df["AdmissionDt"]).dt.days
    df.loc[:, "HospitalStayDays"] = df["HospitalStayDays"].fillna(0).astype(int)
    df.loc[:, "ClaimDuration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days
    df.loc[:, "ClaimDuration"] = df["ClaimDuration"].fillna(df["ClaimDuration"].median()).astype(int)
    df.loc[:, "DaysBeforeAdmission"] = (df["AdmissionDt"] - df["ClaimStartDt"]).dt.days
    df.loc[:, "DaysBeforeAdmission"] = df["DaysBeforeAdmission"].fillna(0).astype(int)
    df.loc[:, "ClaimStartMonth"] = df["ClaimStartDt"].dt.month
    df.loc[:, "ClaimStartMonth"] = df["ClaimStartMonth"].fillna(0).astype(np.int32)
    df.loc[:, "ClaimStartWeekday"] = df["ClaimStartDt"].dt.weekday
    df.loc[:, "ClaimStartWeekday"] = df["ClaimStartWeekday"].fillna(0).astype(np.int32)
    df.loc[:, "ClaimStartYear"] = df["ClaimStartDt"].dt.year
    df.loc[:, "ClaimStartYear"] = df["ClaimStartYear"].fillna(0).astype(np.int32)
    df.loc[:, "DaysSinceLastClaim"] = df.groupby("BeneID")["ClaimStartDt"].transform(lambda x: x.diff().dt.days)
    df.loc[:, "DaysSinceLastClaim"] = df["DaysSinceLastClaim"].fillna(0).astype(np.int32)
    df.loc[:, "AgeAtClaim"] = df["ClaimStartYear"] - df["DOB"].dt.year
    df.loc[:, "AgeAtClaim"] = df["AgeAtClaim"].fillna(0).astype(np.int32)
    print(f"Added datetime features. Time elapsed: {time.time() - start_time:.2f}s")

    print("Discretizing age...")
    # Descritize age
    df.loc[:, "AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 49, 60, 79, 97, 116],
        labels=["Under 50", "50-60", "61-79", "80-97", "98+"]
    )
    print(f"Discretized age. Time elapsed: {time.time() - start_time:.2f}s")

    print("Filling in missing values...")
    # Fill in missing values
    df.loc[:, "DeductibleAmtPaid"] = df["DeductibleAmtPaid"].fillna(0)
    df.loc[:, "AttendingPhysician"] = df.groupby("Provider")["AttendingPhysician"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
    df.loc[:, "OtherPhysician"] = df["OtherPhysician"].fillna("None")
    df.loc[:, "ClmAdmitDiagnosisCode"] = df["ClmAdmitDiagnosisCode"].fillna("Not Applicable")
    df.loc[:, "OperatingPhysician"] = df["OperatingPhysician"].fillna("None")
    diag_cols = [col for col in df.columns if "ClmDiagnosisCode" in col]
    df.loc[:, diag_cols] = df[diag_cols].fillna("No Diagnosis")
    proc_cols = [col for col in df.columns if "PrimaryProcedure" in col or "ClmProcedureCode" in col]
    df.loc[:, proc_cols] = df[proc_cols].fillna("No Procedure")
    print(f"Filled in missing values. Time elapsed: {time.time() - start_time:.2f}s")

    print("Transforming skewed distributions...")
    # Log transform skewed distributions
    df['InscClaimAmtReimbursed'] = np.log1p(df['InscClaimAmtReimbursed'])
    df['DeductibleAmtPaid'] = np.log1p(df['DeductibleAmtPaid'])
    df['IPAnnualDeductibleAmt'] = np.log1p(df['IPAnnualDeductibleAmt'])
    df['OPAnnualDeductibleAmt'] = np.log1p(df['OPAnnualDeductibleAmt'])
    print(f"Transformed skewed distributions. Time elapsed: {time.time() - start_time:.2f}s")

    print("Encoding categorical columns...")
    # Encode categorical distributions
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    apply_encoding(df, cols= cat_cols)
    print(f"Encoded categorical columns. Time elapsed: {time.time() - start_time:.2f}s")

    print("Dropping unnecessary columns...")
    # Drop unnecessary columns
    df.drop(columns=[
        "DOD", 
        "AdmissionDt", 
        "DischargeDt", 
        "ClaimStartDt", 
        "ClaimEndDt", 
        "DOB",
        "BeneID", 
        "ClaimID", 
        "InscClaimAmtReimbursed", 
        "DeductibleAmtPaid", 
        "IPAnnualDeductibleAmt", 
        "OPAnnualDeductibleAmt", 
        "Flag_Unknown_Procedures", 
        "Flag_Unknown_Diagnoses"
    ] + df.filter(like='Desc').columns.tolist(), inplace=True)
    print(f"Dropped unnecessary columns. Time elapsed: {time.time() - start_time:.2f}s")
    
    print("Feature engineering complete!")
    return df

def apply_encoding(df, cols):
    """
    Applies label encoding to the specified columns in a DataFrame.

    Parameters:
    - df: pandas dataframe, a dataframe containing the columns to be encoded
    - cols: list, a list of column names to apply label encoding

    Returns:
    None: The function modifies the DataFrame in place.
    """
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])