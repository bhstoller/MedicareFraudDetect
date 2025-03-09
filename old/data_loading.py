from datetime import datetime
import os
import pandas as pd

def load_data():
    """
    Load the dataset using parquet and pyarrow
    """
    merge_data()
    correct_codes()
    date_cols = [
        'ClaimStartDt',
        'ClaimEndDt',
        'AdmissionDt',
        'DischargeDt',
        'DOB'
    ]
    df = pd.read_csv('data/train_data_cleaned.csv', parse_dates= date_cols, low_memory= False)
    return df

def merge_data():
    raw_data_location = "data/raw/"
    train_inpatient = pd.read_csv(raw_data_location + "Train_Inpatientdata-1542865627584.csv", dtype={"DiagnosisGroupCode": str})
    train_outpatient = pd.read_csv(raw_data_location + "Train_Outpatientdata-1542865627584.csv")
    train_beneficiaries = pd.read_csv(raw_data_location + "Train_Beneficiarydata-1542865627584.csv", dtype={"DOD": str})
    train_providers = pd.read_csv(raw_data_location + "Train-1542865627584.csv")
    unlabeled_inpatient = pd.read_csv(raw_data_location + "Test_Inpatientdata-1542969243754.csv", dtype={"DiagnosisGroupCode": str})
    unlabeled_outpatient = pd.read_csv(raw_data_location + "Test_Outpatientdata-1542969243754.csv")
    unlabeled_beneficiaries = pd.read_csv(raw_data_location + "Test_Beneficiarydata-1542969243754.csv", dtype={"DOD": str})
    unlabeled_providers = pd.read_csv(raw_data_location + "Test-1542969243754.csv")
    date_columns = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]
    for col in date_columns:
        for df in [train_inpatient, train_outpatient, train_beneficiaries, unlabeled_inpatient, unlabeled_outpatient, unlabeled_beneficiaries]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    current_year = datetime.now().year
    train_beneficiaries["Age"] = current_year - train_beneficiaries["DOB"].dt.year
    unlabeled_beneficiaries["Age"] = current_year - unlabeled_beneficiaries["DOB"].dt.year
    chronic_cols = [col for col in train_beneficiaries.columns if "ChronicCond" in col]
    train_beneficiaries["ChronicCount"] = train_beneficiaries[chronic_cols].apply(lambda x: sum(x == 1), axis=1)
    unlabeled_beneficiaries["ChronicCount"] = unlabeled_beneficiaries[chronic_cols].apply(lambda x: sum(x == 1), axis=1)
    train_inpatient["TotalClaims"] = train_inpatient.groupby("Provider")["ClaimID"].transform("count")
    train_outpatient["TotalClaims"] = train_outpatient.groupby("Provider")["ClaimID"].transform("count")
    train_data = train_inpatient.merge(train_beneficiaries, on="BeneID", how="left")
    train_data = train_data.merge(train_providers, on="Provider", how="left")
    train_data_outpatient = train_outpatient.merge(train_beneficiaries, on="BeneID", how="left")
    train_data_outpatient = train_data_outpatient.merge(train_providers, on="Provider", how="left")
    train_data["ClaimType"] = "Inpatient"
    train_data_outpatient["ClaimType"] = "Outpatient"
    train_data = pd.concat([train_data, train_data_outpatient], axis=0)
    unlabeled_data = unlabeled_inpatient.merge(unlabeled_beneficiaries, on="BeneID", how="left")
    unlabeled_data = unlabeled_data.merge(unlabeled_providers, on="Provider", how="left")
    unlabeled_data_outpatient = unlabeled_outpatient.merge(unlabeled_beneficiaries, on="BeneID", how="left")
    unlabeled_data_outpatient = unlabeled_data_outpatient.merge(unlabeled_providers, on="Provider", how="left")
    unlabeled_data["ClaimType"] = "Inpatient"
    unlabeled_data_outpatient["ClaimType"] = "Outpatient"
    unlabeled_data = pd.concat([unlabeled_data, unlabeled_data_outpatient], axis=0)
    os.makedirs("data", exist_ok=True)
    train_data.to_csv("data/train_data.csv", index=False)
    unlabeled_data.to_csv("data/unlabeled_data.csv", index=False)

def correct_codes():
    # Create a library to map datatypes correctly
    dtype_mapping = {
        "DiagnosisGroupCode": str,
        "DOD": str,
        "ClmAdmitDiagnosisCode": str,
        **{f"ClmDiagnosisCode_{i}": str for i in range(1, 11)}
    }

    # Identify date columns
    date_columns = ["ClaimStartDt", "ClaimEndDt", "DOB", "DOD", "AdmissionDt", "DischargeDt"]

    train_data = pd.read_csv("data/train_data.csv", dtype=dtype_mapping, parse_dates=date_columns)
    unlabeled_data = pd.read_csv("data/unlabeled_data.csv", dtype=dtype_mapping, parse_dates=date_columns)

    date_format = "%Y-%m-%d"

    # Convert date columns
    for col in ["AdmissionDt", "DischargeDt", "DOD"]:
        if col in train_data.columns:
            train_data[col] = pd.to_datetime(train_data[col], format=date_format, errors="coerce")
        if col in unlabeled_data.columns:
            unlabeled_data[col] = pd.to_datetime(unlabeled_data[col], format=date_format, errors="coerce")

    diagnosis_columns = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    procedure_columns = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

    def fix_icd_format(code):
        """Fix ICD-9/10 codes that are incorrectly formatted due to float conversion."""
        if pd.isna(code) or code in ["", "NAN", "NaN", "<NA>"]:
            return pd.NA

        code = str(code).strip().upper()

        if code.endswith(".0"):
            code = code[:-2]

        return code

    # dealing with various NaN formats across records
    for col in diagnosis_columns + procedure_columns:
        train_data[col] = train_data[col].replace(
            {"nan": pd.NA, "0nan": pd.NA, "NaN": pd.NA, "NAN": pd.NA, "<NA>": pd.NA, "": pd.NA},
            regex=True
        )

        train_data[col] = train_data[col].astype("string").apply(fix_icd_format)

    # apply the function to fix the format of codes
    def map_category(code, category_dict):
        """Maps an ICD-9 code to a category based on provided ranges."""
        if pd.isna(code) or code=="":
            return "Unknown"

        if code.startswith("V"):
            return "V-Codes"
        if code.startswith("E"):
            return "E-Codes"

        code = int(code)

        for (low, high), category in category_dict.items():
            if low*100 <= code <= high*100:
                return category

        return "Unknown"

    # Define the ranges for diagnosis and procedure categories
    diagnosis_category_map = {
        (1, 139): "Infectious & Parasitic Diseases",
        (140, 239): "Neoplasms",
        (240, 279): "Endocrine, Nutritional, and Metabolic Diseases",
        (280, 289): "Diseases of the Blood",
        (290, 319): "Mental Disorders",
        (320, 389): "Diseases of the Nervous System and Sense Organs",
        (390, 459): "Diseases of the Circulatory System",
        (460, 519): "Diseases of the Respiratory System",
        (520, 579): "Diseases of the Digestive System",
        (580, 629): "Diseases of the Genitourinary System",
        (630, 679): "Pregnancy, Childbirth, and the Puerperium",
        (680, 709): "Diseases of the Skin and Subcutaneous Tissue",
        (710, 739): "Diseases of the Musculoskeletal System",
        (740, 759): "Congenital Anomalies",
        (780, 799): "Symptoms, Signs, and Ill-Defined Conditions",
        (800, 999): "Injury and Poisoning",
    }

    for col in diagnosis_columns:
        train_data[col] = train_data[col].astype(str).replace({"<NA>": ""}).str.strip().str.upper()
        train_data[f"{col}_Category"] = train_data[col].apply(lambda x: map_category(x, diagnosis_category_map))

    procedure_category_map = {
        (0, 0): "Miscellaneous Diagnostic & Therapeutic Procedures",
        (1, 5): "Procedures on the Nervous System",
        (6, 7): "Procedures on the Endocrine System",
        (8, 16): "Procedures on the Eye & Ear",
        (17, 20): "Operations on the Cardiovascular System",
        (21, 29): "Operations on the Respiratory System",
        (30, 34): "Operations on the Digestive System",
        (35, 39): "Cardiovascular Procedures",
        (40, 41): "Procedures on the Lymphatic & Hemic System",
        (42, 54): "Procedures on the Digestive System",
        (55, 59): "Procedures on the Urinary System",
        (60, 64): "Procedures on the Male Genital Organs",
        (65, 71): "Procedures on the Female Genital Organs",
        (72, 75): "Obstetric & Gynecological Procedures",
        (76, 84): "Orthopedic Procedures",
        (85, 86): "Operations on the Breast and Skin",
        (87, 99): "Radiology, Physical Therapy, and Other Miscellaneous Procedures",
    }

    # Create new procedure category columns
    for col in procedure_columns:
        train_data[col] = train_data[col].astype(str).replace({"<NA>": ""}).str.strip().str.upper()
        train_data[f"{col}_Category"] = train_data[col].apply(lambda x: map_category(x, procedure_category_map))

    # Create PrimaryProcedure column from first element
    train_data["PrimaryProcedure"] = (
    train_data["ClmProcedureCode_1"]
        .astype(str)
        .str.strip()
        .replace({"nan": pd.NA, "0nan": pd.NA, "NaN": pd.NA, "": pd.NA})
        .str.zfill(4)
    )

    # Fill NaN values in PrimaryProcedure
    train_data["PrimaryProcedure"] = train_data["PrimaryProcedure"].fillna("Unknown")

    train_data["NumProcedures"] = train_data[procedure_columns].notna().sum(axis=1)

    proc_desc = pd.read_excel("medical_codes/CMS32_DESC_LONG_SHORT_SG.xlsx", dtype=str)

    # Rename columns relevant to procedures
    proc_desc.rename(columns={
        "PROCEDURE CODE": "ProcedureCode",
        "LONG DESCRIPTION": "PrimaryProcedure_LongDesc",
        "SHORT DESCRIPTION": "PrimaryProcedure_ShortDesc"
    }, inplace=True)
    proc_desc["ProcedureCode"] = proc_desc["ProcedureCode"].astype(str).str.zfill(4)

    # Merge procedure descriptions from XLSX
    train_data = train_data.merge(proc_desc, left_on="PrimaryProcedure", right_on="ProcedureCode", how="left")

    train_data.drop(columns=["ProcedureCode"], inplace=True, errors="ignore")

    # extract long and short descriptions
    dx_desc = pd.read_excel("medical_codes/CMS32_DESC_LONG_SHORT_DX.xlsx", dtype=str)
    dx_desc.rename(columns={"DIAGNOSIS CODE": "DiagnosisCode"}, inplace=True)
    dx_desc["DiagnosisCode"] = dx_desc["DiagnosisCode"].astype(str).str.zfill(4)

    # Merge diagnosis descriptions from XLSX
    for col in diagnosis_columns:
        if col in train_data.columns:
            train_data = train_data.merge(dx_desc, left_on=col, right_on="DiagnosisCode", how="left", suffixes=("", f"_{col}"))

    # Drop numerical reference columns
    columns_to_drop = [
        "DiagnosisGroupCode",
        *diagnosis_columns,
        *[f"ClmProcedureCode_{i}" for i in range(1, 7)],
        "DiagnosisCode",
        *[f"DiagnosisCode_ClmDiagnosisCode_{i}" for i in range(1, 11)]
    ]
    train_data.drop(columns=[col for col in columns_to_drop if col in train_data.columns], inplace=True, errors="ignore")

    # fix some naming issues
    for i in range(1, 11):
        if f"LONG DESCRIPTION_ClmDiagnosisCode_{i}" in train_data.columns:
            train_data.rename(columns={
                f"LONG DESCRIPTION_ClmDiagnosisCode_{i}": f"ClmDiagnosisCode_{i}_LongDesc",
                f"SHORT DESCRIPTION_ClmDiagnosisCode_{i}": f"ClmDiagnosisCode_{i}_ShortDesc"
            }, inplace=True)

    # fixing naming issue with first column
    if "LONG DESCRIPTION" in train_data.columns:
        train_data.rename(columns={"LONG DESCRIPTION": "ClmDiagnosisCode_1_LongDesc"}, inplace=True)
    if "SHORT DESCRIPTION" in train_data.columns:
        train_data.rename(columns={"SHORT DESCRIPTION": "ClmDiagnosisCode_1_ShortDesc"}, inplace=True)

    # Create flags for unknown procedures and diagnoses
    num_diagnosis_desc = 10

    unknown_procedure_count = (train_data[["PrimaryProcedure_LongDesc", "PrimaryProcedure_ShortDesc"]] == "Unknown").sum(axis=1)
    unknown_diagnosis_count = train_data[[f"ClmDiagnosisCode_{i}_LongDesc" for i in range(1, 11)]].apply(lambda row: (row == "Unknown").sum(), axis=1)

    train_data["Flag_Unknown_Procedures"] = unknown_procedure_count > 1
    train_data["Flag_Unknown_Diagnoses"] = unknown_diagnosis_count > (num_diagnosis_desc / 2)

    train_data.to_csv("data/train_data_cleaned.csv", index=False)