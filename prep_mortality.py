import pandas as pd
import numpy as np

from mimic3benchmark.readers import InHospitalMortalityReader

HEADERS = [
    "Index", "Hours", "Capillary refill rate", "Diastolic blood pressure", 
    "Fraction inspired oxygen", "Glascow coma scale eye opening",
    "Glascow coma scale motor response", "Glascow coma scale total", 
    "Glascow coma scale verbal response", "Glucose", "Heart Rate",
    "Height", "Mean blood pressure", "Oxygen saturation", "Respiratory rate", 
    "Systolic blood pressure", "Temperature", "Weight", "pH", "Label"
]

COMA_SCALE_EYE_OPENING_REPLACEMENTS = {
    "1 No Response": 1,
    "None": 1,
    "2 To pain": 2,
    "To Pain": 2,
    "3 To speech": 3,
    "To Speech": 3,
    "4 Spontaneously": 4,
    "Spontaneously": 4,
}
COMA_SCALE_MOTOR_REPLACEMENTS = {
    "1 No Response": 1,
    "No response": 1,
    "2 Abnorm extensn": 2,
    "Abnormal extension": 2,
    "3 Abnorm flexion": 3,
    "Abnormal Flexion": 3,
    "4 Flex-withdraws": 4,
    "Flex-withdraws": 4,
    "5 Localizes Pain": 5,
    "Localizes Pain": 5,
    "6 Obeys Commands": 6,
    "Obeys Commands": 6
}
COMA_SCALE_VERBAL_REPLACEMENTS = {
    "No Response-ETT": 0,
    "1.0 ET/Trach": 0,
    "1 No Response": 1,
    "No Response": 1,
    "2 Incomp sounds": 2,
    "Incomprehensible sounds": 2,
    "3 Inapprop words": 3,
    "Inappropriate Words": 3,
    "4 Confused": 4,
    "Confused": 4,
    "5 Oriented": 5,
    "Oriented": 5,
}

def preprocess_coma_scales(data):
    to_replace = {
        "Glascow coma scale eye opening":
            COMA_SCALE_EYE_OPENING_REPLACEMENTS,
        "Glascow coma scale motor response":
            COMA_SCALE_MOTOR_REPLACEMENTS,
        "Glascow coma scale verbal response":
            COMA_SCALE_VERBAL_REPLACEMENTS,
    }
    coma_scale_columns = list(to_replace.keys())
    coma_scales = data[coma_scale_columns]
    coma_scales = coma_scales.astype(str)
    coma_scales = coma_scales.replace(
        to_replace=to_replace
    )
    # coma_scales = coma_scales.astype(float)
    data = data.copy()
    data[coma_scale_columns] = coma_scales
    return data

def preprocess(
    train_dir="data/in-hospital-mortality/train",
    test_dir="data/in-hospital-mortality/test",
    split=False,
):
    train_reader = InHospitalMortalityReader(
        dataset_dir=train_dir,
        listfile=f"{train_dir}/listfile.csv"
    )
    test_reader = InHospitalMortalityReader(
        dataset_dir=test_dir,
        listfile=f"{test_dir}/listfile.csv"
    )

    train_data = []
    test_data = []

    for i in range(train_reader.get_number_of_examples()):
        data = train_reader.read_example(i)
        index = np.array([[i] * data["X"].shape[0]]).T
        label = np.array([[data["y"]] * data["X"].shape[0]]).T
        tmp = np.concatenate((data["X"], label), axis=1)
        out = np.concatenate((index, tmp), axis=1)
        train_data.append(out)

    for j in range(test_reader.get_number_of_examples()):
        data = test_reader.read_example(j)
        index = np.array([[i+j] * data["X"].shape[0]]).T
        label = np.array([[data["y"]] * data["X"].shape[0]]).T
        tmp = np.concatenate((data["X"], label), axis=1)
        out = np.concatenate((index, tmp), axis=1)
        test_data.append(out)

    # Stack training data and testing data
    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    if split:
        # Create dataframe
        train_df = pd.DataFrame(train_data, index=None, columns=HEADERS)
        test_df = pd.DataFrame(test_data, index=None, columns=HEADERS)
        # Preprocess coma scales
        train_df = preprocess_coma_scales(train_df)
        test_df = preprocess_coma_scales(test_df)
        return train_df, test_df

    else:
        # Create dataframe
        all_data = np.cat(X)
        df = pd.DataFrame(all_data, index=None, columns=HEADERS)
        # Preprocess coma scales
        df = preprocess_coma_scales(df)
        return df

if __name__ == "__main__":
    df = preprocess(
        train_dir="data/in-hospital-mortality/train",
        test_dir="data/in-hospital-mortality/test",
        split=False,
    )
    df.to_csv("mortality.csv")

