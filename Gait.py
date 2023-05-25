import preproc
import featGait
import classifierRF
import pandas as pd
import os
import json
import pickle
from sklearn.preprocessing import MinMaxScaler

# Import demographic data
Demog_raw = pd.read_csv('Demographics.csv')
# Cleaning
Demog = preproc.preproc_demog(Demog_raw)


# Load walking descriptive table
Walking_raw = pd.read_csv('Walking.csv')
# Cleaning based on demographics
Walking = preproc.demog_gait(Demog, Walking_raw)

# Mpower folder
data_folder = 'C:\\Users\\maria\\Desktop\\escritorio\\github\\mPower\\mPower_BIDS'# Path to the walking data folder

# Calculate features for walking outbound and return
modes = ["deviceMotion_walking_outbound", "deviceMotion_walking_return"]

featuresGait = pd.DataFrame()
demo_walk = pd.DataFrame() # I will temporarily save the demographic data of the ones under consideration
for mode in modes:
    for index, row in Walking.iterrows():
        try:
            health_code = row['healthCode']
            created_on = row['createdOn_y']

            folder_path = os.path.join(data_folder, health_code, mode, str(created_on))

            if os.path.exists(folder_path):
                file_list = os.listdir(folder_path)

                if file_list:
                    file_name = file_list[0]
                    file_path = os.path.join(folder_path, file_name)

                    # File exists
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # GAIT PREPROCESSING
                    # flag indicating signal less than 5 seconds
                    acc, fs, flag = preproc.linearacceleration(data)

                    # GAIT FEATURE EXTRACTION
                    if flag == 0:
                        demo_walk = pd.concat([demo_walk, Walking.loc[index].to_frame().transpose()], ignore_index=True)
                        features = featGait.features(acc, fs)
                        featuresGait = pd.concat([featuresGait, features], ignore_index=True)

        except Exception as e:
            # Exception occurred, print the index and row
            print("Error occurred at mode:", mode)
            print("Error occurred at index:", index)
            print("Error occurred in row:", row)
            print("Error message:", str(e))


# Normalize features
scaler = MinMaxScaler()
numeric_columns = featuresGait.select_dtypes(include=['float64', 'int64']).columns
scaler.fit(featuresGait[numeric_columns])
# Transform the features to their normalized values
featuresGait[numeric_columns] = scaler.transform(featuresGait[numeric_columns])
featuresGait = pd.DataFrame(featuresGait, columns=featuresGait.columns)

featuresGait = pd.concat([demo_walk, featuresGait], axis=1)

##### EXPERIMENTS #####
# All recordings, no matching
resultsRF_all_nomatch = classifierRF.RF(featuresGait, 0, 0)

# First recording, no matching
resultsRF_onerec_nomatch = classifierRF.RF(featuresGait, 1, 0)

# All recordings, matching
resultsRF_all_match = classifierRF.RF(featuresGait, 0, 1)

# First recording, matching
resultsRF_onerec_match = classifierRF.RF(featuresGait, 1, 1)


# Visualize data
# List of dictionaries with the results
results = [resultsRF_all_nomatch, resultsRF_onerec_nomatch, resultsRF_all_match, resultsRF_onerec_match]
titles = ["All & no matched", "One record & no matched", "All & matched", "One record & matched"]

# Print the table headers
headers = ["Experiment", "Accuracy", "Sensitivity", "Specificity", "AUC"]
print("{:<25} {:<12} {:<12} {:<12} {:<12}".format(*headers))

for i, (result, title) in enumerate(zip(results, titles), start=1):
    accuracy = result["accuracy"]
    sensitivity = result["sensitivity"]
    specificity = result["specificity"]
    auc = result["auc"]

    row = [title, accuracy, sensitivity, specificity, auc]
    print("{:<25} {:<12.2f} {:<12.2f} {:<12.2} {:<12.2}".format(*row))