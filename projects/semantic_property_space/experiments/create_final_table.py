""" In a format feature | cos | f1-neigh | f1-lr | f1-net | type """

import pandas as pd
import os

# read evaluation data for each feature
folder = "../evaluation"
paths = os.listdir(folder)

model_name = "google_news"

final_df = []

for file in paths:
    data = pd.read_csv(f"{folder}/{file}")
    feature = file.replace(".csv", "")
    # if len(data) != 3:
    #    continue

    """ test """
    data = data[data.iloc[:, 0].str.contains(model_name)]

    nearest_neighbors = None
    neural_net_classification = None
    logistic_regression = None

    try:
        nearest_neighbors = data[data.iloc[:, 0].str.contains('nearest_neighbors')]['f1'].iloc[0]
    except Exception as e:
        print('nearest_neighbors', feature)

    try:
        neural_net_classification = data[data.iloc[:, 0].str.contains('neural_net_classification')]['f1'].iloc[0]
    except Exception as e:
        print('neural_net_classification', feature)

    try:
        logistic_regression = data[data.iloc[:, 0].str.contains('logistic_regression')]['f1'].iloc[0]
    except Exception as e:
        print('logistic_regression', feature)

    final_df.append([feature, nearest_neighbors, neural_net_classification, logistic_regression])

pre_table = pd.DataFrame(final_df, columns=["feature", "f1-neigh", "f1-lr", "f1-net"])

cosine_data = pd.read_csv(f"../results/cosine_distances_pairs/{model_name}.txt")

final_table = pre_table.merge(cosine_data, left_on='feature', right_on='feature')

final_table.to_csv("final_table.csv")

print("DONE")
