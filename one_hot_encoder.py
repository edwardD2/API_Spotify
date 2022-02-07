import pandas as pd


def one_hot_encoder(features, values):
    ohe = pd.DataFrame(columns=features)
    feature_dict = {}
    for feature in features:
        if feature in values:
            feature_dict[feature] = 1
        else:
            feature_dict[feature] = 0

    ohe = ohe.append(feature_dict, ignore_index=True)

    return ohe
