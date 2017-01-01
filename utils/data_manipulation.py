import pandas as pd

import sampling

allData = pd.DataFrame.from_csv("../data/train.csv");
# testData = pd.DataFrame.from_csv("test.csv");

train_sample, test_sample, validate_sample = sampling.sample_dataset(allData)


def get_data():
    train = train_sample.loc[:, 'feat_1':'feat_93']
    validate = validate_sample.loc[:, 'feat_1':'feat_93']
    validate_labels = validate_sample['target']
    train_labels = train_sample['target']
    return train, validate, train_labels, validate_labels
