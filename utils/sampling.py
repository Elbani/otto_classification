import numpy as np


# Split dataset on train, test and validate datasets
def sample_dataset(train_dataset):
    train_sample = train_dataset.sample(n=int(len(train_dataset) * 0.60))
    drop_indices = np.random.choice(train_sample.index, int(len(train_sample) * 0.60), replace=False)
    test_validate_sample = train_dataset.drop(drop_indices)

    drop_indices = np.random.choice(test_validate_sample.index, int(len(test_validate_sample) * 0.50), replace=False)
    test_sample = test_validate_sample.loc[drop_indices]
    validate_sample = test_validate_sample.drop(drop_indices)

    return train_sample, test_sample, validate_sample
