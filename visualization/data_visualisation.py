import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

train = pd.DataFrame.from_csv("../data/train.csv");
# test = pd.DataFrame.from_csv("test.csv");


# Plot summation for every product feature
def product_feature_summation_plot():
    sum_otto = train.sum()
    sum_otto = sum_otto.drop(['target']).order()
    ploti = sum_otto.plot(kind='barh', figsize=(15, 20))


def label_count_plot():
    # map each class to numerical value from 0 to 8(i.e. 9 classes)
    range_of_classes = range(1, 10)
    map_values_dic = {}

    for n in range_of_classes:
        map_values_dic['Class_{}'.format(n)] = n - 1

    train['target'] = train['target'].map(map_values_dic)

    # Plot
    sns.countplot(x='target', data=train)
