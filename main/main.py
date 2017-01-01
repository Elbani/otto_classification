import utils.data_manipulation as data_manipulation
import ml.random_forest as random_forest

# Example of how to run Random Forest.
train, validate, train_labels, validate_labels = data_manipulation.get_data()
print random_forest.run_random_forest_multiclass_classification(train, train_labels, validate, validate_labels)
