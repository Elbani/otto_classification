from sklearn import feature_extraction, preprocessing
from sklearn import metrics
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


def run_decision_tree_multiclass_classification(train, train_labels, validate, validate_labels):
    decisionTree = DecisionTreeClassifier()
    decisionTree.fit(train, train_labels)
    predicted_labels = decisionTree.predict(validate)
    return metrics.accuracy_score(validate_labels, predicted_labels)


def run_decision_tree_probabilistic_classification(train, train_labels, validate, validate_labels):
    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
    train = tfidf.fit_transform(train).toarray()
    validate = tfidf.transform(validate).toarray()

    # encode labels
    label_encode = preprocessing.LabelEncoder()
    train_labels = label_encode.fit_transform(train_labels)

    decisionTree = ExtraTreesClassifier(n_jobs=4, n_estimators=1000, max_features=20, min_samples_split=3,
                                        bootstrap=False, verbose=3, random_state=23)
    decisionTree.fit(train, train_labels)
    predicted_labels = decisionTree.predict_proba(validate)
    print "Extra Trees Classifier LogLoss"
    print str(metrics.log_loss(validate_labels, predicted_labels))
