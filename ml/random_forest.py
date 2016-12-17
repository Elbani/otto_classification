from sklearn import feature_extraction, preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def run_random_forest_multiclass_classification(train, train_labels, validate, validate_labels):
    poly_feat = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    train = poly_feat.fit_transform(train, train_labels)
    validate = poly_feat.transform(validate)

    randomForest = RandomForestClassifier(n_jobs=4, n_estimators=1000, max_features=20, min_samples_split=3,
                                          bootstrap=False, verbose=3, random_state=23, max_depth=100)
    randomForest.fit(train, train_labels)
    predicted_labels = randomForest.predict(validate)
    return metrics.accuracy_score(validate_labels, predicted_labels)


def run_random_forest_probabilistic_classification(train, train_labels, validate, validate_labels):
    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
    train = tfidf.fit_transform(train).toarray()
    validate = tfidf.transform(validate).toarray()

    # encode labels
    label_encode = preprocessing.LabelEncoder()
    train_labels = label_encode.fit_transform(train_labels)

    poly_feat = preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
    train = poly_feat.fit_transform(train, train_labels)
    validate = poly_feat.transform(validate)

    randomForest = RandomForestClassifier(n_jobs=4, n_estimators=1000, max_features=20, min_samples_split=3,
                                          bootstrap=False, verbose=3, random_state=23, max_depth=100)
    randomForest.fit(train, train_labels)
    predicted_labels = randomForest.predict_proba(validate)
    return metrics.log_loss(validate_labels, predicted_labels)
