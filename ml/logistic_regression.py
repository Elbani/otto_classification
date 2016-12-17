from sklearn import ensemble, feature_extraction, linear_model, preprocessing
from sklearn import metrics
from sklearn.linear_model.logistic import LogisticRegression


def run_logistic_regression_multiclass_classification(train, train_labels, validate, validate_labels):
    logisticReg = LogisticRegression()
    logisticReg.fit(train, train_labels)
    predicted_labels = logisticReg.predict(validate)
    return metrics.accuracy_score(validate_labels, predicted_labels)


def run_logistic_regression_bagging_probabilistic_classification(train, train_labels, validate, validate_labels):
    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
    train = tfidf.fit_transform(train).toarray()
    validate = tfidf.transform(validate).toarray()

    # encode labels
    label_encode = preprocessing.LabelEncoder()
    train_labels = label_encode.fit_transform(train_labels)

    linear_classifier = linear_model.LogisticRegression(C=1, penalty='l1',
                                                 fit_intercept=True, random_state=23)

    classifier = ensemble.BaggingClassifier(base_estimator=linear_classifier, n_estimators=500,
                                     max_samples=1., max_features=1., bootstrap=True,
                                     n_jobs=5, verbose=True, random_state=23)
    classifier.fit(train, train_labels)
    predicted_labels = classifier.predict_proba(validate)
    return metrics.log_loss(validate_labels, predicted_labels)
