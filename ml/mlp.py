from sklearn import feature_extraction, preprocessing
from sklearn import metrics
from sklearn.neural_network import MLPClassifier


# Multi-Layer Perceptron
def mlp_multiclass_classifier(train, train_labels, validate, validate_labels):
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(2000,), random_state=1)
    clf.fit(train, train_labels)
    predicted_labels = clf.predict(validate)
    print "Neural Network"  # 0.835673164327 for 1000 hidden layers
    print str(metrics.accuracy_score(validate_labels, predicted_labels))


def mlp_probabilistic_classifier(train, train_labels, validate, validate_labels):
    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer(smooth_idf=False)
    train = tfidf.fit_transform(train).toarray()
    validate = tfidf.transform(validate).toarray()

    # encode labels
    label_encode = preprocessing.LabelEncoder()
    train_labels = label_encode.fit_transform(train_labels)

    clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(1000,), random_state=1)
    clf.fit(train, train_labels)
    predicted_labels = clf.predict_proba(validate)
    print "Neural Network LogLoss"  # 0.835673164327 for 1000 hidden layers
    print str(metrics.log_loss(validate_labels, predicted_labels))
