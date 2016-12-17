from sklearn.model_selection import cross_val_score
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn import metrics


def run_knn_multi_level_classifier(train, train_labels):
    k_range = list(range(2, 5))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, train, train_labels, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    return k_scores


def run_knn_probabilistic_classifier(train, train_labels, validate, validate_labels):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train, train_labels)
    predicted_labels = knn.predict_proba(validate)
    return metrics.log_loss(validate_labels, predicted_labels)
