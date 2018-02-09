from os import path

from sklearn import tree, metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from k_medoids import KMedoids
import graphviz
import numpy as np

# path of the iris data file
iris_file_path = path.join(
    path.dirname(path.abspath(__file__)),
    'iris.data.txt'
)

iris_data = ''
with open(iris_file_path) as iris_file:
    iris_data = iris_file.read()

# contains attributes with the class itself
iris_data = [data.split(',') for data in iris_data.split('\n')][:-2]

# convert numeric values from string to float
for data in iris_data:
    for i in range(4):
        data[i] = float(data[i])

# contains attributes
iris_training_data = [data[:-1] for data in iris_data]

# contains class labels of training values
iris_training_labels = [data[-1] for data in iris_data]

clf = tree.DecisionTreeClassifier()
clf.fit(iris_training_data, iris_training_labels)

iris_feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris_labels_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# output the generated tree classifier
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=iris_feature_names,
                                class_names=iris_labels_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('graphviz/output1/iris')

# predict the training the data using the generated model
predicted_labels = clf.predict(iris_training_data)

print('------------------------------------------------------------')
print('                       Classification                       ')
print('------------------------------------------------------------')
print()
print("Decision tree with use training set:")
confusion_matrix = metrics.confusion_matrix(iris_training_labels, predicted_labels, labels=iris_labels_names)
print("    TP Rate (Sum): ", confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])
print("    ────────────────────────────────")
print("    FP Rate (Sum): ",
      confusion_matrix[1][0] + confusion_matrix[2][0] +
      confusion_matrix[0][1] + confusion_matrix[2][1] +
      confusion_matrix[0][2] + confusion_matrix[1][2]
      )
print("    ────────────────────────────────")
print("    Precision Micro-average: ", metrics.precision_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))
print("    ────────────────────────────────")
print("    Recall Micro-average: ", metrics.recall_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))
print("    ────────────────────────────────")
print("    F-measure Micro-average: ", metrics.f1_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))

# change iris training data to NumPy array
iris_training_data_np = np.array(iris_training_data)

# change iris training data labels to NumPy array
iris_training_labels_np = np.array(iris_training_labels)

fold_10 = KFold(n_splits=10)

# Below values will contain each fold's measures
tp_rate = []
fp_rate = []
precision = []
recall = []
f_measure = []

# For each fold
for train, test in fold_10.split(iris_training_data):
    train_set = iris_training_data_np[train]
    train_set_labels = iris_training_labels_np[train]

    test_set = iris_training_data_np[test]
    test_set_labels = iris_training_labels_np[test]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_set, train_set_labels)

    predicted_labels = clf.predict(test_set)

    confusion_matrix = metrics.confusion_matrix(test_set_labels, predicted_labels, labels=iris_labels_names)

    tp_rate.append(confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])

    fp_rate.append(confusion_matrix[1][0] + confusion_matrix[2][0] +
                   confusion_matrix[0][1] + confusion_matrix[2][1] +
                   confusion_matrix[0][2] + confusion_matrix[1][2])

    precision.append(metrics.precision_score(
        y_true=test_set_labels,
        y_pred=predicted_labels,
        labels=iris_labels_names,
        average='micro'
    ))
    recall.append(metrics.recall_score(
        y_true=test_set_labels,
        y_pred=predicted_labels,
        labels=iris_labels_names,
        average='micro'
    ))
    f_measure.append(metrics.f1_score(
        y_true=test_set_labels,
        y_pred=predicted_labels,
        labels=iris_labels_names,
        average='micro'
    ))

print('------------------------------------------------------------')
print("Decision tree with Cross-validation:")
print("    TP Rate (Sum): ", np.average(tp_rate))
print("    ────────────────────────────────")
print("    FP Rate (Sum): ", np.average(fp_rate))
print("    ────────────────────────────────")
print("    Precision Micro-average: ", np.average(precision))
print("    ────────────────────────────────")
print("    Recall Micro-average: ", np.average(recall))
print("    ────────────────────────────────")
print("    F-measure Micro-average: ", np.average(f_measure))

clf = AdaBoostClassifier()
clf.fit(iris_training_data, iris_training_labels)

# predict the training the data using the generated model
predicted_labels = clf.predict(iris_training_data)

print('------------------------------------------------------------')
print("Ada boost:")
confusion_matrix = metrics.confusion_matrix(iris_training_labels, predicted_labels, labels=iris_labels_names)
print("    TP Rate (Sum): ", confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])
print("    ────────────────────────────────")
print("    FP Rate (Sum): ",
      confusion_matrix[1][0] + confusion_matrix[2][0] +
      confusion_matrix[0][1] + confusion_matrix[2][1] +
      confusion_matrix[0][2] + confusion_matrix[1][2]
      )
print("    ────────────────────────────────")
print("    Precision Micro-average: ", metrics.precision_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))
print("    ────────────────────────────────")
print("    Recall Micro-average: ", metrics.recall_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))
print("    ────────────────────────────────")
print("    F-measure Micro-average: ", metrics.f1_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))

clf = RandomForestClassifier()
clf.fit(iris_training_data, iris_training_labels)

# predict the training the data using the generated model
predicted_labels = clf.predict(iris_training_data)

print('------------------------------------------------------------')
print("Random Forest:")
confusion_matrix = metrics.confusion_matrix(iris_training_labels, predicted_labels, labels=iris_labels_names)
print("    TP Rate (Sum): ", confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2])
print("    ────────────────────────────────")
print("    FP Rate (Sum): ",
      confusion_matrix[1][0] + confusion_matrix[2][0] +
      confusion_matrix[0][1] + confusion_matrix[2][1] +
      confusion_matrix[0][2] + confusion_matrix[1][2]
      )
print("    ────────────────────────────────")
print("    Precision Micro-average: ", metrics.precision_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))
print("    ────────────────────────────────")
print("    Recall Micro-average: ", metrics.recall_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))
print("    ────────────────────────────────")
print("    F-measure Micro-average: ", metrics.f1_score(
    y_true=iris_training_labels,
    y_pred=predicted_labels,
    labels=iris_labels_names,
    average='micro'
))

print()
print('------------------------------------------------------------')
print('                         Clustering                         ')
print('------------------------------------------------------------')
print()
print("K-Means:")
for k in range(2, 6):
    k_means = KMeans(n_clusters=k)
    k_means.fit(iris_training_data)
    predicted_labels = k_means.predict(iris_training_data)
    print("    k={}: {}".format(
        k,
        metrics.adjusted_mutual_info_score(iris_training_labels, predicted_labels)
    ))

print('------------------------------------------------------------')
print("K-Medoids:")
for k in range(2, 6):
    k_medoids = KMedoids(n_clusters=k)
    k_medoids.fit(iris_training_data)
    predicted_labels = k_medoids.predict(iris_training_data)
    print("    k={}: {}".format(
        k,
        metrics.adjusted_mutual_info_score(iris_training_labels, predicted_labels)
    ))
