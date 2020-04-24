import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

# from sklearn import datasets
# iris = datasets.load_iris()
# iris.data
# iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=20)
# print(X_train)

data = pd.read_csv("data.csv")
data.head()
X = data['features'].to_array()
y = data['label'].to_array()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

def test_value_of_k(x_train, y_train, x_test):
    index = []
    accuracy = []
    error = []
    for K in range(30):
        K = K + 1
        knn = KNeighborsClassifier(n_neighbors=K, p=2)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        index.append(K)
        accuracy.append(accuracy_score(y_test, y_pred) * 100)
        error.append(mean_squared_error(y_test, y_pred) * 100)

    plt.subplot(2, 1, 1)
    plt.plot(index, accuracy)
    plt.title('Accuracy')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(index, error, 'r')
    plt.title('Error')
    plt.xlabel('Value of K')
    plt.ylabel('Error')
    plt.show()


test_value_of_k(X_train, y_train, X_test)

#k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))