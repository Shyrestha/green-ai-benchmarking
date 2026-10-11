import os
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

process = psutil.Process(os.getpid())

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = SVC()

memory_before = process.memory_info().rss / (1024 * 1024)

model.fit(X_train, y_train)

memory_after = process.memory_info().rss / (1024 * 1024)

memory_increase = memory_after - memory_before

print("SVM")
print("Memory before:", memory_before, "MB")
print("Memory after:", memory_after, "MB")
print("Memory increase:", memory_increase, "MB")