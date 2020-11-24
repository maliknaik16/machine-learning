
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Get the iris dataset.
iris = load_iris()

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize the classifier.
classifier = DecisionTreeClassifier()
# classifier = RandomForestClassifier()
# classifier = KNeighborsClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data.
predictions = classifier.predict(X_test)

# Calculate and print the accuracy score.
print(accuracy_score(y_test, predictions))
