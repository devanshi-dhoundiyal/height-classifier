import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
data = pd.read_csv("height_dataset.csv")

# 2. Separate input features and target
X = data[["Age", "Gender", "Height_cm"]]
y = data["Tall_or_Not"]


# 3. Convert Gender into numerical values
# Female = 0, Male = 1
X = X.copy()
X["Gender"] = X["Gender"].map({
    "Female": 0,
    "Male": 1
})


# 4. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Create the Decision Tree classifier
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)


# 6. Train the classifier
model.fit(X_train, y_train)


# 7. Make predictions on the test data
y_pred = model.predict(X_test)


# 8. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Results ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


# 9. Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. Test the model with new data
new_students = pd.DataFrame({
    "Age": [20, 25, 18],
    "Gender": [0, 1, 0],
    "Height_cm": [170, 180, 150]
})

predictions = model.predict(new_students)

print("\n--- Predictions for New Data ---")

for i in range(len(new_students)):
    print(
        "Age:", new_students.iloc[i]["Age"],
        "| Gender:", "Female" if new_students.iloc[i]["Gender"] == 0 else "Male",
        "| Height:", new_students.iloc[i]["Height_cm"], "cm",
        "=>", predictions[i]
    )
