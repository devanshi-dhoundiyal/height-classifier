## Height Classification using Machine Learning

# Project Overview

This project demonstrates a simple machine learning classification system using Python. The model predicts whether a person is Tall or Not Tall based on their age, gender, and height.

The dataset used in this project is a small synthetic dataset created for educational purposes.

# Dataset

The dataset contains 50 records with the following features:

Age – Age of the person
Gender – Gender of the person
Height_cm – Height in centimeters
Tall_or_Not – Target variable indicating whether the person is Tall or Not Tall

# Machine Learning Algorithm

A Decision Tree Classifier is used for this project.

A Decision Tree works by learning decision rules from the training data and using those rules to classify new data.

# Methodology

The project follows these steps:

1. Load the dataset using Pandas.
2. Separate the input features from the target variable.
3. Convert the Gender column into numerical values.
4. Divide the dataset into training and testing sets.
5. Train a Decision Tree Classifier using the training data.
6. Use the trained model to make predictions on the testing data.
7. Calculate the accuracy of the classifier.
8. Test the model using new sample data.

# Input Features

The model uses:
Age
Gender
Height in centimeters

## Target

The target variable is:
Tall
Not Tall

# Train-Test Split

The dataset is divided into:

80% training data
20% testing data

The training data is used to build the model, while the testing data is used to evaluate its performance.

# Model Evaluation

The accuracy of the classifier is calculated using the `accuracy_score` function from Scikit-learn.

The program displays the training samples, testing samples, accuracy percentage, classification report, and predictions for new data.

# Technologies Used

Python
Pandas
Scikit-learn
Decision Tree Classifier

# Files in the Repository

 File                 - Description                                          
 `height_dataset.csv` - Dataset used for training and testing                
 `classifier.py`      - Python program containing the machine learning model 
 `README.md`          - Project documentation                                

# How to Run

Make sure Python and the required libraries are installed.

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Then run:

```bash
python classifier.py
```

# Conclusion

This project demonstrates how a small dataset can be used to build a machine learning classifier using Python. The Decision Tree model learns patterns from the given data and predicts whether a person belongs to the Tall or Not Tall category.
