# Day 2 – Python Libraries & Machine Learning Workflow

## Introduction

Day 2 focused on understanding essential Python libraries used in Data Science and Machine Learning: NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn. We also implemented a basic Machine Learning workflow using a student performance dataset.

---

## Why NumPy?

### Python Lists vs NumPy Arrays

Python lists:
- High memory usage
- Slow operations (manual loops)
- No vectorized math

NumPy arrays:
- Low memory usage
- Very fast (runs in C)
- Supports vectorized operations

### Example

```python
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

a + b

Performs element-wise addition automatically (vectorization).

Advantages of NumPy
	•	Faster calculations
	•	Less memory consumption
	•	Mathematical operations
	•	Linear algebra support

In ML:
	•	Data → matrices
	•	Features → vectors
	•	Calculations → NumPy

⸻

Why Pandas?

Pandas simplifies data analysis.

Without Pandas:
	•	Manual file reading
	•	Loops for calculations
	•	Difficult null handling

With Pandas:

import pandas as pd

df = pd.read_csv("students.csv")
df.describe()

Pandas Benefits
	•	DataFrame structure (Excel-like)
	•	Column access by name
	•	Built-in statistics
	•	Easy cleaning

Used in ML for:
	•	Data loading
	•	Preprocessing
	•	Feature selection
	•	Exploratory Data Analysis

⸻

Visualization – Matplotlib & Seaborn

Visualization converts numbers into graphs.

Matplotlib

Low-level plotting library.

Seaborn

High-level statistical visualization (built on Matplotlib).

Charts used:
	•	Bar Chart → Compare values
	•	Line Chart → Trends
	•	Scatter Plot → Relationships

Example:

import seaborn as sns

sns.scatterplot(x="Study_Hours", y="Test_Score", data=df)


⸻

Correlation

Correlation measures relationship between variables.

Example:
Study hours ↑ → Test score ↑

Important for prediction.

⸻

Function, Module, Package

Function:
Reusable block of code.

Module:
Single Python file.

Package:
Collection of modules.

Examples:
NumPy, Pandas are packages.

⸻

Machine Learning Using Scikit-Learn

Scikit-Learn is used to build ML models.

⸻

Typical ML Workflow

Step 1 – Load Data

df = pd.read_csv("students.csv")


⸻

Step 2 – Separate Features and Target

X = df[['Study_Hours']]
y = df['Test_Score']

X → Input
y → Output

⸻

Step 3 – Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

Used to evaluate model on unseen data.

⸻

Step 4 – Choose Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()


⸻

Step 5 – Train Model

model.fit(X_train, y_train)

Model learns weights.

⸻

Step 6 – Make Predictions

y_pred = model.predict(X_test)


⸻

Step 7 – Evaluate Model

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)

Lower error = better model.

⸻

transform()

Used to scale or normalize data before training.

⸻

Key Libraries
	•	NumPy → Mathematical operations
	•	Pandas → Data handling
	•	Matplotlib / Seaborn → Visualization
	•	Scikit-Learn → Machine Learning

⸻

What I Learned Today
	•	Difference between Python lists and NumPy arrays
	•	Data handling using Pandas
	•	Visualization using Matplotlib and Seaborn
	•	Complete ML workflow
	•	Importance of train-test split
	•	How models learn from data

⸻

Reflection

Day 2 Morning Sessioin built the technical foundation required for Machine Learning. Understanding these libraries is essential before moving into advanced ML algorithms.


# Day 2 – Afternoon Session  

## Linear Regression & Model Evaluation Fundamentals

## Introduction

The afternoon session introduced Machine Learning fundamentals using Linear Regression. We learned how models make predictions, how features and targets work, and how to evaluate model performance. We also explored overfitting, underfitting, and bias–variance tradeoff using real-life examples.


## What is a Machine Learning Model?

A Machine Learning model is a mathematical function that learns patterns from data and uses those patterns to make predictions on new inputs.

Example:

House Size → Price  
Study Hours → Marks  
TV Ads → Sales  

The model learns the relationship between input and output.


## Feature and Target

### Feature (X)

Input variables used to make predictions.

Examples:
- Study Hours
- House Size
- TV Advertising Spend

### Target (y)

Output variable the model predicts.

Examples:
- Marks
- House Price
- Sales

In code:
X = df[['Study_Hours']]
y = df['Test_Score']


⸻

Linear Regression – Ravi Story

Ravi tracks his study hours and marks:

Hours	Marks
1	35
2	40
3	45
4	50
5	55

Linear Regression finds the best straight line through these points.

Equation:

y = mx + c

Where:
	•	y = predicted output
	•	x = input feature
	•	m = slope (rate of increase)
	•	c = intercept (baseline value)

From class:

Marks = 5 × Hours + 30

If Ravi studies 6 hours:

Marks = 5 × 6 + 30 = 60

Linear Regression draws a smart line to predict future values.

⸻

Definition of Linear Regression

Linear Regression is a supervised learning algorithm used to predict continuous values by modeling the relationship between features and target using a straight line.

⸻

Goal of Linear Regression

To minimize prediction error using a cost function such as:

Mean Squared Error (MSE)

Lower MSE means better model performance.

⸻

When to Use Linear Regression
	•	Output is continuous
	•	Relationship is linear
	•	Simple interpretable model required

⸻

When NOT to Use
	•	Relationship is non-linear
	•	Classification problems
	•	Highly correlated features

⸻

Typical Machine Learning Workflow (from Colab)

Step 1 – Load Data

df = pd.read_csv()


⸻

Step 2 – Define X and y

X = df[['TV']]
y = df['Sales']


⸻

Step 3 – Train-Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

Used to test model on unseen data.

⸻

Step 4 – Choose Model

from sklearn.linear_model import LinearRegression
model = LinearRegression()


⸻

Step 5 – Train Model

model.fit(X_train, y_train)

Model learns weights.

⸻

Step 6 – Make Predictions

y_pred = model.predict(X_test)


⸻

Step 7 – Evaluate Model

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


⸻

Visualization

Scatter plots show:
	•	Data distribution
	•	Linear trend
	•	Prediction line

Used to understand correlation visually.

⸻

Overfitting vs Underfitting (Exam Analogy)

Underfitting

Model too simple.
	•	Performs poorly on training
	•	Performs poorly on testing

Error Type: High Bias

⸻

Overfitting

Model memorizes training data.
	•	Excellent training performance
	•	Poor testing performance

Error Type: High Variance

⸻

Balanced Model

Good understanding + practice.
	•	Good training accuracy
	•	Good testing accuracy

Low Bias + Low Variance

⸻

Bias vs Variance

Bias: Model is too simple
Variance: Model is too complex

Goal: Balance both for best generalization.

⸻

Real World Applications of Linear Regression
	•	House price prediction
	•	Student performance prediction
	•	Sales forecasting
	•	Salary estimation
	•	Healthcare recovery time
	•	Stock trend analysis

⸻

Key Takeaways
	•	ML models learn relationships from data
	•	Features are inputs, target is output
	•	Linear Regression predicts continuous values
	•	Overfitting memorizes, underfitting oversimplifies
	•	Bias–Variance balance is critical
	•	Model evaluation uses Mean Squared Error
	•	Visualization helps understand data behavior

⸻

Reflection

Day 2 Afternoon session introduced the first real Machine Learning algorithm. Understanding Linear Regression, model workflow, and evaluation metrics forms the foundation for advanced ML techniques.
