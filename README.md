# Data Modeling GUI
Application for small data modeling tasks with GUI

![](sample_window.PNG?raw=true)

## Features
With this application the user is able to load data, preprocess and explore it (to some degree), and make predictions for regression or classification tasks.
* Select Feature types (inputs/outputs, numeric/categorical)
* automatic handeling of missing data
* automatic data summary and visualization
* classification and regression tasks (with models below)
* custom model parameters
* data and model export (.csv, .pkl)

### Models
Classifiers
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

Regressors
* Linear Model
* $k$-Nearest Neighbors Regression
* Decision Tree Regression
* Gaussian Process Regression
* Neural Network
* Support Vector Regression
* XGB Regressor

## Possible Future Features 
Better support for categorical data: use scikit ColumnTransfromer! Include more models, add a real progressbars, and add cross validation.

## Dependencies
The following packes are required
* `tkinter`
* `sklearn`
* `xgboost`
* `numpy`
* `pandas`
* `pickle`
* `ctypes`
* `matplotlib`
* `seaborn`
* `pandastable`
* `inspect`
* `webbrowser`

# Feedback is appreciated!
