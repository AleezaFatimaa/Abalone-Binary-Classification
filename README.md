# Abalone Binary Classification using Linear Regression

## Project Overview

This project demonstrates how Linear Regression can be adapted for binary classification problems using thresholding. We use the **Abalone dataset** to predict whether an abalone is above or below the median age.

The workflow includes data preprocessing, **model training**, applying a threshold to convert regression predictions into binary class labels, evaluating the model using both regression and classification metrics, and visualizing the results.

Graphs and visualizations are included in the repository to help interpret model performance.

## Dataset

* Source: [UCI Abalone Dataset]([https://archive.ics.uci.edu/ml/datasets/Abalone](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)
* Features: Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight
* Target: Rings (converted to Age = Rings + 1.5)
* Binary target: Age above median = 1, Age below median = 0

## Preprocessing Steps

1. Load dataset using Pandas.
2. Convert `Rings` to `Age`.
3. Create binary target `Age_Binary` based on median age.
4. Convert categorical feature `Sex` into numeric via OneHotEncoding.
5. Split dataset into training and testing sets (80% / 20%).

## Model Training

1. Build a pipeline with preprocessing and Linear Regression.
2. Fit the model on the training data to learn the relationship between features and age.
3. Predict continuous age on the test set.
4. Apply a threshold (median age) to convert regression predictions into binary labels.

## Model

* **Type:** Linear Regression
* **Purpose:** Predict continuous age and classify binary age using thresholding.
* **Threshold:** Median age used to convert regression predictions into 0/1 labels.

## Evaluation Metrics

### Regression Metrics

* Mean Squared Error (MSE)
* R² Score

### Classification Metrics

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve and AUC

## Visualizations

The following visualizations are included in the `images/` folder of the repository:

### Confusion Matrix

![Confusion Matrix]([images/confusion_matrix.png](https://1drv.ms/i/c/f5dcf9917a077858/IQDBs4qoget1RL5nMdg0STUgAepOWJ9Ed3ocfE4scxKwz6s?e=GyeCpA)

### ROC Curve

![ROC Curve]([images/roc_curve.png](https://1drv.ms/i/c/f5dcf9917a077858/IQCdSzrVRUjjQaDCZY_A2a_aAQRVfaHsyhIM8yBjwl_gmKE?e=X3HeWs)

### Classification Metrics

![Classification Metrics]([images/classification_metrics.png](https://1drv.ms/i/c/f5dcf9917a077858/IQBdB5H-DzMeRblhjikT3YtyARUH_Yvt4EmlsmXIOnh1y5E?e=ElXkLp)

(Additional visualizations, such as predicted vs actual age, can also be added.)

## Results

* Regression:

  * MSE ≈ 4.89
  * R² ≈ 0.55
* Classification:

  * Accuracy ≈ 0.76
  * Precision ≈ 0.70
  * Recall ≈ 0.91
  * F1 Score ≈ 0.79
  * Confusion matrix shows more false positives than false negatives, consistent with high recall.

 ## Tools Used

Python – Core programming language

Google Colab – Cloud-based notebook environment for development and execution

Pandas – Data loading and manipulation

NumPy – Numerical computations

Scikit-learn – Model training, preprocessing, and evaluation metrics

Matplotlib – Plotting graphs and visualizations

Seaborn – Enhanced statistical visualizations (confusion matrix heatmap)

## Conclusion

This project demonstrates that Linear Regression can be adapted for binary classification using thresholding. While not optimized for classification, the approach provides meaningful results and can serve as a baseline before trying logistic regression or other classifiers.

## How to Run

1. Open [Google Colab](https://colab.research.google.com) or any Python environment.
2. Install required packages (if not already installed): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
3. Run the notebook containing the code.
4. All metrics, visualizations, and model training steps will be executed automatically.

## Author

Aleeza Fatima- BAI243024

