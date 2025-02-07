# Stroke Prediction Using Machine Learning

## Overview

This project aims to predict the likelihood of a patient having a stroke based on various health-related parameters. The dataset includes features such as age, gender, medical history, and lifestyle habits. The objective is to develop a machine learning model that can assist in early stroke detection, potentially aiding in preventive healthcare.

This project was completed as an assignment for the course **COMP00172 - Artificial Intelligence for Biomedicine and Healthcare**.

## Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and consists of 5110 patient records with the following attributes:

- **gender**: Male, Female, or Other
- **age**: Patient's age
- **hypertension**: 0 (No) or 1 (Yes)
- **heart_disease**: 0 (No) or 1 (Yes)
- **ever_married**: No or Yes
- **work_type**: Type of employment (e.g., Private, Govt_job, Self-employed)
- **Residence_type**: Rural or Urban
- **avg_glucose_level**: Average glucose level in blood
- **bmi**: Body Mass Index
- **smoking_status**: formerly smoked, never smoked, smokes, or Unknown
- **stroke**: Target variable (0 = No Stroke, 1 = Stroke)

## Files in This Repository

The repository contains the following files:

- **02 Stroke Prediction.pdf**: A document providing context on stroke prediction and dataset details.
- **Stroke Predictor.ipynb**: A Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- **Report.pdf**: A detailed report explaining the methodology, results, and conclusions of the project.
- **02 Stroke Prediction.csv**: The dataset used for training and testing the models.

## Methods

The project explores various machine learning models to classify whether a patient is likely to have a stroke:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest**

### Preprocessing Steps

- Handling missing values (BMI imputation, treating unknown smoking status as a separate category)
- One-hot encoding categorical variables
- Standardizing numerical features
- Splitting the dataset (70% training, 30% testing)

### Model Selection & Tuning

Hyperparameter tuning was performed for each model:

- **KNN**: Number of neighbors, weight function
- **Logistic Regression**: Regularization penalty, max iterations
- **SVM**: Kernel type, regularization parameter
- **Random Forest**: Number of trees, depth, and min samples per split

## Results

- **Logistic Regression** achieved the best F1-score (0.23) and was the most effective model for stroke prediction.
- **KNN, SVM, and Random Forest** struggled due to the dataset's imbalance, often failing to predict stroke cases.
- **Evaluation metrics used**: F1-score, accuracy, precision-recall curves, and ROC curves.

## Future Improvements

- **Enhancing the dataset**: More stroke cases could improve model performance.
- **Exploring deep learning models**: Neural networks may better capture complex relationships.
- **Ensemble techniques**: Combining multiple models might yield better predictions.

## How to Run the Code

### Requirements

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook (optional for running the notebook)

### Running the Model

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stroke-prediction.git
   cd stroke-prediction
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Stroke Predictor.ipynb"
   ```

## References

- [World Health Organization - Stroke](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)
- [Performance Analysis of Machine Learning Approaches in Stroke Prediction](https://ieeexplore.ieee.org/abstract/document/9297525)
- [A Hybrid Machine Learning Approach to Stroke Prediction](https://www.sciencedirect.com/science/article/pii/S0933365719302295)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author:** Giovanni Fantoni

