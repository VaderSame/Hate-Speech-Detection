# Project Overview

Hate speech is an ongoing problem on social media.  The platform providers are obligated to be gatekeepers of this content which requires them to have automated hate speech detection algorithms.  This paper focused on the creation and utilization of a new data set that divided the suspicious texts into three categories:  hate speech, offensive and neither.  The authors created a multi-class classifier to predict the category of the text.  The learnings from where the classifier succeeded and failed will further work in hate speech detection.

# Installation and Setup

To ensure a consistent and isolated environment, you can use Conda to create a virtual environment. Follow these steps:

Open a terminal or command prompt.

Navigate to the project directory.

1. Create a new Conda environment with Python 3.8:
```
conda create --name your_environment_name python=3.8
```
2. Activate the environment:
```
conda activate your_environment_name
```
3. Install the required packages from the `requirements.txt` file:
```
pip install -r requirements.txt
```
4. This will set up a Conda environment with the necessary packages for your project, ensuring consistency and reproducibility.

## Codes and Resources Used
In this section I give user the necessary information about the software requirements.
- **Editor Used:**   Visual Studio Code (VSCode)
- **Python Version:** 3.8.18

## Python Packages Used
In this section, I include all the necessary dependencies needed to reproduce the project, so that the reader can install them before replicating the project. I categorize the long list of packages used as - 
- **General Purpose:** ` urllib ,os,requests `
- **Data Manipulation:**  `pandas, numpy` 
- **Data Visualization:** `seaborn, matplotlib` 
- **Machine Learning:** `scikit`
- **Natural Language Processing:** `nltk, vaderSentiment ,textstat`

# Data
For Training
labelled_data.csv - hate tweets from twitter

For Testing
ETHOS: multi-labEl haTe speecH detectiOn dataSet. This repository contains a dataset for hate speech detection on social media platforms, called Ethos. There are two variations of the dataset:
- Ethos_Dataset_Binary.csv[Ethos_Dataset_Binary.csv] contains 998 comments in the dataset alongside with a label about hate speech *presence* or *absence*. 565 of them do not contain hate speech, while the rest of them, 433, contain. 


# Code structure

```bash
.
├── .gitignore          <- (add your own description)
├── LICENSE             <- (add your own description)
├── main.ipynb          <- (add your own description) main notebook or something
├── Makefile            <- Makefile with commands like `make data` or `make train`
├── README.md           <- The top-level README 'for' developers using this project.
├── data
│   ├── train           <- Data for training
│   └── test            <- Data for testing
│
├── figures             <- Generated graphics and figures to be used in reporting
│
├── models              <- Saved pickled (.pkl) files
│
├── notebooks
│   └── 1.0-jqp-initial-data-exploration.ipynb   <- Jupyter notebook for initial data exploration
│
├── references          <- Manuals and all other explanatory materials
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g., generated with `pip freeze > requirements.txt`
│
├── src
│   ├── __init__.py     <- Makes src a Python module
│   ├── data
│   │   └── make_dataset.py
│   ├── features
│   │   └── build_features.py
│   ├── models
│   │   ├── predict_model.py
│   │   └── train_model.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   ├── models.py
│   │   ├── pos_processing.py
│   │   ├── sentence_cleaner.py
│   │   └── text_processing.py
│   └── visualization
│       └── visualize.py

```

# Results and evaluation

# Results and Evaluation

## Overview

In this project, four machine learning models were employed for training and testing on the dataset. The models used include Linear Regression, Random Forests, Linear Support Vector Classification (LinearSVC), and Multinomial Naive Bayes (MultinomialNB). The evaluation of these models involved the use of relevant metrics and visualizations to assess their performance.

## Model Performance

### Linear Regression

The Linear Regression model was utilized to predict numerical outcomes. Evaluation metrics such as Mean Squared Error (MSE) and R-squared were used to gauge the accuracy of predictions.

### Random Forests

The Random Forests model, known for its ensemble learning capabilities, was employed to handle both classification and regression tasks. Accuracy, precision, recall, and F1-score were key metrics in evaluating the classification performance.

### Linear Support Vector Classification (LinearSVC)

Linear Support Vector Classification is a linear model for classification tasks. Precision, recall, and F1-score were used to assess the model's performance on the classification task.

### Multinomial Naive Bayes (MultinomialNB)

Multinomial Naive Bayes is commonly used for text classification tasks. Evaluation metrics such as precision, recall, and F1-score were utilized to measure its effectiveness.

## Evaluation Methodology

**Training and Testing Split:** The dataset was divided into training and testing sets to train the models on one subset and evaluate their performance on another, ensuring a fair assessment of generalization.

**Confusion Matrix:** Confusion matrices were generated for each model to provide a detailed breakdown of true positive, true negative, false positive, and false negative predictions.

## Visualizations

Confusion matrices were visualized to enhance the interpretation of model performance. The following is an example of a confusion matrix for one of the models:

### Confusion Matrix

The diagonal elements represent the number of correct predictions, while off-diagonal elements indicate misclassifications. This visual representation facilitates a clear understanding of the model's strengths and weaknesses.

The project's results and evaluations collectively showcase the efficacy of the selected models, providing insights into their performance on the given task.



# Future work
From the paper and associated literature review, there is a strong conclusion that the datasets being used at the time of the paper were very different in their labeling process.  This will hugely impact any machine learning classification done from that point on.  Further work in this field should be focused on creating a standard for the labeling process with the optimization of machine learning coming second.  Public hate speech data sets that are currently available are not standardized leading to stagnation in the research.  Private company work inside the social network platforms is likely far ahead of what is publically available.

The second learning is that without a well-laid-out code repository, it is extremely difficult to duplicate the results.  This paper, though being published in a technical journal, was definitely more focused on the social science side of the project than on AI optimization.  A well-laid-out repository explaining all of the code would have made this project more useful and would have allowed us more time to advance the classification.

To achieve the next tier of modeling accuracy, we would have to move to embeddings produced by large language models.  There is a group of papers from 2020 and 2021 that have achieved good results on a variety of data sets that would be useful to build from.

# Acknowledgments/References

Special thanks to the following individuals who have made valuable contributions and provided insights to this project:

- **Thomas Shank**

- **David**

- **Professor Garima:** for her mentorship and valuable feedback throughout the project.

We also acknowledge any additional contributors, data sources, or relevant parties who have played a role in the successful completion of this project.

Thank you for your support and collaboration!

# License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT/).

## Dataset License

Please be aware of the licenses associated with the dataset used in this project. Refer to the dataset documentation or source to understand the terms and conditions under which the dataset is released.

It is important to comply with the dataset's license if you plan to use or contribute to this project.

Thank you for respecting the licenses and terms associated with this project.
