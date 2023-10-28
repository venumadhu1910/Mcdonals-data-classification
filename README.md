# McDonald's Dataset Analysis             
                                                                      

This document provides a summary of the analysis performed on the McDonald's dataset. The dataset contains various features related to McDonald's menu items, such as serving size, ingredients, nutritional information, and more. The analysis includes the application of different machine learning models to predict the 'energy' content of menu items based on other attributes.

Analysis Details
Data Preprocessing
The 'serving_size' column was converted to a numeric format (grams) for consistency.
The columns 'ingredients', 'tag', 'allergy', and 'description' were omitted from the analysis, as more advanced text preprocessing was required, which is not covered in this example.
Machine Learning Models
Three machine learning models were applied to predict 'energy':

Linear Regression:

Mean Squared Error: 239.30
R-squared Score: 0.9942
Cross-Validation MSE: 297.22
Decision Tree Regression:

Mean Squared Error: 10,019.94
R-squared Score: 0.7551
Cross-Validation MSE: 6,987.28
Random Forest Regression:

Mean Squared Error: 5,331.74
R-squared Score: 0.8697
Cross-Validation MSE: 4,007.14
Summary
Linear Regression produced the lowest Mean Squared Error (MSE), indicating a strong predictive performance. The R-squared score suggests that this model explains approximately 99.42% of the variance in 'energy'.
Decision Tree Regression and Random Forest Regression performed well, but their MSE was higher than that of Linear Regression. Decision Tree showed lower predictive power, with an R-squared score of 0.7551. Random Forest improved the results, with an R-squared score of 0.8697.
Cross-validation MSE values suggest that Linear Regression may have slightly overfit the data, as the validation MSE is higher than the test MSE. Decision Tree and Random Forest performed more consistently.



Pothireddy Venumadhavi.

