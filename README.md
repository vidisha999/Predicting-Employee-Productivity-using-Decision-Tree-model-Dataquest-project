# Predicting-Employee-Productivity-using-Decision-Tree-model-Dataquest-project

## Project Focus 

The goal of this project is to use Tree models in  predicting the productivity of employees of a garment factory. Both Decision Tree and Random Forest models were used for this task. The prediction of the productivity can be meassured as a classification or regression task by using the above tree models. When using as a classification purpose, the predictivity which is the target variable should be transformed in to different classes in the range of (0-1), while when using as a regression purpose, the predictivity can be predicted in the range of 0-1.

## Background 

The Decision Tree and Random Forest Tree models are both supervised learning algorithms, which uses labellebd data to predict to make predictions. The Decison Tree is a single tree structure that splits data based on feature values to make predictions. A Random Forest Tree model is a ensambled collection of  multiple decision trees which were often trained on bootstraped subsets of data. Random Forest tree method considered more robust and accurate compared to a single decision tree as it reduces the overfitting by  averaging the predictions of the multiple trees.

## Data

The dataset for this projec is obtained from the [UCI machine learning repository](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees). The [dataset](Data/garments_worker_productivity.csv) contains several feature columns and the target variable `actual_productivity` which represents the actual percentage of the productivity that was delivered by the workers in the range of 0-1. 

## Exploratory Data Analysis 
The following insights were made by performing the EDA on the dataset:

- The feature columns `wip`,`over_time`,`incetive`,`idle_time`,`idle_men`,`no_of_workers` have very high maximum values which are way greater than than 75% percentile of the data, representing the presence of outliers in the data.
- `wip` and `over_time `columns also have very less minimum values which are too lesser than the 25% percentile of data, convincing the  presence of strong  outliers.
- The average `over-time` is quite high and the maximum `over-time` value is hugely dispropotionate with average value indicating workers have to often work outside of regular hours to acheive targets.
- `Quarter` column includes 5 quarters, in which the **quarter5** represents the last two dates of January which were  29 th and 31st but not the 30th Friday.
- Looking at the `day` columns, it includes days of the week, and contains no **Fridays** on the entire dataset, which explains why **quarter5** has omitted 30 th January which is a Friday.
- So that, Friday could be a company closed day or a all employee holiday.
- Analyzing the `incetive` column, which represents the financial incetive to motivates the action shows 50% of the data has no incetive to perform actions.
- When comparing the `actual productivity` with the `targeted productivity`, there are observations with both instances where actual production was higher as well as lower than the target production for each team for each day,however 73% of the data have actual productivity greater than the targeted productivity.
- 96% of times, when the `targeted productivity` was acheived or passed, employees had to work overtime to acheive the goal.
- [Actual Vs Targeted plot](Images/decision_tree_image1.png) shows time periods when the `actual productivity` acheived or passed the `targeted productivity` and when it did not.

