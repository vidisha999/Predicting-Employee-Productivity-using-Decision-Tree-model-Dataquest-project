# Predicting-Employee-Productivity-using-Decision-Tree-model-Dataquest-project

## Project Focus 

The goal of this project is to use Tree models in  predicting the productivity of employees of a garment factory. Both Decision Tree and Random Forest models were used for this task. The prediction of the productivity can be meassured as a classification or regression task by using the above tree models. When using as a classification purpose, the predictivity which is the target variable should be transformed in to different classes in the range of (0-1), while when using as a regression purpose, the predictivity can be predicted in the range of 0-1.

## Background 

The Decision Tree and Random Forest Tree models are both supervised learning algorithms, which uses labellebd data to predict to make predictions. The Decison Tree is a single tree structure that splits data based on feature values to make predictions. A Random Forest Tree model is a ensambled collection of  multiple decision trees which were often trained on bootstraped subsets of data. Random Forest tree method considered more robust and accurate compared to a single decision tree as it reduces the overfitting by  averaging the predictions of the multiple trees.

## Data

The dataset for this projec is obtained from the [UCI machine learning repository](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees). The [dataset](Data/garments_worker_productivity.csv) contains several feature columns and the target variable `actual_productivity` which represents the actual percentage of the productivity that was delivered by the workers in the range of 0-1. 

## Exploratory Data Analysis 
The following insights were made by performing the EDA on the dataset:
### I. Exploring the Data
- The feature columns `wip`,`over_time`,`incetive`,`idle_time`,`idle_men`,`no_of_workers` have very high maximum values which are way greater than than 75% percentile of the data, representing the presence of outliers in the data.
- `wip` and `over_time `columns also have very less minimum values which are too lesser than the 25% percentile of data, convincing the  presence of strong  outliers.
- The average `over-time` is quite high and the maximum `over-time` value is hugely dispropotionate with average value indicating workers have to often work outside of regular hours to acheive targets.
- `Quarter` column includes 5 quarters, in which the **quarter5** represents the last two dates of January which were  29 th and 31st but not the 30th Friday.
- Looking at the `day` columns, it includes days of the week, and contains no **Fridays** on the entire dataset, which explains why **quarter5** has omitted 30 th January which is a Friday.
- So that, Friday could be a company closed day or all employee holiday.
- Analyzing the `incetive` column, which represents the financial incetive to motivates the action shows 50% of the data has no incetive to perform actions.
- When comparing the `actual productivity` with the `targeted productivity`, there are observations with both instances where actual production was higher as well as lower than the target production for each team for each day,however 73% of the data have actual productivity greater than the targeted productivity.
- 96% of times, when the `targeted productivity` was acheived or passed, employees had to work overtime to acheive the goal.
- [Actual Vs Targeted plot](Images/decision_tree_image1.png) shows time periods when the `actual productivity` acheived or passed the `targeted productivity` and when it did not.
### II. Data Cleaning 
- During the data cleaning phase, **quarter5** group was replaced as **quarter4** , because there were only few distinct values in the **quarter5** and having a seperate quarter for last two days of January doesn't add any logical explanation on data analysis and focus of the project. 
- As shown in the [Notebook](Notebook/vidisha_decision_tree.ipynb) after data cleaning process, final set of columns are listed as 'quarter', 'day', 'department', 'team', 'no_of_workers', 'over_time','no_of_style_change','smv', 'incentive', 'targeted_productivity','target_achieved'.
- The target variable `target_acheived` is a new column which was created to acheive the classification task of the project by using the `actual_productivity` and `targeted_productivity` columns.

 ### III.Feature Engineering 

- In the feature selection phase, [boxplot diagram](Images/decision_tree_images2.png) was plotted to understand the relationship between the categorical columns with the targeted variable and a [scatterplot](Images/decision_tree_images3.png) was plotted to explore the relationship between the numerical variables with the target variable.
  - Based on the above plots, it can be seen the categorical columns `day` and `department` had no impact on the `targeted_productivity`.
  - Average productivity was less in last two quarters compared to the first two quarters.
  - With the increase of `no_of_style_change` the targeted productivity has decreased.
  - None of the numerical columns show significant impact on the targeted productivity.
- As the final set of features, `department`, `day`, `team`, `no-of-style-change`,`no_of_workers`,`over_time`,`smv`,`incentive` and `quarter` columns were selected.
 
### IV.Feature Transformation 
Prior feeding the data to the machine learning model, the selected categorical columns were transformed into numerical columns. 
The following code was used for the transformation of features : 
```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
transform_cols=make_column_transformer((OneHotEncoder(),['department','day','team','no_style_of_change']),
                                       (OrdinalEncoder(),['quarter']),remainder='passthrough',
                                        verbose_feature_names_out=False)
column_names=transform_cols.get_feature_names_out()
transform_df=transform_cols.fit_transform(clean_df)
final_df=pd.DataFrame(transform_df,cols=column_names)
```

## Model Building 

The decision tree models can be used for two scenarios. 
 - The **DecisionTreeClassifier** model could be used to classify if the target variable `target_acheived` was achieved or not.
 - The **DecisionTreeRegressor** model could be used to predict the continuous value of target variable `targeted_productivity`.
 - The focus of this project is framed to use **DecisionTreeClassifier** for the classification task of target variable.

Using the following code the DecisionTreeClasifier model was built: 

```python
from sklearn.tree import DecisionTreeClassifier,plot_tree
X=final_df.drop(columns=['targeted_acheived','targeted_productivity'])
y=final_df['targeted_productivity']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=72)
class_tree=DecisionTreeClassifier(criterion='gini',random_state=72,max_depth=3)
class_tree.fit(X_train,y_train)
```
## Model Optimization and Evaluation 

### I.Minimum Cost Complexity Pruning 
- The traning accuracy of the model was 82.3 % and the test accuracy of the model was 78.75 %. In order to improve the model's accuracy a post-pruning technique called **minimum cost complexity pruning** was used.
- The minimum cost complexity pruning creates a diverse subtrees and finds the subtree with the highest scoring accuracy.


The goal of using this cost path is to create diverse subtrees and to find the subtree with the highest scroing accuracy.
The alpha value of the subtree with highest scoring accuracy is the optimal alpa value which is used to add a penalty for subtrees that overfits the data.

The follwoing code is used to build the subset of pruned trees: 
```python
unpruned_tree=DecisionTreeClassifier()
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=72, test_size=0.2)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2*X.shape[0]/X_train.shape[0],random_state=72)

cost_path=unpruned_tree.cost_complexity_pruning_path(X_train,y_train)
subtrees=[]
for alpha in cost_path.ccp_alphas:
    subtree=DecisionTreeClassifier(ccp_alpha=alpha)
    subtree.fit(X_train,y_train)
    subtrees.append(subtree)
subtree_scores=[subtree.score(X_val,y_val) for subtree in subtrees]
 ```   

The [subtree scores Vs. alpha plot](Images/decision_tree_image4.png) shows how the scoring accuracy of each subtree varies with their corresponding alpha values. 

The subtree with the highest scoring accuracy and its alpha value is used to find out the best tree and tested on the test dataset to determine the test accuracy as shown in the below code: 

```python
sklearn.metrics import classification_report
alpha_scores=tuple(zip(cost_path.ccp_alphas,subtree_scores))
val_accuracy=max(alpha_scores,key=lambda x: x[1]) # selects the second element of the tuple object

best_score_index=np.argmax(subtree_scores)
best_pruned_tree=subtrees[best_score_index)
test_score=best_pruned_tree.score(X_test,y_test)
y_predict=best_pruned_tree.predict(y_test)
class_report=classification_report(y_test,y_predict)
```
- Both the validation accuracy and test accuracy of the tree model was 81.6% which is higher than test accuracy of the unpruned tree.
- Based on the classification report the recall or the sensitivity of the model is 96% which is an excellent value.
- It is further shown by the [confusion matrix](Images/decision_tree_image5.png), indicating the model is great at classifying positive classes which shows when the `targeted_productivity` was achieved.

The [graphical representation of the Decision Tree](Images/decision_tree_image6.png) drawn on the best pruned decision tree shows `incentive`,no_of_workers`,`smv` are the most impactful attributes on the `targeted_productivity` as they appear multiple times on the decision tree diagram contributing the decision made at the terminal leaf.

The [feature importance plot](Images/decision_tree_image7.png) shows `incentive` has the highest impact on the productvity of the production process along with `smv` and `no_of_workers`. `day_Monday` and `over_time` also contributed a certain amount of impact on the productivity of employess while rest of the attributes had caused no impact at all. 

### II.RandomForestClassifier model

For the purpose of selecting the optimal model, the  random forest tree model which is an ensemble decison tree model was built:
```python
random_tree=RandomForestClassifier(oob_score=True,random_state=42)
random_tree.fit(X_train,y_train)
rf_y_predict=random_tree.predict(X_test)
rf_accuracy_score=random_tree.score(y_test,y_predict)
```
The accuracy score obtained was around 80% which is less than value gained from the best pruned decision tree. So, that retraining the RandomForestClassifier with only using the highly impacted features selected from the analysis  DecisionTreeClassifier, could improve the test accuracy.

## Conclusion 

One of the most important takeaways from this project is the importance of feature selection and preprocessing in improving model performance. Another important takeaway is that while ensemble methods such as random forests can improve the accuracy of a model, they are not always necessary. In this project, we found that the decision tree model was already quite robust and that the random forest model did not significantly outperform it in terms of accuracy.









































































