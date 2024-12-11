# Recipes-Calories
## Introduction 
The aim of this project was to investigate the relationship between various metrics and predict the calories of a meal using a Random Forest Regressor. The dataset used focuses on recipes and includes attributes such as the number of ingredients, steps, and nutritional values like protein, carbohydrates and fat. The objective was to create a model pipeline and evaluate its performance while gaining insights form the data throughout this nporcess by using exploratory data analysis 

Write about dataset we used and reason for research here.

The recipe and interaction datasets were merged on id and recipe id to create a recipes datset with ratings and reviews that were used for advanced analysis. 

The recipes dataset contains 83782 rows, indicating 83782 unique recipes. The data was presented using the following columns:

| Column           | Description                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------|
| `name`           | Name of Recipe                                                                                   |
| `id`             | Recipe ID                                                                                     |
| `minutes`        | Minutes to prepare recipe                                                                      |
| `contributor_id` | User ID who submitted this recipe                                                              |
| `submitted`      | Date recipe was submitted                                                                      |
| `tags`           | Food.com tags for recipe                                                                       |
| `nutrition`      | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `n_steps`        | Number of steps in recipe                                                                      |
| `steps`          | Text for recipe steps, in order                                                                |
| `description`    | User-provided description                                                                      |
| `ingredients`    | Text for recipe ingredients                                                                    |
| `n_ingredients`  | Number of ingredients in recipe                                                                |

The interactions dataset had 731927 rows with each row repsrenting a review. The data was presnted using the following columns:

| Column      | Description          |
|-------------|----------------------|
| `user_id`   | User ID              |
| `recipe_id` | Recipe ID            |
| `date`      | Date of interaction  |
| `rating`    | Rating given         |
| `review`    | Review text          |

The combined dataset had 234,429 rows. The 2 dataframes were merged so that each recipe had a row for each of its rows. 

This project focuses on predicting the calorie content of recipes, based on their number of ingredients, nutrional contents and preparation details. The goal is to make people make healthier meal choices through identifying possible correlations with high calorie meals. 

Calorie prediction is extremely important for people aiming to maintain dietary goals for weight management, athletic performance, or health requirements like diabetes. Given the complex nature of managing diets, any insight is valuable. 

To facilitate this exploration, the merged datset was modified slightly.
The nutrition column was used to create the following columns by splitting up the list: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates`

`submitted` and `date` were also converted to datetime to aid with analysis using the *pd.to_datetime* function . 

Then, a boolean `highprotein` column was created using the mean protein value to aid with future hypothesis testing. 
`protein` was also used to create another column for protein proportion of the whole meal for future analysis 

## Data Cleaning and Exploratory Data Analysis
To merge the dataframes, we took the following steps:
  1. Left merge the recipes and interactions datasets together.
  2. In the merged dataset, fill all ratings of 0 with np.nan. Thsi was done because the rating of a recipe could only 1, 2, 3, 4 or 5. Hence, we counted 0 as a missing value. 
  3. Find the average rating per recipe, as a Series.
  4. This was then used to create a new column so that next to each review of each recipe there was a rating in the rating column and the recipe's average rating as well.

With the other modifications that we discuss in the introduction section, we were left with a dataset head that looked like this. 

*dataset head here*

And the following columns:
| **Column Name**      | **Description**                                                                                              | **How It Was Obtained**                                                                                                                                           |
|-----------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `name`               | The name of the recipe.                                                                                   | Directly from the original dataset.                                                                                                                               |
| `id`                 | The unique identifier for each recipe.                                                                    | Directly from the original dataset.                                                                                                                               |
| `minutes`            | The number of minutes it takes to prepare the recipe.                                                     | Directly from the original dataset.                                                                                                                               |
| `contributor_id`     | The unique identifier of the user who submitted the recipe.                                                | Directly from the original dataset.                                                                                                                               |
| `submitted`          | The date the recipe was submitted.                                                                         | Converted from a string format to `datetime` using `pd.to_datetime()`.                                                                                            |
| `tags`               | A list of tags associated with the recipe, such as "vegan" or "gluten-free".                               | Directly from the original dataset.                                                                                                                               |
| `nutrition`          | A list containing various nutrition values: `[calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates]`. | Directly from the original dataset.                                                                                                                               |
| `calories`           | The calorie content of the recipe.                                                                         | Extracted from the `nutrition` column.                                                                                                                            |
| `total_fat`          | The total fat content (in PDV - Percentage of Daily Value) of the recipe.                                  | Extracted from the `nutrition` column.                                                                                                                            |
| `sugar`              | The sugar content (in PDV) of the recipe.                                                                  | Extracted from the `nutrition` column.                                                                                                                            |
| `sodium`             | The sodium content (in PDV) of the recipe.                                                                 | Extracted from the `nutrition` column.                                                                                                                            |
| `protein`            | The protein content (in PDV) of the recipe.                                                                | Extracted from the `nutrition` column and cleaned (e.g., stripped spaces, converted to float).                                                                     |
| `saturated_fat`      | The saturated fat content (in PDV) of the recipe.                                                          | Extracted from the `nutrition` column.                                                                                                                            |
| `carbohydrates`      | The carbohydrate content (in PDV) of the recipe.                                                           | Extracted from the `nutrition` column and converted to `float`.                                                                                                   |
| `n_steps`            | The number of steps required to prepare the recipe.                                                       | Directly from the original dataset.                                                                                                                               |
| `steps`              | The detailed step-by-step instructions for preparing the recipe.                                           | Directly from the original dataset.                                                                                                                               |
| `description`        | The user-provided description of the recipe.                                                              | Directly from the original dataset.                                                                                                                               |
| `ingredients`        | A list of ingredients required for the recipe.                                                            | Directly from the original dataset.                                                                                                                               |
| `n_ingredients`      | The number of ingredients required for the recipe.                                                        | Directly from the original dataset.                                                                                                                               |
| `prop_protein`       | The proportion of protein relative to the recipe's calorie content.                                        | Computed as `protein / calories` (calories converted to `float`).                                                                                                 |
| `highprotein`        | A binary indicator of whether the recipe is high-protein (`True` if `prop_protein` > mean proportion, `False` otherwise). | Computed by comparing `prop_protein` to the dataset’s average protein proportion.                                                                                 |
| `date`               | The interaction date or rating date.                                                                      | Converted from a string format to `datetime` using `pd.to_datetime()`.                                                                                            |
| `rating`             | The rating given to the recipe by users (on a scale).                                                     | Directly from the interactions dataset and replaced `0` values with `NaN`.                                                                                        |
| `review`             | The text of user reviews for the recipe.                                                                  | Directly from the interactions dataset.                                                                                                                           |
### Univariate analysis

First, univariate analysis was performed on the protein proportion column to see the distribution of protein percentage in the dataset. This clearly demonstartes that most meals have a fairly low proportion of protein

<img width="903" alt="image" src="https://github.com/user-attachments/assets/645a0bd7-e71b-4cb3-a06c-68ccef703e86">
*BAR PLOT HERE*

Next, we performed univariate analysis on the n-steps column to see its distribution as well. This demonstrates that most recipes don't have that many steps and suggests an overall lesser complexity than I initially thought

*BAR PLOT HERE*

We also looked at the distirbution of average rating. You can clearly see that there were significantly mor higher rated meals 

*BAR Plot here*

### Bivariate Analysis 

The bivariate analysis we decided to perform was on the prop_protein column and the rating column. We created the follwing bar chart, which suggests that most meals in genral were higher protein. It also suggest that most meals, high or low protein were given a 5 star rating, which perhaps suggested a flaw in the rating column. 

*Insert Bar Chart here*

### Interesting Aggregates

| Rating | High Protein | Count | Proportion |
|--------|--------------|-------|------------|
| 1.0    | False        | 517   | 0.006480   |
| 1.0    | True         | 289   | 0.003622   |
| 2.0    | False        | 506   | 0.006342   |
| 2.0    | True         | 307   | 0.003848   |
| 3.0    | False        | 1683  | 0.021093   |
| 3.0    | True         | 1297  | 0.016256   |
| 4.0    | False        | 8915  | 0.111734   |
| 4.0    | True         | 7112  | 0.089136   |
| 5.0    | False        | 36191 | 0.453590   |
| 5.0    | True         | 22971 | 0.287900   |

This table aggregates the data by the rating and if a recipe is "high protein." It shows the frequency of recipes (`count`) and their corresponding proportion (`proportion`) within the dataset. This enables us to identify the wya in which high-protein recipes correlate with higher ratings, providing insights into user preferences.

#### Line Plot: Proportion of Protein by Number of Steps

```markdown
**Proportion of Protein by Number of Steps**
```
This visualization shows the mean, median, max, and min proportion of protein as the number of recipe steps increases. It displays insights into whether complex recipes (with more steps) tend to have higher or lower protein proportions. The pattern of decreasing maximum protein proportion could indicate protein being a less complex dish to make.

## Assessment of Missingness

### NMAR Analysis
One column that was NMAR was `review` was the missingness of that data column was dependent on the quality of the review as people have a tendency to put in effort when they enjoyed something. If they didn't enjoy it, they will not want to give the recipe, that they wasted their time on, more time. Hence, many of the less positive reviews would be missing. 

### Missingness Dependency 











