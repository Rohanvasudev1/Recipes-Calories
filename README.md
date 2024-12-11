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


