# ml-linear-regression-20240926
 Machine Learning Series ``` 🎓 ```

## ``` 📑 ``` Table of Contents

1. [Introduction to Linear Regression](#introduction-to-linear-regression)
   - Definition
   - Use Cases
2. [Mathematical Foundations](#mathematical-foundations)
   - Equation of Linear Regression
   - Dependent and Independent Variables
   - Types of Linear Regression
3. [Assumptions of Linear Regression](#assumptions-of-linear-regression)
4. [Fitting a Linear Regression Model](#fitting-a-linear-regression-model)
   - Ordinary Least Squares (OLS)
   - Gradient Descent
   - Why is the OLS method limited compared to the Gradient Descent?
5. [Evaluation Metrics](#evaluation-metrics)
   - R-squared
   - Adjusted R-squared
   - Mean Absolute Error (MAE) and Mean Squared Error (MSE)
6. [Interpreting Coefficients](#interpreting-coefficients)
7. [Model Diagnostics](#model-diagnostics)
   - Residual Analysis
   - Multicollinearity
   - Homoscedasticity and Normality Tests
8. [Common Issues and Solutions](#common-issues-and-solutions)
   - Overfitting
   - Underfitting
9. [Conclusion](#conclusion)
10. [Further Reading/Resources](#further-readingresources)

---

➥ _**Implementing Linear Regression (.ipynb)**_
   - [Code Example Using sklearn LinearRegression model (OLS based) →](ml-linear-regression-ols-20240926.ipynb) 
   - [Code Example Using sklearn SGDRegresser model (Gradient Descent based) →](ml-linear-regression-sgd-20240926.ipynb) 

---

## ✦ [Introduction to Linear Regression](#introduction-to-linear-regression)

<p align="center">
  <img src="https://github.com/user-attachments/assets/e47a7943-6fd4-4664-9d29-27c2c13bafaa" alt="lrmeme" width="500"/>
</p>

### ```ⓘ``` Definition
 - Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The primary goal is to predict the value of the dependent variable based on the values of the independent variables by fitting a linear equation to the observed data.
 - Linear regression is a supervised learning technique because it uses labeled data to learn a function that maps input features to the target variable.

### Use Cases
```Linear Regression``` is widely used across various fields for its simplicity and interpretability. Some common use cases include:

- 🏠︎ **Predicting Housing Prices**: Estimating the price of a house based on features such as size, location, and number of bedrooms.
- 💲**Sales Forecasting**: Analyzing historical sales data to forecast future sales based on marketing spend or economic indicators.
- ❗**Risk Assessment**: Evaluating risk in finance by modeling the relationship between investment variables and returns.
- ⚕️**Healthcare Analysis**: Predicting patient outcomes based on factors like age, weight, and pre-existing conditions.

Linear Regression serves as a fundamental building block in machine learning and data analysis, making it essential for anyone looking to understand predictive modeling.

## ```૮₍•᷄ ࡇ •᷅₎ა``` [Mathematical Foundations](#mathematical-foundations)

Understanding the mathematical foundations of linear regression is crucial for effectively applying the technique in real-world scenarios. This section covers the essential components, including the equation, dependent and independent variables, and the types of linear regression.

### Equation of Linear Regression
At its core, linear regression seeks to establish a relationship between the dependent variable $\( y \)$ and one or more independent variables $\( x \)$. The equation of a simple linear regression model is given by:

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

Where:
- $\( y \)$: The predicted value of the dependent variable.
- $\( \beta_0 \)$: The y-intercept, representing the predicted value of $\( y \)$ when $\( x = 0 \)$.
- $\( \beta_1 \)$: The coefficient of the independent variable $\( x \)$, indicating the change in $\( y \)$ for a one-unit increase in $\( x \)$.
- $\( \epsilon \)$: The error term (the difference between the actual and predicted values), capturing the variability in $\( y \)$ that cannot be explained by the linear relationship with $\( x \)$.

For multiple linear regression, the equation extends to accommodate multiple independent variables:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon
$$

Where $\( x_1, x_2, \ldots, x_n \)$ are the independent variables.

### Dependent and Independent Variables
In the context of linear regression:

- **Dependent Variable (Response Variable)**: This is the variable we aim to predict or explain. It is dependent on the values of the independent variables.
- **Independent Variables (Predictors or Features)**: These are the variables used to predict the dependent variable. They are considered independent because their values do not depend on other variables in the model.

### Types of Linear Regression
There are two main types of linear regression:

1. **Simple Linear Regression**:
   - Involves a single independent variable.
   - It aims to find the best-fitting line through the data points in a two-dimensional space (one predictor).

<p align="center">
  <img src="https://www.scribbr.com/wp-content/uploads//2020/02/simple-linear-regression-graph.png" alt="Simple Linear Regression Graph" width="50%" />
</p>

2. **Multiple Linear Regression**:
   - Involves two or more independent variables.
   - It models the relationship between the dependent variable and multiple predictors, which can lead to more accurate predictions when relationships are complex.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:597/1*RqL8NLlCpcTIzBcsB-3e7A.png" alt="Simple Linear Regression Graph" width="35%" />
</p>

## [Assumptions of Linear Regression](#assumptions-of-linear-regression)

To ensure that our linear regression model provides valid and reliable results, we must consider the following assumptions:

### ✔ Linearity
   - The relationship between the independent (predictor) variables $\(X\)$ and the dependent (response) variable $\(Y\)$ is linear. This means that the change in $\(Y\)$ can be expressed as a linear function of $\(X\)$. In mathematical terms, we can represent this relationship as:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

   - A linear relationship implies that for each unit increase in $\(X\)$, $\(Y\)$ increases or decreases by a constant amount $(\(\beta_1\))$. 

---

### ✔ Independence
   - The observations (data points) are independent of each other. This means that the value of one observation should not influence or be influenced by another observation. In statistical terms, this can be expressed as:

$$
P(A \cap B) = P(A) \cdot P(B)
$$
     
where:
   - $\(P(A \cap B)\)$ is the probability of both events $\(A\)$ and $\(B\)$ occurring,
   - $\(P(A)\)$ is the probability of event $\(A\)$,
   - $\(P(B)\)$ is the probability of event $\(B\)$.

```🛈``` For linear regression, independence of observations ensures that the residuals (the differences between observed and predicted values) do not show any patterns or correlations. If there is dependence among observations, it can lead to underestimation of the variability of the estimates, resulting in less reliable predictions. 

---

### ✔ Homoscedasticity
The variance of the residuals (the differences between observed and predicted values) is constant across all levels of the independent variables. The spread of the residuals should be roughly the same for low, medium, and high predicted values.

<h4><u><strong><em>🔹Understanding Homoscedasticity Through a House Pricing Analogy</em></strong></u></h4>

Imagine you’re evaluating the prices of houses in a neighborhood. Each house has a predicted price based on features like size, number of bedrooms, and location. The difference between the actual selling price and the predicted price represents the **error** in our pricing model.

<h4><u><strong><em>🔹Consistency in Prices</em></strong></h4>

Now, let’s say for smaller houses, the actual prices tend to be close to the predicted prices. However, for larger houses, the actual prices sometimes vary widely from the predicted prices. This indicates that the “error” changes depending on the size of the house. What we want to see is a situation where, regardless of whether you are predicting the price of a small or large house, the errors in pricing remain consistent.

``` 💡 This idea is what we call **homoscedasticity**! ```

<h4><u><strong><em>🔹What Is Homoscedasticity?</em></strong></h4>

In simple terms, homoscedasticity means that the variance (or spread) of your errors remains constant, no matter the predicted price of the house. In our mathematical model of this pricing scenario, we can express it as follows:

1. **Linear Equation**:
   When we describe this relationship mathematically, we say:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

   - $\(Y\)$ is the actual price of the house.
   - $\(X\)$ is the predicted price based on the features.
   - $\(\beta_0\)$ represents the average expected price of a house (the y-intercept).
   - $\(\beta_1\)$ tells us how much the expected price changes with each unit change in the features (the slope).
   - $\(\epsilon\)$ is the error term, representing how far off our predicted price is from the actual price.

2. **Constant Variance**:
   Homoscedasticity means that the error $\((\epsilon)\)$ has a constant spread, regardless of the predicted price of the house. Mathematically, this is expressed as:

$$
\text{Var}(\epsilon | X) = \sigma^2
$$

   - Here, $\(\text{Var}(\epsilon | X)\)$ means "the variance of the error when we look at the prices."
   - $\(\sigma^2\)$ is a constant number representing how much the errors spread out from the expected price (the predicted price).

<h4><u><strong><em>🔹Visual Representation</em></strong></h4>

To better understand this concept, let’s take a look at a graph that illustrates both homoscedasticity and heteroscedasticity:

<p align="center">
  <img src="https://lh3.googleusercontent.com/B2hT0jQlT2xw6-pRWlqMktNDhiteFjk32W13_stPWUU72uaMOxIKGDqhGOzS1x48rl1vMWF72x08x34xnuHueiJ2YcQZHqTpT9jYU_iENLlV9RfJ5nAaWOELMOUEUJJ1ATkJ1E01z6mpI0Ko" />
</p>

In the image, you can see how in homoscedasticity, the errors are randomly scattered around zero with a consistent spread, whereas in heteroscedasticity, the errors vary in spread, making it difficult to predict outcomes.

---
### ✔ Normality of Residuals
The residuals of the model should be approximately normally distributed. When plotted, the residuals should form a bell-shaped curve around zero.

In the context of house prices, imagine you're predicting the prices of homes based on various features like size, location, and number of bedrooms. After building your model, you compare the predicted prices to the actual prices, resulting in residuals (the differences between predicted and actual values).

For the model to be valid, these residuals should ideally follow a normal distribution. This means that most of the residuals are close to zero (indicating accurate predictions), while fewer residuals are far from zero (indicating occasional significant prediction errors). 

Mathematically, we can express the normality assumption as:

> **Residual Distribution**:
   If we denote the residuals as $\( \epsilon \)$, we expect:

$$ 
\epsilon \sim N(0, \sigma^2)
$$

   - Here, $\( N(0, \sigma^2) \)$ represents a normal distribution with a mean of zero and a constant variance $\( \sigma^2 \)$.
   - A normal distribution of residuals indicates that our predictions are generally accurate, with random variations that don’t systematically skew higher or lower.

---

### ✔ No Multicollinearity
The independent variables should not be highly correlated with each other. Multicollinearity can make it difficult to determine the individual effect of each predictor variable.

Continuing with the house prices analogy, consider that you’re trying to predict home prices based on features like square footage, number of bathrooms, and the neighborhood quality. If square footage and the number of bathrooms are highly correlated (for example, larger homes tend to have more bathrooms), it becomes challenging to understand how each feature independently influences the home price.

When multicollinearity is present, it can inflate the variance of the coefficient estimates, leading to unreliable predictions and interpretations. This means that if you were to increase the number of bathrooms in a home, you might not clearly see how that change impacts the price due to its correlation with square footage.

We can mathematically express this by examining the variance inflation factor (VIF):

> **Variance Inflation Factor**:
   A VIF value can be calculated for each predictor variable as:

$$ 
\text{VIF}(X_i) = \frac{1}{1 - R^2_i}
$$

   - Where $\( R^2_i \)$ is the coefficient of determination obtained from regressing the $\( i \)$-th variable against all other independent variables.
   - A VIF value greater than 10 typically indicates high multicollinearity, suggesting that the model may struggle to discern the individual contributions of correlated variables.

```🛈``` ensuring normality of residuals helps confirm that predictions are reliable and unbiased, while avoiding multicollinearity allows us to accurately assess the impact of each feature on house prices.

---

## [Fitting a Linear Regression Model](#fitting-a-linear-regression-model)

 Fitting a linear regression model refers to the process of finding the best-fit line that describes the relationship between the independent (predictor) variables and the dependent (response) variable. This is done by adjusting the model parameters (coefficients) to minimize the difference between the predicted values and the actual observed values in the data. This process is typically executed when calling the `train()` method on the model.

### 1. Ordinary Least Squares (OLS)

**What It Is:**
Ordinary Least Squares (OLS) is a method used to estimate the coefficients of a linear regression model. The goal of OLS is to find the line (or hyperplane in multiple dimensions) that minimizes the sum of the squared differences (residuals) between the observed values and the values predicted by the model.

**How It Works:**
- Imagine you have a dataset with several houses, and you want to predict their prices based on their features (like size, number of bedrooms, etc.).
- You can represent this relationship mathematically as:

$$ 
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon 
$$

  - $\(Y\)$ is the house price you want to predict.
  - $\(X_1, X_2, ..., X_n\)$ are the features (e.g., size, number of bedrooms).
  - $\(\beta_0\)$ is the intercept (the predicted price when all features are zero).
  - $\(\beta_1, \beta_2, ..., \beta_n\)$ are the coefficients that represent the effect of each feature on the house price.
  - $\(\epsilon\)$ is the error term (the difference between actual and predicted prices).

- For **Simple Linear Regression**, where there is one independent variable, the equation simplifies to:

$$ 
Y = \beta_0 + \beta_1 X + \epsilon 
$$

- The coefficients can be calculated using the formulas:

  - **Intercept $(\(\beta_0\))$**:

 $$ 
 \beta_0 = \bar{Y} - \beta_1 \bar{X} 
 $$

  - **Slope $(\(\beta_1\))$**:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{White}\hat{\beta}_1=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}" alt="\hat{\beta}_1=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sum_{i=1}^{n}(x_i-\bar{x})^2}" />
</p>

  where $\(\bar{y}\)$ and $\(\bar{y}\)$ are the means of the dependent and independent variables, respectively, and $\(n\)$ is the number of observations.

- For **Multiple Linear Regression**, the coefficients can be estimated using matrix algebra as:

$$ 
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} 
$$

  where:
  - $\(\boldsymbol{\beta}\)$ is the vector of coefficients.
  - $\(\mathbf{X}\)$ is the matrix of independent variables.
  - $\(\mathbf{Y}\)$ is the vector of dependent variable values.

---

### 2. Gradient Descent

**What It Is:**
Gradient Descent is an optimization algorithm used to minimize the cost function (in this case, the sum of squared residuals). Instead of calculating all coefficients at once, Gradient Descent iteratively adjusts them to find the best fit.

**How It Works:**

<p align="center">
  <img src="https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/gradient-descent.png" alt="GD" width="500" />
</p>

- You start with random values for the coefficients $\(\beta_0, \beta_1, ..., \beta_n\)$.
- You calculate the cost function (the sum of squared residuals) based on these initial values:

$$ 
\text{Cost} = \sum (Y_i - \hat{Y}_i)^2 
$$

  where $\(Y_i\)$ is the actual price, and $\(\hat{Y}_i\)$ is the predicted price.

- Then, for each coefficient, you calculate the gradient (the slope of the cost function) to see how to adjust the coefficients to reduce the cost.

- You update the coefficients in the direction of the steepest descent (the negative gradient) using a learning rate $(\(\alpha\))$, which determines how big of a step to take:

$$ 
\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} \text{Cost} 
$$

- This process is repeated until the changes in the cost function become very small (indicating convergence), often expressed as:

$$ 
|\text{Cost}_{t+1} - \text{Cost}_t| < \epsilon 
$$

  where $\(\epsilon\)$ is a small threshold.

  ```🛈``` Finally & as an example, the loss function should be reduced as follow:

<p align="center">
  <img src="https://developers.google.com/static/machine-learning/crash-course/linear-regression/images/convergence.png" alt="GD" width="500" />
</p>

```🔎``` We will be going deeper into gradient descent as an optimization algorithm in a separate documentation. This will cover its applications, variations, and practical implementations in more detail.

### 3. Why is the OLS method limited compared to the Gradient Descent?
⚠️ Ordinary Least Squares (OLS) requires the entire dataset to be loaded into memory because it calculates the optimal coefficients by solving a system of linear equations based on all available data points. This makes OLS impractical for larger datasets, as memory constraints can become a significant issue. In contrast, Gradient Descent doesn’t require the entire dataset in memory at once because it processes the data in smaller, manageable chunks (mini-batches) during each iteration. This allows it to scale more effectively to larger datasets.

⚠️ In addition to memory limitations, OLS has other constraints. Mathematically, it assumes that the relationship between the independent and dependent variables is linear, which may not always hold true in real-world scenarios. OLS is also sensitive to outliers, as they can disproportionately affect the calculated coefficients. Moreover, OLS requires the independent variables to be uncorrelated (no multicollinearity); if this condition is violated, it can lead to unstable coefficient estimates and make it difficult to assess the individual contribution of each predictor variable. Thus, OLS requires more preprocessing to ensure the data fits its assumptions, while Gradient Descent can handle a wider variety of datasets with less stringent requirements.

## [Evaluation Metrics](#evaluation-metrics)

Understanding the performance of our linear regression model is essential, and we use metrics like Mean Squared Error (MSE) and R-squared (R²) to gauge its effectiveness. Let’s break these down using the analogy of house prices.

### Mean Squared Error (MSE)

MSE measures how far our predicted house prices are from the actual prices on average. It calculates the squared difference between each predicted price and the actual price, helping us understand the accuracy of our predictions.

Mathematically, MSE is defined as:

$$
\text{MSE} = \frac{1}{N} \sum (y_{\text{test}} - y_{\text{pred}})^2
$$

where:
- $\( y_{\text{test}} \)$ = actual house prices
- $\( y_{\text{pred}} \)$ = predicted house prices
- $\( N \)$ = number of observations

If the predicted house prices are close to the actual prices, the MSE will be small, indicating good predictions. Conversely, if the predictions are widely off, the MSE will be large, showing poor performance.

### Root Mean Squared Error (RMSE)

The Root Mean Squared Error (RMSE) is another useful metric that provides the error in the same unit as the original data by taking the square root of MSE:

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum (y_{\text{test}} - y_{\text{pred}})^2}
$$

### Mean Absolute Error (MAE)

Mean Absolute Error (MAE) measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s defined as:

$$
\text{MAE} = \frac{1}{N} \sum |y_{\text{test}} - y_{\text{pred}}|
$$

### R-squared (R²)

R-squared is a key metric that tells us how well our model explains the variation in house prices. It measures the proportion of the variance in the dependent variable (actual prices) that can be predicted from the independent variables (features used for predictions).

Mathematically, R² is calculated as:

$$
R^2 = 1 - \frac{\sum (y_{\text{test}} - y_{\text{pred}})^2}{\sum (y_{\text{test}} - \bar{y})^2}
$$

As you notice, it simply is: 

$$
R^2 = 1 - \frac{\text{squared sum of residuals}}{\text{variance of actual values}}
$$

where:
- $\( \bar{y} \)$ is the mean of the actual house prices.

In simpler terms, R² tells us the percentage of the variance in actual house prices that is explained by our model. For instance, if R² = 0.39, it means our model explains 39% of the variation in house prices, while 61% remains unexplained.

Here's how it relates to our house analogy:
- If R² = 1, it means our predictions perfectly match the actual prices.
- If R² = 0, our predictions are no better than just guessing the average house price.

Thus, a higher R² value indicates that our model captures more of the variability in the actual prices, while a lower R² suggests that our model may be missing key factors influencing house prices.

### Adjusted R-squared

Adjusted R-squared modifies R² to account for the number of predictors in the model. It penalizes excessive use of unhelpful variables, providing a more accurate measure of model performance:

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(N - 1)}{N - k - 1} \right)
$$

where:
- $\( k \)$ = number of independent variables
- $\( N \)$ = number of observations

While R-squared can only increase or stay the same when you add more predictors (even if they are irrelevant), Adjusted R-squared can decrease if the new predictors don't improve the model. This makes Adjusted-R squared a better measure when comparing models with different numbers of predictors.

```🛈``` This metric is especially useful for comparing models with different numbers of predictors, as it helps prevent overfitting.

### Visualization Insights

1. **Residuals vs. Predicted Prices Plot**:
   - Points close to the zero line indicate good predictions, while points far from this line suggest larger errors in prediction.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f44bd02f-10c6-4d24-ba83-cc6539c39a75" alt="RP" width="500" />
</p>

2. **Histogram of Residuals**:
   - A normal distribution centered at zero indicates that most of the predictions are accurate, with no systematic bias in over- or under-predicting house prices.
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/f1148b1d-c577-4b03-92f7-2f574120e192" alt="RP" width="500" />
</p>

`🗲` By analyzing MSE, RMSE, MAE, R², and Adjusted R², along with visualizing residuals, we can gain valuable insights into how well our linear regression model performs in predicting house prices.

## [Interpreting Coefficients](#interpreting-coefficients)

### Purpose of Interpreting Coefficients `(‘•.•’)?`

 Understanding the coefficients of a regression model helps you answer an important question: **How do changes in our input features affect our predicted outcome?** In our house price analogy, think of it like this: **How much does an extra bedroom or square footage increase the predicted sale price?**

Each coefficient tells us the expected change in the dependent variable (house price) for a one-unit increase in an independent variable (e.g., number of rooms, lot size), **while holding all other variables constant**.

By interpreting these coefficients, we can gain insights into the relationships between our predictors and the target variable. This not only helps in model evaluation but also aids in decision-making, for instance, understanding which features are the most important drivers of house prices.

### The Coefficients' Role Explained

> The regression equation takes the form:

$$
y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n
$$

Where:
- $\( y \)$ is the predicted value (house price)
- $\( X_1, X_2, ..., X_n \)$ are the independent variables (like bedrooms, bathrooms, lot size, etc.)
- $\( \beta_0 \)$ is the intercept (the price when all features are zero, often a baseline price)
- $\( \beta_1, \beta_2, ..., \beta_n \)$ are the coefficients for each independent variable

`💰` In house prices:
- A positive $\( \beta \)$ value means that increasing the corresponding feature increases house prices. For example, if adding a room increases the price by `$20,000`, then $\( \beta_1 \)$ would be 20,000.
- A negative $\( \beta \)$ value suggests that increasing the feature decreases the house price.

### p-value: Is the Feature Significant?

The **p-value** tells us whether each coefficient is significantly different from zero. In simpler terms, it answers this question: **Is this predictor (e.g., the number of bedrooms) contributing meaningful information, or is its effect mostly due to chance?**

- If the p-value is small (typically less than 0.05), the coefficient is considered **statistically significant**, meaning this feature likely has a real effect on house prices.
- If the p-value is large, the feature’s effect might just be noise.

Mathematically, the p-value is derived from the **t-statistic**, which is calculated as:

$$
t = \frac{\beta}{\text{standard error of } \beta}
$$

> The **standard error of $\( \beta \)$** measures the variability of the coefficient estimate, and it is used in the calculation of the t-statistic. It's calculated as:

$$
\text{SE}(\beta) = \sqrt{\frac{\text{MSE}}{\sum (X_i - \bar{X})^2}}
$$

Where:
- $\( \text{MSE} \)$ is the Mean Squared Error
- $\( X_i \)$ are the independent variable values
- $\( \bar{X} \)$ is the mean of the independent variable values

`🛈` This formula helps determine how much the coefficient is likely to vary if we were to repeat the sampling process.

`🗲` The larger the t-statistic, the more significant the coefficient.

### F-statistic: How Well Does the Model Fit?

While the p-value focuses on individual predictors, the **F-statistic** looks at the model as a whole. It tells us whether the group of features we included has a significant relationship with the target variable.

- A high F-statistic suggests that at least one of the predictors has a meaningful effect on house prices, meaning the model isn't just random guessing.

The F-statistic is calculated as:

$$
F = \frac{\text{Explained Variance / Number of Predictors}}{\text{Unexplained Variance / Degrees of Freedom}}
$$

> The **unexplained variance**, also called the residual variance, is the portion of the total variance that the model fails to explain. It is the sum of squared residuals divided by the degrees of freedom:

$$
\text{Unexplained Variance} = \frac{\sum (y_{\text{test}} - y_{\text{pred}})^2}{n - p - 1}
$$

Where:
- $\( n \)$ is the number of observations
- $\( p \)$ is the number of predictors (independent variables)

`🛈` This helps quantify how much variability in the data is left unexplained by the model.

> The **degrees of freedom** (df) in a regression model refer to the number of values in the final calculation that are free to vary. For residuals, it is calculated as:

$$
\text{Degrees of Freedom (Residuals)} = n - p - 1
$$

Where:
- $\( n \)$ is the number of observations
- $\( p \)$ is the number of predictors

`🛈` Degrees of freedom reflect the amount of information available to estimate the regression model parameters.

> The **explained variance** represents the portion of the total variance that is explained by the regression model. It is calculated as:

$$
\text{Explained Variance} = \sum (\hat{y} - \bar{y})^2
$$

Where:
- $\( \hat{y} \)$ are the predicted values
- $\( \bar{y} \)$ is the mean of the actual values

`🛈` This shows how much of the variation in the dependent variable can be attributed to the independent variables in the model.

`🗲` In our house analogy, think of the F-statistic as asking: **Is this collection of features (bedrooms, lot size, etc.) doing a good job of predicting house prices, or would we be better off guessing the average house price every time?**

### ✍️ To Tie It All Together: The Impact of Coefficients, R², and Adjusted R²

 The **coefficients** tell us how each feature influences house prices, while the **p-values** tell us which features have significant effects. The **F-statistic** evaluates whether the model overall is useful for prediction. To cap it off, **R²** and **Adjusted R²** let us know how well the model explains the variance in the actual house prices:

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a995928-e778-4070-bd9e-8540f76433e7" alt="tie" width="500" />
</p>

- $\( R² \)$ tells us how much of the variation in house prices is explained by the model:

$$
R^2 = 1 - \frac{\text{squared sum of residuals}}{\text{variance of actual values}}
$$

- **Adjusted R²** adjusts for the number of predictors, penalizing overly complex models:

$$
R_{\text{adj}}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

Where $\( n \)$ is the number of observations, and $\( p \)$ is the number of predictors.

### `🐦‍🔥`

- **Positive coefficient**: More bedrooms? Price goes up!
- **Negative coefficient**: More crime in the area? Price goes down!
- **p-value**: Does the lot size have a significant effect, or is it just noise?
- **F-statistic**: Is the whole model doing better than just predicting the average price?

`🗲` With these metrics, we not only understand which factors drive house prices but also ensure that our model is more reliable.

## `⛉` [Model Diagnostics](#model-diagnostics)

 Interpreting the coefficients gives us valuable insights, but it’s equally important to check the assumptions and validity of our regression model through diagnostics. Let’s dive into a few key checks: Residual Analysis, Multicollinearity, and Homoscedasticity/Normality Tests.

### Residual Analysis

Residuals are the differences between the actual and predicted values (as we discussed). Analyzing residuals helps us understand if our model is appropriately capturing the relationships in the data.

- **Purpose**: Residual analysis checks whether the errors (residuals) are randomly distributed. Ideally, residuals should have:
  - **`⛊` No patterns** (indicating good fit),
  - **`⛊` Constant variance** (homoscedasticity),
  - **`⛊` Normal distribution**.

- **Steps**:
  1. **Plot Residuals vs. Fitted Values**: This helps to check for homoscedasticity and the presence of any patterns in the errors.
  2. **Create a Q-Q Plot (Quantile-Quantile Plot)**: It visualizes whether residuals follow a normal distribution.

**`💰` House Price Analogy**: If our house price predictions leave residuals that show a pattern or are concentrated in certain areas (e.g., mostly overestimating prices for large houses), it means our model might be missing key relationships.

### Multicollinearity

**Multicollinearity** occurs when two or more predictors in a model are highly correlated, making it hard to isolate their individual effects. This inflates the standard errors of the coefficients and can cause issues with interpreting the coefficients.

- **Purpose❔**: Detecting multicollinearity ensures that each predictor provides unique information.
- **Tests 🧐**:
  - **Variance Inflation Factor (VIF)**: VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity. A VIF value exceeding 10 indicates potential multicollinearity.

**Formula**:

$$
\text{VIF}_i = \frac{1}{1 - R_i^2}
$$

Where $\( R_i^2 \)$ is the R² of regressing the $\( i \)$-th predictor against all the other predictors.

**`💰` House Price Analogy**: Imagine if square footage and the number of rooms in a house are highly correlated. It becomes tough to know which feature is truly affecting the house price if both metrics rise and fall together.

### Homoscedasticity and Normality Tests

Homoscedasticity refers to the assumption that the residuals have constant variance across all levels of the predicted values.

- **Purpose❔**: Ensuring homoscedasticity validates the model's assumptions, and if violated, our inferences from the model could be misleading.
- **Tests 🧐**:
  - **Breusch-Pagan Test**: This test detects heteroscedasticity by checking if residuals vary systematically with fitted values.
  - **Shapiro-Wilk Test**: This test checks whether the residuals are normally distributed.
  
`🗲` If either assumption is violated, the model might need to be revised (e.g., applying transformations to the predictors or the target variable).

## [Common Issues and Solutions](#common-issues-and-solutions)

  _**No model is perfect**_, and during regression modeling, two frequent issues are **overfitting** and **underfitting**.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1125/1*_7OPgojau8hkiPUiHoGK_w.png" alt="tie" width="500" />
</p>

### Overfitting

Overfitting happens when the model learns the noise in the data instead of the true signal. It performs exceptionally well on the training data but poorly on unseen data.

- **Solution**: Simplify the model by:
  - `🛡️` Reducing the number of predictors,
  - `🛡️` Applying regularization techniques (e.g., Ridge or Lasso regression),
  - `🛡️` Using cross-validation to tune the model.

**`💰` House Price Analogy**: Imagine a model that fits the exact price fluctuations in a small neighborhood, but when predicting prices in another area, it fails miserably. That’s overfitting — it learned too much from one dataset.

<p align="center">
  <img src="https://pbs.twimg.com/media/FSAM8FpWUAISGyd.jpg" alt="tie" width="500" />
</p>

### Underfitting

Underfitting occurs when the model is too simple and fails to capture the underlying pattern in the data. This leads to poor performance on both training and testing data.

- **Solution**: Increase the complexity of the model by:
  - `🛡️` Adding more relevant features,
  - `🛡️` Using a more sophisticated algorithm,
  - `🛡️` Allowing for non-linear relationships.

**`💰` House Price Analogy**: If a model just uses the average price of all houses without considering any other features (like location, size, etc.), it will likely underfit and miss key predictors that influence house prices.

## .𖥔 ݁ ˖ Conclusion 

 Linear regression is often considered the backbone of predictive modeling and serves as one of the simplest yet most powerful algorithms in the world of machine learning. Its linear approach provides a clear and interpretable relationship between input features and output predictions, laying the groundwork for more advanced techniques like decision trees, neural networks, and ensemble methods. 

 While linear regression assumes a linear relationship between variables, the skills and insights gained from mastering this algorithm open doors to understanding more complex models, especially in the field of supervised learning. From simple trend analysis to sophisticated forecasting, linear regression remains a key component in the toolkit of data scientists and machine learning practitioners.

 In this project, we've walked through a practical example, applying linear regression to predict housing prices, exploring essential concepts like interpreting coefficients, model diagnostics, and addressing common issues like overfitting. Through such exercises, we begin to appreciate not just the power of linear models, but also the importance of critically evaluating their assumptions and results.

## Further Reading/Resources

- [An Introduction to Statistical Learning](https://www.statlearning.com/)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
- [Linear Regression in Python: A Complete Guide](https://realpython.com/linear-regression-in-python/)
- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Understanding Residuals and Diagnostics in Regression](https://online.stat.psu.edu/stat462/node/172/)
- [Multicollinearity: How to Detect and Fix](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)
