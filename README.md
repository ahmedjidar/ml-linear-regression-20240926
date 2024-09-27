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
9. [Implementing Linear Regression](#implementing-linear-regression)
   - Code Example
   - Data Preparation
10. [Conclusion](#conclusion)
11. [Further Reading/Resources](#further-readingresources)

## ✦ [Introduction to Linear Regression](#introduction-to-linear-regression)

### ```ⓘ``` Definition
Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The primary goal is to predict the value of the dependent variable based on the values of the independent variables by fitting a linear equation to the observed data.

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

By understanding these mathematical foundations, you can better grasp how linear regression models function and how to apply them effectively to various data-driven problems.

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
  <img src="" width="400" />
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




