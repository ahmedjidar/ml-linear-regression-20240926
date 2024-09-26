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
   - The relationship between the independent (predictor) variables and the dependent (response) variable is linear. Changes in the predictor variables should result in proportional changes in the response variable.
---
### ✔ Independence
   - The observations (data points) are independent of each other. One observation's value should not influence another's.
---
### ✔ Homoscedasticity
   - The variance of the residuals (the differences between observed and predicted values) is constant across all levels of the independent variables. The spread of the residuals should be roughly the same for low, medium, and high predicted values.

<h4><u><strong><em>🔹Understanding Homoscedasticity Through a Basketball Analogy</em></strong></u></h4>

Imagine you’re on the playground, playing a game where you throw a basketball into a hoop. Each time you throw, you want to see how close you can get the ball to the basket. The distance from where the ball lands to the basket represents how "off" your throw is, which we can think of as an **error** in our game.

<h4><u><strong><em>🔹Consistency in Throws</em></strong></u></h4>

Now, let’s say every time you throw the ball softly, it lands fairly close to the basket. But when you throw it harder, the ball sometimes lands really far away. This means that the “error” changes based on how hard you throw. What we want to see is a situation where, whether you throw the ball softly or hard, the distance it lands from the basket stays pretty much the same.

This idea is what we call **homoscedasticity**!

<h4><u><strong><em>🔹What Is Homoscedasticity?</em></strong></u></h4>

In simple terms, homoscedasticity means that the variance (or spread) of your throws remains constant no matter how you throw the ball. In our math model of this game, we can express it as follows:

1. **Linear Equation**:
   When we describe this relationship mathematically, we say:

$$Y = \beta_0 + \beta_1 X + \epsilon$$
   
   - $\(Y\)$ is how far the ball lands from the basket.
   - $\(X\)$ is how hard you throw the ball.
   - $\(\beta_0\)$ represents where we expect the ball to land on average (the y-intercept).
   - $\(\beta_1\)$ tells us how much the expected distance changes with each throw (the slope).
   - $\(\epsilon\)$ is the error term, which represents how far off your throw is.

<h4><u><strong><em>🔹The Importance of Constant Variance</em></strong></u></h4>

2. **Constant Variance**:
   Homoscedasticity means that the error $(\(\epsilon\))$ has a constant spread, regardless of how hard you throw the ball. Mathematically, this is expressed as:

$$\text{Var}(\epsilon | X) = \sigma^2$$

   - Here, $\(\text{Var}(\epsilon | X)\)$ means "the variance of the error when we look at the throws."
   - $\(\sigma^2\)$ is a constant number representing how much the errors spread out from the expected distance (the basket).

<h4><u><strong><em>🔹Visual Representation</em></strong></u></h4>

To better understand this concept, let’s take a look at a graph that illustrates both homoscedasticity and heteroscedasticity:

<p align="center">
  <img src="" width="400" />
</p>

In the image, you can see how in homoscedasticity, the errors are randomly scattered around zero with a consistent spread, whereas in heteroscedasticity, the errors vary in spread, making it difficult to predict outcomes.

<h4><u><strong><em>🔹Summary</em></strong></u></h4>

So, in summary, homoscedasticity is essential for ensuring that your throws (errors) stay consistent, allowing you to better understand how well you’re aiming for the basket (modeling your data). When this condition holds, it allows for more accurate predictions and better model performance.

---
### ✔ Normality of Residuals
   - The residuals of the model should be approximately normally distributed. When plotted, the residuals should form a bell-shaped curve around zero.
---
### ✔ No Multicollinearity
   - The independent variables should not be highly correlated with each other. Multicollinearity can make it difficult to determine the individual effect of each predictor variable.
---
### Visual Representation
To better understand these assumptions, consider the following visual aids:
- **Scatter Plot**: Illustrates the linearity assumption with a scatter plot of the independent and dependent variables, including the regression line.
- **Residual Plot**: Demonstrates homoscedasticity by plotting residuals against predicted values.
- **Histogram/QQ Plot**: Displays the normality of residuals using a histogram or a QQ plot.



