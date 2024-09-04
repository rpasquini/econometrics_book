

### 1. **Introduction and Review of Basic Statistics**
   - **Topic:** Sampling Distributions
   - **Simulation:** Visualizing the Central Limit Theorem (CLT)
   - **Learning Objective:** Demonstrate how the sampling distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population distribution.
   - **Este queda pendiente para mas adelante**

### 2. **The Simple Regression Model** 
   - **Topic:** OLS Estimators

   - **Simulation:** Generating Random Data to Estimate OLS Coefficients

   - **Learning Objective:** Illustrate the properties of OLS estimators, including unbiasedness, and the determinants of the variance of the estimators. 

   - Idea: The dashboard shows the distribution of the estimation. Allows changing the $\sigma$ and $SST_x$ . Computes the mean and the sampling variance as estimates of E[\beta] and Var[\beta]

   - **Lo toma Fran y se discute entre grupo.**

     ## 2.1 Unbiased estimation of /sigma^2

     * Similarly to the betas, simulate the distribution of $\hat{\sigma}. See Wooldridge page 49.

     * **Lo puede tomar alguno Juan/Benjamin/Octi**
     
       

# 3. **Multiple Regression Analysis**



## 3.1 Multicollinearity

   - **Topic:** Multicollinearity
   - **Simulation:** Generating Data with Varying Degrees of Multicollinearity
   - **Learning Objective:** Show the impact of multicollinearity on coefficient estimates, standard errors, and overall model interpretation.
   - Using a model with two variables, allow changing the degree of correlation. As the correlation changes show the effects on the variability of estimates. 
   - **Lo puede tomar alguno Juan/Benjamin/Octi**

## 3.2 Omitted variable bias

* Allow changing the relationship between the omitted variable , the dependent and the independent variable.
* **Lo puede tomar alguno Juan/Benjamin/Octi**

### 4. **Multiple Regression Analysis: Inference**
   - **Topic:** Hypothesis Testing in Regression

   - **Simulation:** Simulating the Distribution of Test Statistics under Null and Alternative Hypotheses

   - **Learning Objective:** Explore the power of tests, Type I and Type II errors, and how sample size and effect size affect hypothesis testing.

     * **Mostrariamos primero el ejemplo que hicimos para el blog de beta Sigma. Despues lo asignamos.**

       

<u>Los siguientes van a la segunda etapa:</u>

### 5. **Asymptotics and Large-Sample Properties of OLS**

   - **Topic:** Law of Large Numbers and Central Limit Theorem for OLS Estimators
   - **Simulation:** Simulating Large Samples and Observing the Convergence of OLS Estimates
   - **Learning Objective:** Demonstrate how OLS estimators converge in probability to the true parameter values and how their distribution approximates normality as sample size increases.

### 6. **Heteroskedasticity**
   - **Topic:** Consequences and Detection of Heteroskedasticity
   - **Simulation:** Simulating Data with Varying Levels of Heteroskedasticity
   - **Learning Objective:** Illustrate how heteroskedasticity affects the efficiency of OLS estimates, and demonstrate tests like the Breusch-Pagan and White tests.

### 7. **Autocorrelation**
   - **Topic:** Consequences and Detection of Autocorrelation
   - **Simulation:** Generating Time-Series Data with Different Autocorrelation Structures
   - **Learning Objective:** Show the impact of autocorrelation on OLS estimates and illustrate tests like the Durbin-Watson test.

### 8. **Endogeneity and Instrumental Variables**
   - **Topic:** Bias from Endogeneity
   - **Simulation:** Simulating the Effects of Endogenous Regressors on OLS Estimates
   - **Learning Objective:** Explain the bias introduced by endogenous variables and demonstrate how instrumental variables (IV) can be used to obtain consistent estimates.

### 9. **Simultaneous Equations Models**
   - **Topic:** Identification and Estimation in Simultaneous Equations
   - **Simulation:** Simulating Systems of Equations with Identification Issues
   - **Learning Objective:** Illustrate the concepts of identification and the use of two-stage least squares (2SLS) in estimating simultaneous equation models.

### 10. **Panel Data Models**
   - **Topic:** Fixed Effects vs. Random Effects
   - **Simulation:** Simulating Panel Data with Individual-Specific Effects
   - **Learning Objective:** Compare fixed effects and random effects estimators, showing how each handles individual-specific heterogeneity.

### 11. **Limited Dependent Variable Models**
   - **Topic:** Binary Response Models (Probit/Logit)
   - **Simulation:** Simulating Binary Outcomes and Estimating Probit/Logit Models
   - **Learning Objective:** Show the difference between linear probability models and nonlinear models like Probit and Logit, focusing on the interpretation of coefficients and prediction.

### 12. **Time-Series Econometrics**
   - **Topic:** Stationarity and Unit Roots
   - **Simulation:** Generating Non-Stationary and Stationary Time-Series Data
   - **Learning Objective:** Illustrate the concept of stationarity, unit roots, and how these properties affect the analysis and interpretation of time-series models.

### 13. **Forecasting**
   - **Topic:** Model Selection and Forecasting Accuracy
   - **Simulation:** Simulating Forecasts from Different Models and Comparing Their Performance
   - **Learning Objective:** Demonstrate the trade-offs between bias and variance in forecasting and how to evaluate forecast accuracy using different metrics.

### 14. **Advanced Topics: Nonlinear Models and Maximum Likelihood Estimation**
   - **Topic:** Maximum Likelihood Estimation (MLE)
   - **Simulation:** Estimating Parameters Using MLE on Simulated Data
   - **Learning Objective:** Illustrate the process of MLE, comparing it to OLS, and discussing its properties such as consistency and asymptotic normality.

These simulations can be developed incrementally, starting with the foundational concepts and progressively integrating more complex models and econometric techniques.