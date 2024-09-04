# Assumptions and Diagnostics

This chapter delves into the essential assumptions underlying regression analysis, exploring their implications for model accuracy and interpretation. We will then examine practical methods for diagnosing potential violations of these assumptions, using a combination of statistical tests and visual exploration of model residuals.

## Assumptions of Linear Regression
Linear regression models, despite their simplicity, rest on a set of key assumptions that must be met for valid inference and interpretation. These assumptions ensure that the model's parameters are estimated accurately and that the statistical tests are reliable.

### Linearity 
The fundamental assumption of linearity posits a straight-line relationship between the dependent variable (Y) and the independent variable(s) (X). This assumption is inherent in the model's structure, where the relationship is represented by a linear equation: $$Y=\beta_0+\beta_1X+\varepsilon$$
Here, $\beta_0$ represents the intercept, $\beta_1$ represents the slope, and $\epsilon$ represents the error term. This equation implies that for every unit change in X, Y changes by a constant amount determined by $\beta_1$. 
Deviation from linearity can lead to biased parameter estimates and misinterpretations of the model's fit. For instance, if the true relationship between Y and X is curvilinear, a linear model will misrepresent the relationship, potentially underestimating or overestimating the effects of X on Y.
Example: Imagine we are trying to model the relationship between the number of bedrooms in a house (X) and its rental price (Y). If the true relationship is curvilinear, meaning that the price increases at a decreasing rate as the number of bedrooms increases, a linear model will likely overestimate the price of smaller houses and underestimate the price of larger houses.

Consequences of Violating Linearity:
* Biased parameter estimates: The estimated coefficients (β) will not accurately reflect the true relationship between the variables.
* Incorrect predictions: The model will generate inaccurate predictions for the dependent variable.
* Misleading inferences: The statistical tests conducted on the model will be unreliable, potentially leading to incorrect conclusions about the significance of the independent variables.

Addressing Linearity Violations:
Transformations: Applying transformations to the variables (e.g., taking logarithms, square roots) can sometimes linearize the relationship.
Non-Linear Models: When linearity cannot be achieved through transformations, non-linear models, such as polynomial regression or spline regression, might be more appropriate.

### Random Sampling
The assumption of random sampling mandates that the data points are drawn randomly from the population of interest. Violations of this assumption introduce bias, making it impossible to generalize the findings from the sample to the population.
Example: If we are studying the relationship between income and housing prices, but our sample only includes residents of a wealthy neighborhood, our findings cannot be generalized to the broader population. This is because our sample is not representative of the entire population.

Consequences of Violating Random Sampling:
Biased estimates: The estimated coefficients will be biased toward the characteristics of the non-random sample.
Inaccurate generalizations: The findings cannot be generalized to the population of interest.
Misleading inferences: Statistical tests will be unreliable due to the biased nature of the sample.
Addressing Random Sampling Violations:
Collect a new dataset: A new dataset with proper random sampling procedures must be collected to address violations of random sampling.

### No Perfect Multicollinearity
In models with multiple independent variables, perfect multicollinearity arises when two or more variables are perfectly correlated. This scenario makes it impossible to isolate the individual effects of each variable, hindering model interpretation.
Example: Imagine a model predicting housing prices using both the number of bedrooms and the number of bathrooms. If all houses in the dataset have the same ratio of bedrooms to bathrooms (e.g., one bedroom for every bathroom), these variables are perfectly correlated. This perfect correlation prevents the model from determining the independent effect of each variable on the price.

Consequences of Violating No Perfect Multicollinearity:
Indeterminable individual effects: The individual effects of the perfectly correlated variables cannot be separated.
Unstable estimates: The coefficients of the correlated variables may be highly sensitive to small changes in the data.
Misleading inferences: Statistical tests may be unreliable due to the unstable estimates.
Addressing Multicollinearity Violations:
Remove variables: One of the highly correlated variables can be removed from the model.
Combine variables: The correlated variables can be combined into a single variable that represents their joint effect.
Ridge Regression: Ridge regression is a technique that can be used to stabilize the estimates in the presence of multicollinearity.

### Exogeneity
This assumption stipulates that the independent variables are not influenced by the dependent variable. If exogeneity is violated, the estimated relationship may be confounded by feedback loops or omitted variable bias.
Example: Suppose we are examining the relationship between educational attainment (X) and income (Y). If higher income levels lead to more investment in education, then exogeneity is violated because X is influenced by Y.

Consequences of Violating Exogeneity:
Confounded effects: The estimated effect of X on Y may be confounded by the feedback loop, making it difficult to isolate the true causal relationship.
Omitted variable bias: If a relevant variable that influences both X and Y is omitted from the model, the estimates of the relationship between X and Y may be biased.
Misleading inferences: Statistical tests may be unreliable due to the biased estimates.
Addressing Exogeneity Violations:
Incorporate additional variables: Including the omitted variable(s) that influence both X and Y in the model can mitigate omitted variable bias.
Instrumental variables: Instrumental variables techniques can be used to estimate the causal effect of X on Y when exogeneity is violated.
Panel data models: Panel data models, which analyze data over time and across individuals, can account for feedback loops and omitted variable bias.
This section has outlined the core assumptions of linear regression models, emphasizing their importance for model validity and interpretation. The next sections will delve into practical methods for assessing these assumptions and addressing violations.


## Diagnosing Model Assumptions

Various diagnostic tools are available to assess the adequacy of model assumptions and detect potential violations.
3.2.1 Visual Inspection of Residuals: A fundamental technique for diagnosing model assumptions involves examining the model's residuals (the difference between observed and predicted values).
Linearity: A plot of residuals against the fitted values can reveal non-linear patterns, suggesting a violation of the linearity assumption. If the residuals exhibit a systematic pattern, such as a curved shape, it suggests that the linear model does not fully capture the relationship between the variables.
Example: A plot of residuals against the fitted values might show that residuals tend to be positive for low fitted values, then become negative for higher fitted values, and then positive again, indicating a non-linear relationship.
Homoscedasticity: A plot of residuals against the independent variables can indicate heteroscedasticity if the spread of residuals changes systematically with the independent variables. If the spread of residuals increases or decreases with the independent variable, it indicates that the variance of the errors is not constant, violating the homoscedasticity assumption.
Example: A plot of residuals against the number of bedrooms might show a wider spread of residuals for houses with fewer bedrooms compared to houses with more bedrooms, suggesting heteroscedasticity.
Normality: A histogram of the residuals should resemble a normal distribution. Deviations from normality, such as skewed distributions or outliers, can indicate violations of the normality assumption. A normal distribution of residuals implies that the error term in the model is normally distributed. Deviations from normality might suggest that the model is not capturing all the relevant factors influencing the dependent variable.
Example: A histogram of residuals might show a skewed distribution, with a long tail to the right, indicating that the residuals are not normally distributed.
Autocorrelation: Time series data should be plotted against time, while spatial data should be plotted on a map to detect patterns of autocorrelation. Time series data exhibiting autocorrelation might show that residuals from one time period are correlated with residuals in the next time period. Similarly, spatial autocorrelation might be evident if residuals in one location are correlated with residuals in nearby locations.
Example: A plot of residuals over time might exhibit a cyclical pattern, indicating that positive residuals are likely to be followed by positive residuals and negative residuals by negative residuals. In spatial data, a map of residuals might show clusters of high or low residuals in certain areas, suggesting spatial autocorrelation.


Statistical Tests: Specific statistical tests can be employed to formally assess the validity of certain assumptions.
Shapiro-Wilk and Kolmogorov-Smirnov tests: Used to test the normality of the residuals. These tests compare the observed distribution of residuals to the theoretical normal distribution, rejecting the null hypothesis of normality if significant deviations are found.
Breusch-Pagan test: Used to test for heteroscedasticity. This test examines whether the variance of the error term is constant across different values of the independent variables. A significant result indicates heteroscedasticity.
Durbin-Watson test: Used to test for autocorrelation in time series data. This test examines the correlation between residuals from adjacent time periods. A value close to 2 suggests no autocorrelation, while values significantly lower or higher than 2 indicate the presence of autocorrelation.
Moran's I test: Used to test for spatial autocorrelation in spatial data. Moran's I measures the spatial autocorrelation of a variable based on its spatial distribution. A positive Moran's I value indicates that similar values are clustered together, while a negative value indicates that dissimilar values are clustered together.


# Addressing Violations of Assumptions
While some assumptions are more critical than others, it is vital to identify and address any substantial violations to ensure accurate model interpretation and inference.
Linearity: If linearity is violated, transforming the variables (e.g., taking logarithms, square roots) can sometimes linearize the relationship. Applying logarithms can help to linearize relationships that are exponential in nature, while square roots can linearize relationships that are quadratic.
Random Sampling: Addressing violations of random sampling requires collecting a new data set with proper sampling procedures.
Multicollinearity: Addressing multicollinearity may involve removing one of the highly correlated variables, combining the correlated variables, or using specialized techniques like ridge regression.
Exogeneity: Addressing exogeneity may involve incorporating additional variables into the model to account for omitted variable bias or using instrumental variables techniques.
Homoscedasticity: Heteroscedasticity can often be addressed by using robust standard errors, which are less sensitive to variations in the error variance. Alternatively, weighting the observations can account for unequal variances. This involves assigning weights to the observations based on their error variances, giving more weight to observations with lower variances.
Normality of Errors: Violations of normality can be mitigated through data transformations, particularly taking logarithms. In cases of severe non-normality, non-parametric methods may be more appropriate. Non-parametric methods, such as the Mann-Whitney U test or the Kruskal-Wallis test, do not rely on the assumption of normality.
Independence of Errors: Autocorrelation is commonly addressed by incorporating lagged variables into the model (e.g., autoregressive models). Spatial autocorrelation can be addressed by using spatial regression methods. Spatial regression methods, such as geographically weighted regression (GWR), account for the spatial dependence of the data.


