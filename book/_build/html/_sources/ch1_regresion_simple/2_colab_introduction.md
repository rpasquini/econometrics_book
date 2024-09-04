### Implementing Regression in Python: A Practical Guide

This section provides a step-by-step guide to implementing regression analysis using Python, mirroring the class's hands-on approach.

- **Colaboratory (Colab):** A cloud-based platform providing an interactive environment for coding, data analysis, and model building. It offers a user-friendly interface with text cells for documentation and code cells for executing Python commands. This allows for easy sharing and collaboration on code and data analysis.
- **Libraries:**
  - **GeoPandas:** A powerful library extending the functionality of Pandas, incorporating support for geospatial data analysis. This is particularly useful when working with data that includes location information.
  - **Scikit-learn:** A versatile machine learning library containing various regression models. This library provides a rich set of tools for building and evaluating regression models.

**Step 1: Importing Libraries:**

```
# Install GeoPandas if not already installed
!pip install geopandas==0.12.2

# Import GeoPandas
import geopandas as gpd
```

content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

**Step 2: Loading Data:**

```
# Load data from a file or URL
departamentos = gpd.read_file("https://...your_data_source...")
```

content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

**Step 3: Data Exploration and Manipulation:**

- **Viewing Data:**

  ```
  # View the first 10 rows of the dataframe
  departamentos.head(10)
  
  # View all column names
  departamentos.columns 
  
  # View the dimensions of the dataframe
  departamentos.shape
  ```

  content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

- **Accessing Columns:**

  ```
  # Access the values in the 'price' column
  departamentos['price'] 
  
  # Access the values in the 'L2' column
  departamentos['L2']
  ```

  content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

- **Calculating Descriptive Statistics:**

  ```
  # Calculate the mean of the 'price' column
  departamentos['price'].mean()
  
  # Calculate the standard deviation of the 'price' column
  departamentos['price'].std()
  ```

  content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

- **Filtering Data:**

  ```
  # Filter data for apartments in Capital Federal
  capital_federal_apartments = departamentos[departamentos['L2'] == 'Capital Federal']
  
  # Filter data for apartments with a price greater than 100,000 AND in Capital Federal
  high_price_capital_apartments = departamentos[(departamentos['price'] > 100000) & (departamentos['L2'] == 'Capital Federal')]
  ```

  content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

**Step 4: Building and Fitting a Regression Model:**

```
from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Select features and target variable
X = departamentos[['number_of_bedrooms']] # Features
y = departamentos['price'] # Target variable

# Fit the model to the data
model.fit(X, y)
```

content_copyUse code [with caution](https://support.google.com/legal/answer/13505487).Python

**Step 5: Interpreting Results and Making Predictions:**

```
# Print the intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Make predictions for new data
new_data = [[3]] # Apartment with 3 bedrooms
predicted_price = model.predict(new_data)
print("Predicted Price:", predicted_price)
```