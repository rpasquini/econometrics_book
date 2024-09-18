import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Add MathJax to the Dash app
"""
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

"""

app.layout = html.Div([
    html.H1("Distribution of Estimated $\\beta_1$"),
    dcc.Markdown('''
        This dashboard simulates the distribution of the estimated coefficient $\\beta_1$ in the linear regression model:
        
        $$ y = \\beta_0 + \\beta_1X + \\epsilon $$
        
        where:
        - $y$ is the dependent variable,
        - $X$ is the independent variable,
        - $\\beta_0$ is the intercept,
        - $\\beta_1$ is the slope coefficient,
        - $\\epsilon$ is the error term.
        
        Use the sliders below to adjust the parameters of the simulation:
        - **Standard Deviation of X (\\(\\sigma_X\\))**: Controls the spread of the independent variable $X$.
        - **Standard Deviation of Error Term (\\(\\sigma\\))**: Controls the spread of the error term $\\epsilon$.
        - **Sample size**: Controls the number of samples in each simulation.
    ''', dangerously_allow_html=True),
    dcc.Graph(id='histogram'),
    html.Label("Standard Deviation of X (\\(\\sigma_X\\))"),
    dcc.Slider(id='sigma_X', min=0.1, max=5, step=0.1, value=1),
    html.Label("Standard Deviation of Error Term (\\(\\sigma\\))"),
    dcc.Slider(id='sigma', min=0.1, max=5, step=0.1, value=1),
    html.Label("Sample size"),
    dcc.Slider(id='n', min=0, max=1000, step=50, value=100),
    html.Div(id='stats')
])

@app.callback(
    [Output('histogram', 'figure'),
     Output('stats', 'children')],
    [Input('sigma_X', 'value'),
     Input('sigma', 'value'),
     Input('n', 'value')]
)
def update_histogram(sigma_X, sigma, n):
    # Parameters
    np.random.seed(42)
    beta_0 = 2
    beta_1 = 3
    mu_X = 0  # mean of X
    num_simulations = 5000

    # Storage for beta_1 estimates
    beta_1_estimates = []

    for _ in range(num_simulations):
        # Generate random X and error term epsilon
        X = np.random.normal(mu_X, sigma_X, n)
        epsilon = np.random.normal(0, sigma, n)
        
        # Generate Y
        Y = beta_0 + beta_1 * X + epsilon
        
        # Perform linear regression
        X = sm.add_constant(X)  # Add intercept term
        model = sm.OLS(Y, X)
        results = model.fit()
        
        # Store the estimated beta_1
        beta_1_estimates.append(results.params[1])

    # Calculate mean and standard deviation of beta_1 estimates
    mean_beta_1 = np.mean(beta_1_estimates)
    std_beta_1 = np.std(beta_1_estimates)

    # Create the histogram
    fig = px.histogram(beta_1_estimates, nbins=30, title='Distribution of Estimated $\\beta_1$',
                       labels={'value': 'Estimated $\\beta_1$', 'count': 'Frequency'})

    # Create stats text
    stats_div = html.Div([
        html.H4("Statistics:"),
        html.P(f"Mean of $\\beta_1$: {mean_beta_1:.4f}"),
        html.P(f"Standard Deviation of $\\beta_1$: {std_beta_1:.4f}")
    ])
    
    return fig, stats_div

if __name__ == '__main__':
    app.run_server(debug=True)
