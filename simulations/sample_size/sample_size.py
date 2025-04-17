######## IMPORTACIÓN DE LIBRERIAS ##############
import dash
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import os
import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from template_dashboard import (
    get_external_stylesheets,
    style,
    title_style,
    COMPONENT_HEIGHT,
    create_menu,
    create_graph,
    paper,
    update_layout,
    main_structure,
    build_layout,
    create_stats_table,
    card_title_w_information_hovercard,
)

######## CONFIGURACIÓN DE ESTILO  ##############

dash._dash_renderer._set_react_version("18.2.0")


############### FUNCIONES PROPIAS DE ESTA SIMULACIÓN ################


# Funciones para la lógica de la aplicación
def simulate_experiment(effect_size, variance, sample_size):
    """Simulates a randomized experiment with a given effect size, variance, and sample size.

    Args:
        effect_size: The difference in means between the treatment and control groups.
        variance: The common variance of the outcome variable in both groups.
        sample_size: The number of observations per group.

    Returns:
        A tuple containing the p-value of the t-test and whether the null hypothesis was rejected.
    """
    treatment_group = np.random.normal(
        loc=effect_size / 2, scale=np.sqrt(variance), size=sample_size
    )
    control_group = np.random.normal(
        loc=-effect_size / 2, scale=np.sqrt(variance), size=sample_size
    )
    p_value = sm.stats.ttest_ind(treatment_group, control_group)[1]
    reject_null = p_value < 0.05  # Assuming significance level of 0.05
    return p_value, reject_null


def calculate_power(effect_size, variance, sample_sizes, num_simulations=1000):
    """Calculates the power of a t-test for different sample sizes.

    Args:
        effect_size: The difference in means between the treatment and control groups.
        variance: The common variance of the outcome variable in both groups.
        sample_sizes: A list of sample sizes to test.
        num_simulations: The number of simulation iterations to perform for each sample size.

    Returns:
        A dictionary containing the power values for each sample size.
    """
    power_results = {}
    for sample_size in sample_sizes:
        rejections = 0
        for _ in range(num_simulations):
            _, reject_null = simulate_experiment(effect_size, variance, sample_size)
            if reject_null:
                rejections += 1
        power_results[sample_size] = rejections / num_simulations
    return power_results


# Crear gráfico de power vs. sample size
def create_power_plot(power_results):
    fig = px.line(
        x=list(power_results.keys()),
        y=list(power_results.values()),
        labels={"x": "Sample Size", "y": "Power"},
    )
    fig.update_layout(
        title="Power of the Test vs. Sample Size",
        xaxis_range=[min(power_results.keys()), max(power_results.keys())],
        yaxis_range=[0, 1],
    )
    return fig


# Función para actualizar el gráfico de power
def update_power_graph(effect_size, variance, sample_sizes):
    power_results = calculate_power(effect_size, variance, sample_sizes)
    fig = create_power_plot(power_results)
    return fig


############### INSTANCIA DE LA APP ###############3
app = Dash(
    __name__,
    external_stylesheets=get_external_stylesheets(),
    suppress_callback_exceptions=True,
)

############# ELEMENTOS DE LA APP (LAYOUT) ##################
### 1) ELEMENTOS DE TEXTO CAJITAS
### 2) GRÁFICOS
### 3) SLIDERS

############ 1) ELEMENTOS DE TEXTO CAJITAS ##############3

# Definimos el contenido de cada sección como strings separados

intro_content = """
This dashboard helps determine the optimal sample size for a randomized experiment by simulating the power of a statistical test for different sample sizes.  The experiment is randomly assigned to a treatment group.  
You can choose the desired effect size, the variance of the outcome variable, and the significance level, then the dashboard will calculate the power of the test for different sample sizes.

"""

simulation_content = """
## Methodology

1. **Data Generation:** The dashboard simulates data for the treatment and control groups using a normal distribution with specified parameters.
2. **Hypothesis Testing:**  A t-test is performed to compare the means of the two groups.
3. **Power Calculation:**  The power of the test is calculated as the proportion of simulations where the null hypothesis is rejected (i.e., a statistically significant difference is detected).
4. **Visualization:** The power values are plotted against the corresponding sample sizes to illustrate the relationship between these factors.

Adjust the sliders to observe how changes in effect size, variance, and significance level impact the power of the test and the required sample size.
"""

# Creamos el componente Accordion (las cajitas) con las secciones

markdown_description = create_menu(
    600,
    {"Introduction": intro_content},
    {"Methodology": simulation_content},
)

############ 2) GRÁFICOS ##############3

# Power vs. Sample Size
power_plot_content = create_graph(
    id="power_plot",
    title="Power of the Test vs. Sample Size",
    information="This plot shows the relationship between the sample size and the power of the test.",
)


######### 3) SLIDERS ##############3

sliders_stack = dmc.Stack(
    [
        card_title_w_information_hovercard(
            title="Experiment Parameters",
            information="Adjust these parameters to simulate different scenarios for your experiment.",
        ),
        # dmc.Text("Desviación estándar del error (Sigma ε)", fw=600),
        dcc.Markdown("*Effect Size (Cohen's d)* $(d)$", mathjax=True),
        dmc.Slider(
            id="effect_size",
            updatemode="drag",
            min=0.1,
            max=1.0,
            step=0.05,
            marks=[
                {"value": 0.1, "label": "0.1"},
                {"value": 0.2, "label": "0.2"},
                {"value": 0.5, "label": "0.5"},
                {"value": 0.8, "label": "0.8"},
                {"value": 1.0, "label": "1.0"},
            ],
            value=0.5,
            style={"width": "80%"},
        ),
        dmc.Space(h=2),
        # dmc.Text("Desviación estándar de X (Sigma X)", fw=600),
        dcc.Markdown("Variance of Outcome ($σ^2$)", mathjax=True),
        dmc.Slider(
            id="variance",
            updatemode="drag",
            min=1,
            max=100,
            marks=[
                {"value": 1, "label": "1"},
                {"value": 25, "label": "25"},
                {"value": 50, "label": "50"},
                {"value": 75, "label": "75"},
                {"value": 100, "label": "100"},
            ],
            value=10,
            style={"width": "80%"},
        ),
        dmc.Space(h=2),
        # dmc.Text("Desviación estándar de X (Sigma X)", fw=600),
        dcc.Markdown("Significance Level (α)", mathjax=True),
        dmc.Slider(
            id="alpha",
            updatemode="drag",
            min=0.01,
            max=0.1,
            step=0.01,
            marks=[
                {"value": 0.01, "label": "0.01"},
                {"value": 0.05, "label": "0.05"},
                {"value": 0.1, "label": "0.1"},
            ],
            value=0.05,
            style={"width": "80%"},
        ),
    ],
    justify="center",
    gap="xl",
)


########## ESTRUCTURA ##################3
resumen_content = main_structure(
    menu=markdown_description,
    structure=[
        # Fila 1
        [[paper(sliders_stack)], [paper(power_plot_content)]],
    ],
)


################# CALLBACK ###############
@app.callback(
    Output("power_plot", "figure"),
    Input("effect_size", "value"),
    Input("variance", "value"),
    Input("alpha", "value"),
)
def update_power_graph(effect_size, variance, alpha):
    # Set the sample size range
    sample_sizes = range(10, 501, 10)  # From 10 to 500, in steps of 10

    # Update the power plot
    fig = update_power_graph(effect_size, variance, sample_sizes)
    return fig


############ LAYOUT ##################
app.layout = build_layout(
    title="Sample Size Calculator for Randomized Experiments",
    content=resumen_content,
)

############ RUN SERVER #################
if __name__ == "__main__":
    # port = int(os.environ.get("DASH_PORT"))
    # app.run_server(host="0.0.0.0", debug=False, port=port)
    app.run_server(debug=False, port=8070)
