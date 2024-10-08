####### IMPORTACIÓN DE LIBRERIAS ##############
import dash
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import os
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
)

######## CONFIGURACIÓN DE ESTILO  ##############

dash._dash_renderer._set_react_version("18.2.0")


############### FUNCIONES PROPIAS DE ESTA SIMULACIÓN ################
def calculate_stats(beta_0_list, beta_1_list, beta_2_list):
    return {
        "mean_beta_0": np.mean(beta_0_list),
        "var_beta_0": np.var(beta_0_list),
        "mean_beta_1": np.mean(beta_1_list),
        "var_beta_1": np.var(beta_1_list),
        "mean_beta_2": np.mean(beta_2_list),
        "var_beta_2": np.var(beta_2_list),
    }


# Crear histograma
def create_histogram(data, range_x, true_value, x_label="x"):
    fig = px.histogram(
        x=data,
        color_discrete_sequence=["indianred"],
        range_x=range_x,
        labels={"x": x_label},
    )
    fig.add_vline(
        x=true_value,
        line_dash="dash",
        line_color="black",
        annotation_text="Valor verdadero",
    )
    return fig


# Crear scatter
def create_scatter_matrix_plot(X1, X2, Y):
    return px.scatter_matrix(
        data_frame=pd.DataFrame({"X1": X1, "X2": X2, "Y": Y}),
        dimensions=["X1", "X2", "Y"],
        color_discrete_sequence=["indianred"],
    )


# Funciones para la lógica de la aplicación
def generate_sample(
    sample_size, beta_0, beta_1, beta_2, mean: list, correlation: float
):
    cov_matrix = np.array([[10**2, correlation * 10**2], [correlation * 10**2, 10**2]])
    X_1_2 = np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=sample_size)
    X_1 = X_1_2[:, 0]
    X_2 = X_1_2[:, 1]
    epsilon = np.random.normal(loc=0, scale=10, size=sample_size)
    Y = beta_0 + beta_1 * X_1 + beta_2 * X_2 + epsilon
    return X_1, X_2, Y


# Estimar con OLS
def estimate_ols(X1, X2, Y):
    X_with_constant = sm.add_constant(np.column_stack((X1, X2)))
    model = sm.OLS(exog=X_with_constant, endog=Y)
    results = model.fit()
    return results.params


# Función de Simulación
def run_simulation(
    n_samples, sample_size, beta_0, beta_1, beta_2, mean: list, correlation: float
):
    estimates = [
        estimate_ols(
            *generate_sample(sample_size, beta_0, beta_1, beta_2, mean, correlation)
        )
        for _ in range(n_samples)
    ]
    return np.array(estimates)

# Estimar con OLS y calcular R^2 de X1 ~ X2
def estimate_r_squared(X1, X2):
    X_with_constant2 = sm.add_constant(X2)  # Add constant to the X2 matrix
    model = sm.OLS(X1, X_with_constant2)  # Fit OLS model with X1 as dependent
    results = model.fit()
    r_squared = results.rsquared  # Get the R-squared value
    return r_squared


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
Esta simulación demuestra el efecto de la multicolinealidad en los estimadores OLS $\\beta_0$, $\\beta_1$ y $\\beta_2$ en el modelo:
$$
y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\epsilon
$$
$$
\\beta_0 = 10, \\beta_1 = 3, \\beta_2 = 5
$$
donde $x_1$ y $x_2$ son variables explicativas.
"""

simulation_content = """
1. **Generación de muestras:**
       - Se crean 500 muestras independientes.
       - Cada muestra contiene 250 observaciones.

2. **Modelado de variables para cada muestra:**
       - X1: Generada con distribución normal
         * Media = 10
         * Desviación estándar = 10
       - X2: Generada con distribución normal
         * Media = 10
         * Desviación estándar = 10
       - $\\epsilon$ (error): Generada con distribución normal
         * Media = 0
         * Desviación estándar = 10
       - Y: Calculada usando la ecuación del modelo
         * $Y = 10 + 3X_1 + 5X_2 + \\epsilon$

3. **Estimación:**
       - Para cada muestra, se estiman $\\beta_0$, $\\beta_1$ y $\\beta_2$ usando OLS.
       - Se recopilan todas las estimaciones para crear los histogramas.

Ajusta el slider para observar cómo la correlación entre $X_1$ y $X_2$ afecta las estimaciones de los coeficientes.
"""

unbiasedness_content = """
Los histogramas deben centrarse cerca de los valores verdaderos $(\\beta_0 = 10, \\beta_1 = 3, \\beta_2 = 5)$, demostrando insesgadez.
"""

variance_content = """
1. **Correlación entre $X_1$ y $X_2$:**
       - Aumentar la correlación entre $X_1$ y $X_2$ hace que sea más difícil distinguir los efectos individuales de cada variable.
       - Resulta en un histograma más ancho para $\\beta_1$ y $\\beta_2$.
       - Los errores estándar de $\\beta_1$ y $\\beta_2$ aumentan.
"""

# Creamos el componente Accordion (las cajitas) con las secciones
markdown_description = create_menu(
    600,
    {"Introducción": intro_content},
    {"Metodología": simulation_content},
    {"Demostración de Insesgadez": unbiasedness_content},
    {"Determinantes de la Varianza": variance_content},
)


############ 2) GRÁFICOS ##############3

# scatter
scatter_content = create_graph(id="scatter", title="Gráfico de dispersión")

# histogram of beta 0
histogram_beta_0_content = create_graph(
    "histogram_estimator_beta_0", "Histograma de estimación β₀"
)

# histogram of beta 1
histogram_beta_1_content = create_graph(
    "histogram_estimator_beta_1", "Histograma de estimación β₁"
)

# histogram of beta 2
histogram_beta_2_content = create_graph(
    "histogram_estimator_beta_2", "Histograma de estimación β₂"
)


######### 3) SLIDERS ##############

sliders_stack = dmc.Stack(
    [
        dmc.Text("Ajuste de parámetros", fw=800),
        dcc.Markdown("Correlación entre $X_1$ y $X_2$", mathjax=True),
        # dmc.Text("Correlación entre X₁ y X₂", fw=600),
        dmc.Slider(
            id="correlation",
            updatemode="drag",
            min=-1,
            max=1,
            step=0.05,
            # marks=[
            #     {"value": x, "label": f"{x:.1f}"} for x in np.arange(-1, 1.01, 0.25)
            # ],
            marks=[{"value": x, "label": f"{x}"} for x in np.arange(-1, 1.01, 0.25)],
            value=0,
            style={"width": "80%"},
        ),
        html.Div(id="r_squared_output"),  
    ],
    justify="center",
    gap="xl",
)

########## ESTRUCTURA ##################3
resumen_content = main_structure(
    menu=markdown_description,
    structure=[
        # Fila 1
        [[paper(sliders_stack)], [paper(scatter_content)]],
        # Fila 2
        [
            # Primer elemento de la fila 2
            [
                paper(histogram_beta_0_content),
                dmc.Space(h=10),
                paper(html.Div(id="beta_0_stats"), ""),
            ],
            # Segundo elemento de la fila 2
            [
                paper(histogram_beta_1_content),
                dmc.Space(h=10),
                paper(html.Div(id="beta_1_stats"), ""),
            ],
        ],
        # Fila 3
        [
            [
                paper(histogram_beta_2_content),
                dmc.Space(h=10),
                paper(html.Div(id="beta_2_stats"), ""),
            ],
            [],
        ],
    ],
)


@app.callback(
    Output("histogram_estimator_beta_0", "figure"),
    Output("histogram_estimator_beta_1", "figure"),
    Output("histogram_estimator_beta_2", "figure"),
    Output("scatter", "figure"),
    Output("beta_0_stats", "children"),
    Output("beta_1_stats", "children"),
    Output("beta_2_stats", "children"),
    Output("r_squared_output", "children"),
    Input("correlation", "value"),
)
def update_histogram(correlation):
    sample_size = 250
    amount_samples = 500
    beta_0 = 10
    beta_1 = 3
    beta_2 = 5

    # ejecutar la simulación
    estimates = run_simulation(
        n_samples=amount_samples,
        sample_size=sample_size,
        beta_0=beta_0,
        beta_1=beta_1,
        beta_2=beta_2,
        mean=[10, 10],
        correlation=correlation,
    )

    betas_0_estimated = estimates[:, 0]
    betas_1_estimated = estimates[:, 1]
    betas_2_estimated = estimates[:, 2]

    # Muestra para el scatter
    X_1, X_2, Y = generate_sample(
        sample_size=sample_size,
        beta_0=beta_0,
        beta_1=beta_1,
        beta_2=beta_2,
        mean=[10, 10],
        correlation=correlation,
    )

      # Calculate R² for the model X1 ~ X2
    r_squared = estimate_r_squared(X1=X_1, X2=X_2)

    # fmt: off
    fig_histogram_0=create_histogram(data=np.array(betas_0_estimated), range_x=[0,20], true_value=beta_0 )
    fig_histogram_1=create_histogram(data=np.array(betas_1_estimated), range_x=[0,6], true_value=beta_1 )
    fig_histogram_2=create_histogram(data=np.array(betas_2_estimated), range_x=[0,10], true_value=beta_2 )
    fig_scatter = create_scatter_matrix_plot(X1=X_1, X2=X_2, Y=Y)
    # fmt: on

    for fig in [fig_histogram_0, fig_histogram_1, fig_histogram_2, fig_scatter]:
        update_layout(fig)

    # Calcular estadísticas
    beta_0_stats = {
        "mean": np.mean(betas_0_estimated),
        "var": np.var(betas_0_estimated),
    }
    beta_1_stats = {
        "mean": np.mean(betas_1_estimated),
        "var": np.var(betas_1_estimated),
    }
    beta_2_stats = {
        "mean": np.mean(betas_2_estimated),
        "var": np.var(betas_2_estimated),
    }

    beta_0_table = create_stats_table(beta_0_stats, beta_0, "β₀")
    beta_1_table = create_stats_table(beta_1_stats, beta_1, "β₁")
    beta_2_table = create_stats_table(beta_2_stats, beta_2, "β₂")

    # Create the R² output as a Markdown for display
    r_squared_output = dcc.Markdown(f"**$R^2$ for $X_1 \\sim X_2$ model**: {r_squared:.4f}", 
    mathjax=True, 
    style={"font-size": "18px"}
    )


    return (
        fig_histogram_0,
        fig_histogram_1,
        fig_histogram_2,
        fig_scatter,
        beta_0_table,
        beta_1_table,
        beta_2_table,
        r_squared_output,
    )


############ LAYOUT ##################
app.layout = build_layout(
    title="Simulación OLS: Multicolinealidad y sus Efectos",
    content=resumen_content,
)

############ RUN SERVER #################
if __name__ == "__main__":
    # port = int(os.environ.get("DASH_PORT"))
    # app.run_server(debug=False, port=port)
    app.run_server(debug=False, host="127.0.0.1")