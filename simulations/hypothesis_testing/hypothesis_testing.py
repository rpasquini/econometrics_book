######## IMPORTACIÓN DE LIBRERIAS ##############
import dash
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import os
import pprint
import sys
from scipy import stats
from typing import NamedTuple

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


# Funciones para la lógica de la aplicación
def generate_sample(sample_size, beta_0, beta_1, x_mean, x_std, error_dist):
    """
    Genera una muestra de datos con diferentes distribuciones de error.
    """
    X = np.random.normal(loc=x_mean, scale=1, size=sample_size)
    if error_dist == "normal":
        epsilon = np.random.normal(loc=0, scale=10, size=sample_size)
    elif error_dist == "exponential":
        epsilon = np.random.exponential(scale=10, size=sample_size)
    elif error_dist == "chi2":
        epsilon = np.random.chisquare(df=10, size=sample_size)
    elif error_dist == "uniform":
        epsilon = np.random.uniform(low=-10, high=10, size=sample_size)
    elif error_dist == "laplace":
        epsilon = np.random.laplace(loc=0, scale=10, size=sample_size)
    else:
        raise ValueError("Error distribution not recognized.")
    Y = beta_0 + beta_1 * X + epsilon
    return X, Y, epsilon


class OLS_ESTIMATION(NamedTuple):
    params: np.array
    standard_error: np.array


# Estimar con OLS
def estimate_ols(X, Y, epsilon) -> OLS_ESTIMATION:
    """
    Estima los coeficientes del modelo OLS.
    """
    X_with_constant = sm.add_constant(X)
    model = sm.OLS(exog=X_with_constant, endog=Y)
    results = model.fit()
    return OLS_ESTIMATION(
        params=results.params, standard_error=results.bse[1]
    )  # Return coefficients and standard error


class simulation_results(NamedTuple):
    estimated_params: np.array
    estimated_std_err: np.array


# Función de Simulación
def run_simulation(n_samples, sample_size, beta_0, beta_1, x_mean, x_std, error_dist):
    """
    Ejecuta la simulación de regresión.
    """
    estimates = []
    standard_errors = []
    for _ in range(n_samples):
        estimated = estimate_ols(
            *generate_sample(sample_size, beta_0, beta_1, x_mean, x_std, error_dist)
        )
        estimates.append(estimated.params)
        standard_errors.append(estimated.standard_error)

    return simulation_results(
        estimated_params=np.array(estimates),
        estimated_std_err=np.array(standard_errors),
    )


# Crear histograma
def create_histogram(data, range_x, x_label):
    """
    Crea un histograma.
    """
    fig = px.histogram(
        x=data,
        color_discrete_sequence=["indianred"],
        range_x=range_x,
        labels={"x": x_label},
    )
    return fig


# Crear scatter
def create_scatter_plot(X, Y):
    """
    Crea un gráfico de dispersión.
    """
    return px.scatter(
        x=X,
        y=Y,
        labels={"x": "X", "y": "Y"},
        # range_x=[-150, 150],
        # range_y=[-410, 410],
    )


def calculate_stats(t_stats):
    """
    Calcula las estadísticas de los t-estadísticos.
    """
    return {
        "mean": np.mean(t_stats),
        "std": np.std(t_stats),
    }


# Crear gráfica de la distribución t
def create_t_distribution_plot(t_stats, degrees_of_freedom):
    """
    Crea una gráfica de la distribución t con la superposición de la distribución teórica.
    """
    # Rango para la distribución t teórica
    x = np.linspace(-10, 10, 100)
    print("GRADOS DE LIBERTAD", degrees_of_freedom)
    y_t = stats.t.pdf(x, df=degrees_of_freedom)
    print("Distribución de t students", y_t)
    # Crea el gráfico de la distribución t teórica
    fig_t = go.Figure(data=[go.Scatter(x=x, y=y_t, line=dict(color="black", width=2))])
    fig_t.update_layout(
        # title="Distribución T Teórica",
        xaxis_title="t",
        yaxis_title="Densidad",
    )
    # Crea el histograma de los t-estadísticos simulados
    fig_hist = px.histogram(
        t_stats,
        color_discrete_sequence=["indianred"],
        histnorm="probability density",
        # nbins=20,
        # title="Histograma de T-Estadísticos",
        labels={"x": "t-estadístico"},
    )
    # Superpone el histograma sobre la distribución t teórica
    fig_t.add_traces(fig_hist.data)
    fig_t.update_layout(showlegend=False)
    return fig_t


def perform_hypothesis_test(t_stats, df, alpha):
    """
    Realiza una prueba de hipótesis para beta_1 = 0.
    """
    # Calcula el p-valor
    p_value = 2 * stats.t.cdf(-np.abs(t_stats), df=df)
    # Decide si rechazar la hipótesis nula
    reject_null = p_value < alpha
    rejects_amount = sum(reject_null)
    return rejects_amount


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
Esta simulación explora la robustez de la prueba t y su distribución bajo diferentes distribuciones de error.
En particular, se analiza el impacto de violar el supuesto de normalidad del término de error en la prueba t.
"""

simulation_content = """
1. **Generación de muestras:**
       - Se crean 500 muestras independientes.
       - Se puede ajustar el tamaño de la muestra (slider_value_sample).
       - X: Generada con distribución normal
         * Media = 10
         * Desviación estándar = 10
       - $\\epsilon$ (error): Generada con la distribución seleccionada por el usuario (dropdown_value).
         * Se asume un parámetro de escala = 10 para todas las distribuciones.
       - Y: Calculada usando la ecuación del modelo. Notar que en esta simulación, se asume  $\\beta_1 = 0$ ya que supondremos que nos interesa evaluar la hipotesis de efecto nulo.
         * $Y = 10 + 3X + \\epsilon (\\beta_0 = 10, \\beta_1 = 0)$

2. **Estimación y cálculo de t-estadísticos:**
       - Para cada muestra, se estiman $\\beta_0$ y $\\beta_1$ usando OLS.
       - Se calcula el t-estadístico para $\\beta_1$ como el cociente entre el coeficiente y su error estándar.

3. **Visualización:**
       - Se muestra el histograma de los t-estadísticos simulados.
       - Se superpone la distribución t teórica (asumiendo normalidad).

4. **Prueba de hipótesis:**
       - Se realiza una prueba de hipótesis para $\\beta_1 = 0$ con un nivel de significancia de 0.05.
       - Se muestra el p-valor de la prueba.

Ajusta los sliders y el menú desplegable para observar cómo la distribución del error y el tamaño de la muestra afectan la distribución del t-estadístico y la prueba de hipótesis.
"""

unbiasedness_content = """
La distribución del t-estadístico debería ser similar a la distribución t teórica cuando el error es normal.
"""

variance_content = """
1. **Distribución del error no normal:**
       - El histograma de los t-estadísticos simulados puede desviarse de la distribución t teórica.
       - El p-valor de la prueba de hipótesis puede ser inexacto.

2. **Tamaño de la muestra:**
       - Aumentar el tamaño de la muestra hace que la distribución del t-estadístico sea más similar a la normal, incluso si la distribución del error no es normal.
       - La prueba t se vuelve más robusta a la violación del supuesto de normalidad.
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
# histogram_beta_0_content = create_graph(
#     "histogram_estimator_beta_0", "Histograma de estimación β₀"
# )

# # histogram of beta 1
# histogram_beta_1_content = create_graph(
#     "histogram_estimator_beta_1", "Histograma de estimación β₁"
# )

# t-statistic distribution
t_distribution_content = create_graph(
    "t_distribution_plot", "Distribución T simulada vs. Teórica"
)


histogram_error = create_graph("histogram_epsilon", "Histograma del error")

######### 3) SLIDERS ##############3

sliders_stack = dmc.Stack(
    [
        dmc.Text("Ajuste de parámetros", fw=800),
        # dmc.Text("Desviación estándar del error (Sigma ε)", fw=600),
        # dcc.Markdown("Desviación estándar de $X$ $(\\sigma_{X})$", mathjax=True),
        # dmc.Slider(
        #     id="std_error_x",
        #     updatemode="drag",
        #     min=1,
        #     max=50,
        #     marks=[{"value": 1, "label": "1"}]
        #     + [{"value": x, "label": f"{x}"} for x in range(10, 51, 10)],
        #     value=10,
        #     style={"width": "80%"},
        # ),
        dcc.Markdown("Distribución del error", mathjax=True),
        dmc.Select(
            id="error_distribution",
            data=[
                {"label": "Normal", "value": "normal"},
                {"label": "Exponencial", "value": "exponential"},
                {"label": "Chi-cuadrada", "value": "chi2"},
                {"label": "Uniforme", "value": "uniform"},
                {"label": "Laplace", "value": "laplace"},
            ],
            placeholder="Seleccionar distribución",
            value="normal",
            clearable=False,
            style={"width": "80%"},
        ),
        # dmc.Space(h=2),
        dcc.Markdown("Tamaño de la muestra $(n)$", mathjax=True),
        dmc.Slider(
            id="sample_size",
            updatemode="drag",
            min=50,
            step=50,
            max=1000,
            marks=[{"value": x, "label": f"{x}"} for x in range(50, 1001, 200)],
            value=100,
            style={"width": "80%"},
        ),
    ],
    justify="center",
    gap="xl",
)

# Dropdown para la distribución del error
# dropdown_error_distribution = dmc.Stack(
#     [
#         dmc.Text("Distribución del error", fw=800),
#         dmc.Select(
#             id="error_distribution",
#             data=[
#                 {"label": "Normal", "value": "normal"},
#                 {"label": "Exponencial", "value": "exponential"},
#                 {"label": "Chi-cuadrada", "value": "chi2"},
#                 {"label": "Uniforme", "value": "uniform"},
#                 {"label": "Laplace", "value": "laplace"},
#             ],
#             placeholder="Seleccionar distribución",
#             value="normal",
#             clearable=False,
#             style={"width": "80%"},
#         ),
#     ],
#     justify="center",
#     gap="xl",
# )

# Definir un estilo para los títulos
title_style = {
    "marginBottom": 0,
    "marginLeft": 40,
    "marginTop": 20,
    "textAlign": "left",
}

MODIFIED_HEIGHT = lambda x: str(int(COMPONENT_HEIGHT.replace("px", "")) + x) + "px"

########## ESTRUCTURA ##################3
resumen_content = main_structure(
    menu=markdown_description,
    structure=[
        # Fila 1
        [
            [
                paper(sliders_stack),
            ],
            [paper(histogram_error)],
        ],
        # Fila 2
        # [[paper(histogram_beta_0_content)], [paper(histogram_beta_1_content)]],
        # Fila 3
        [
            [
                paper(t_distribution_content),
                dmc.Space(h=10),
                paper(html.Div(id="percentiles"), ""),
            ],
            [paper(scatter_content)],
        ],
    ],
)


################# CALLBACK ###############
@app.callback(
    Output("histogram_epsilon", "figure"),
    Output("t_distribution_plot", "figure"),
    Output("percentiles", "children"),
    Output("scatter", "figure"),
    Input("error_distribution", "value"),
    Input("sample_size", "value"),
)
def update_histogram(error_dist, sample_size):
    # Parámetros de la simulación
    n_samples = 1000
    beta_0, beta_1 = 10, 0
    x_mean = 10
    alpha = 0.05
    x_std = 1

    # Ejecutar la simulación
    estimates = run_simulation(
        n_samples,
        sample_size,
        beta_0,
        beta_1,
        x_mean,
        x_std,
        error_dist,
    )

    params = estimates.estimated_params
    std_error_beta_1 = estimates.estimated_std_err

    # print(params.shape, type(params), params)
    # print(std_error_beta_1.shape, type(std_error_beta_1), std_error_beta_1)

    # Crear histogramas
    X, Y, epsilon = generate_sample(
        sample_size, beta_0, beta_1, x_mean, x_std, error_dist
    )
    fig_histogram_0 = create_histogram(params[:, 0], [0, 20], "$\\beta_0$")
    fig_histogram_1 = create_histogram(params[:, 1], [0, 6], "$\\beta_1$")
    fig_scatter = create_scatter_plot(X, Y)
    fig_epsilon = create_histogram(epsilon, [-50, 50], "$\\epsilon$")

    # Crear gráfica de la distribución t
    t_statistic = params[:, 1] / std_error_beta_1
    print(t_statistic)

    fig_t_dist = create_t_distribution_plot(
        t_statistic, degrees_of_freedom=sample_size - 1
    )
    rejects_null = perform_hypothesis_test(
        t_stats=t_statistic, df=sample_size - 1, alpha=0.05
    )

    # Crear el histograma del error
    fig_hist_error = create_histogram(epsilon, [-100, 100], "x")

    # Crear tabla de percentiles
    percentiles = create_stats_table(
        {"mean": np.mean(params[:, 1]), "std": np.std(params[:, 1])},
        0,
        "β₁",
    )
    # Actualizar el diseño de los gráficos
    for fig in [fig_hist_error, fig_t_dist]:
        update_layout(fig)

    return fig_hist_error, fig_t_dist, percentiles, fig_scatter


############ LAYOUT ##################
app.layout = build_layout(
    title="Simulación OLS: Robustez de la Prueba T",
    content=resumen_content,
)

############ RUN SERVER #################
if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT"))
    app.run_server(host="0.0.0.0", debug=True, port=port)
    # app.run_server(debug=False, port=8070)
