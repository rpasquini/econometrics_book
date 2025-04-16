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
import argparse
from scipy import stats
from typing import NamedTuple

# Set React version to 18.2.0 which is supported by Dash
dash._dash_renderer._set_react_version("18.2.0")

# Define style constants to match the original
style = {
    "width": "100%",
    "marginTop": 20,
    "marginBottom": 20,
}

# Define title style to match the original
title_style = {
    "marginBottom": 0,
    "marginLeft": 0,
    "marginTop": 10,
    "textAlign": "left",
}

# Helper function for hovercards
def create_title_with_hovercard(title, information):
    return dmc.Group(
        [
            dmc.Text(title, fw=800),
            dmc.HoverCard(
                withArrow=True,
                width=200,
                shadow="md",
                children=[
                    dmc.HoverCardTarget(
                        dmc.ThemeIcon(
                            "?",
                            size="sm",
                            radius="xl",
                            color="gray",
                            variant="light",
                        )
                    ),
                    dmc.HoverCardDropdown(
                        dmc.Text(
                            information,
                            size="sm",
                        )
                    ),
                ],
            ),
        ],
    )

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
    elif error_dist == "gamma":
        epsilon = np.random.gamma(shape=2, scale=3, size=sample_size)
    elif error_dist == "beta":
        epsilon = 10 * np.random.beta(a=8, b=6, size=sample_size)
    else:
        raise ValueError(f"Error distribution --{error_dist}-- not recognized.")
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
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=30, b=30))
    return fig


# Crear scatter
def create_scatter_plot(X, Y):
    """
    Crea un gráfico de dispersión.
    """
    fig = px.scatter(
        x=X,
        y=Y,
        labels={"x": "X", "y": "Y"},
    )
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=30, b=30))
    return fig


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
    y_t = stats.t.pdf(x, df=degrees_of_freedom)
    
    # Crea el gráfico de la distribución t teórica
    fig_t = go.Figure(data=[go.Scatter(x=x, y=y_t, line=dict(color="black", width=2))])
    fig_t.update_layout(
        xaxis_title="t",
        yaxis_title="Densidad",
        height=300,
        margin=dict(l=35, r=35, t=30, b=30)
    )
    
    # Crea el histograma de los t-estadísticos simulados
    fig_hist = px.histogram(
        t_stats,
        color_discrete_sequence=["indianred"],
        histnorm="probability density",
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


# Create stats table with centered text
def create_stats_table(
    stats={"mean": 0, "std": 0},
    true_value=0,
    parameter_name="unknown",
    custom=False,
    custom_title="",
    custom_dict_varnames=dict(),
):
    if custom:
        table = dmc.Table(
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Estadística", style={"textAlign": "center"}),
                            html.Th("Valor", style={"textAlign": "center"}),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(custom_dict_varnames.get(k, k), style={"textAlign": "center"}),
                                html.Td(f"{v:.4f}", style={"textAlign": "center"}),
                            ]
                        )
                        for k, v in stats.items()
                    ]
                ),
            ],
        )
    else:
        table = dmc.Table(
            striped=True,
            highlightOnHover=True,
            withTableBorder=True,
            withColumnBorders=True,
            children=[
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Estadística", style={"textAlign": "center"}),
                            html.Th("Valor", style={"textAlign": "center"}),
                        ]
                    )
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td("Media", style={"textAlign": "center"}),
                                html.Td(f"{stats['mean']:.4f}", style={"textAlign": "center"}),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Desviación estándar", style={"textAlign": "center"}),
                                html.Td(f"{stats['std']:.4f}", style={"textAlign": "center"}),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("Valor verdadero", style={"textAlign": "center"}),
                                html.Td(f"{true_value}", style={"textAlign": "center"}),
                            ]
                        ),
                    ]
                ),
            ],
        )
    return table


############### INSTANCIA DE LA APP ###############
app = Dash(__name__, suppress_callback_exceptions=True)

############# ELEMENTOS DE LA APP (LAYOUT) ##################

# Define the layout with enhanced Dash Mantine Components
app.layout = dmc.MantineProvider(
    theme={
        "colorScheme": "light",
        "primaryColor": "blue",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        },
    },
    children=[
        html.Div(
            style={"backgroundColor": "#f8f9fa", "minHeight": "100vh", "width": "100%"},
            children=[
                dmc.Container(
                    size="lg",
                    style=style,
                    children=[
                        # Title with Mantine components
                        dmc.Paper(
                            p="md",
                            withBorder=False,
                            radius="md",
                            style={"backgroundColor": "#f8f9fa"},
                            children=[
                                dmc.Group(
                                    justify="space-between",
                                    children=[
                                        dmc.Title("Simulación OLS: Robustez de la Prueba T", order=1, style=title_style),
                                    ],
                                ),
                            ],
                        ),
                        dmc.Space(h=20),
                        
                        # Accordion for static information
                        dmc.Accordion(
                            children=[
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl(
                                            html.Span("Introducción", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
                                                Esta simulación explora la robustez de la prueba t y su distribución bajo diferentes distribuciones de error.
                                                En particular, se analiza el impacto de violar el supuesto de normalidad del término de error en la prueba t.
                                                """,
                                                mathjax=True
                                            )
                                        ),
                                    ],
                                    value="intro"
                                ),
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl(
                                            html.Span("Metodología", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
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
                                                """,
                                                mathjax=True
                                            )
                                        ),
                                    ],
                                    value="methodology"
                                ),
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl(
                                            html.Span("Demostración de Insesgadez", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
                                                La distribución del t-estadístico debería ser similar a la distribución t teórica cuando el error es normal.
                                                """,
                                                mathjax=True
                                            )
                                        ),
                                    ],
                                    value="unbiasedness"
                                ),
                                dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl(
                                            html.Span("Determinantes de la Varianza", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
                                                1. **Distribución del error no normal:**
                                                       - El histograma de los t-estadísticos simulados puede desviarse de la distribución t teórica.
                                                       - El p-valor de la prueba de hipótesis puede ser inexacto.

                                                2. **Tamaño de la muestra:**
                                                       - Aumentar el tamaño de la muestra hace que la distribución del t-estadístico sea más similar a la normal, incluso si la distribución del error no es normal.
                                                       - La prueba t se vuelve más robusta a la violación del supuesto de normalidad.
                                                """,
                                                mathjax=True
                                            )
                                        ),
                                    ],
                                    value="variance"
                                ),
                            ],
                            value="intro"
                        ),
                        dmc.Space(h=20),
                        
                        # Main content in a grid layout
                        dmc.Grid(
                            [
                                # First row: Sliders and Error Histogram
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white", "height": "400px", "display": "flex", "flexDirection": "column"},
                                            children=[
                                                create_title_with_hovercard(
                                                    "Ajuste de parámetros",
                                                    "Estos parámetros cambian la distribución del error y el tamaño de la muestra."
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Markdown(
                                                    "Distribución del error", mathjax=True
                                                ),
                                                dmc.Select(
                                                    id="error_distribution",
                                                    data=[
                                                        {"label": "Normal", "value": "normal"},
                                                        {"label": "Exponencial", "value": "exponential"},
                                                        {"label": "Chi-cuadrada", "value": "chi2"},
                                                        {"label": "Uniforme", "value": "uniform"},
                                                        {"label": "Laplace", "value": "laplace"},
                                                        {"label": "Gamma", "value": "gamma"},
                                                        {"label": "Beta", "value": "beta"},
                                                    ],
                                                    placeholder="Seleccionar distribución",
                                                    value="normal",
                                                    clearable=False,
                                                    style={"width": "80%"},
                                                ),
                                                dmc.Space(h=10),
                                                dcc.Markdown("Tamaño de la muestra $(n)$", mathjax=True),
                                                dmc.Slider(
                                                    id="sample_size",
                                                    min=5,
                                                    step=5,
                                                    max=1000,
                                                    marks=[{"value": x, "label": f"{x}"} for x in range(50, 1001, 200)],
                                                    value=100,
                                                    color="blue",
                                                    size="md",
                                                    style={"width": "80%"},
                                                ),
                                                # Add a spacer to push content to the top
                                                html.Div(style={"flexGrow": 1}),
                                            ]
                                        ),
                                    ]
                                ),
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white"},
                                            children=[
                                                create_title_with_hovercard(
                                                    "Histograma del error",
                                                    "Muestra la distribución del término de error en la muestra."
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="histogram_epsilon",
                                                    figure=create_histogram(
                                                        np.random.normal(0, 10, 100),
                                                        [-100, 100],
                                                        "$\\epsilon$"
                                                    )
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                
                                # Second row: T-Distribution and Scatter Plot
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white"},
                                            children=[
                                                create_title_with_hovercard(
                                                    "Distribución T simulada vs. Teórica",
                                                    "Compara la distribución empírica de los t-estadísticos con la distribución t teórica."
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="t_distribution_plot",
                                                    figure=create_t_distribution_plot(
                                                        np.random.normal(0, 1, 1000),
                                                        100
                                                    )
                                                ),
                                                dmc.Space(h=10),
                                                dmc.Alert(
                                                    color="blue",
                                                    variant="light",
                                                    style={"backgroundColor": "white"},
                                                    children=[
                                                        html.Div(id="percentiles_alpha")
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white"},
                                            children=[
                                                create_title_with_hovercard(
                                                    "Gráfico de dispersión",
                                                    "Muestra la relación entre X e Y en la muestra."
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="scatter",
                                                    figure=create_scatter_plot(
                                                        np.random.normal(10, 1, 100),
                                                        np.random.normal(10, 10, 100)
                                                    )
                                                )
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ]
                )
            ]
        )
    ]
)

################# CALLBACK ###############
@app.callback(
    Output("histogram_epsilon", "figure"),
    Output("t_distribution_plot", "figure"),
    Output("scatter", "figure"),
    Output("percentiles_alpha", "children"),
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
    fig_t_dist = create_t_distribution_plot(
        t_statistic, degrees_of_freedom=sample_size - 1
    )
    rejects_null = perform_hypothesis_test(
        t_stats=t_statistic, df=sample_size - 2, alpha=0.05
    )

    percentage_rejects_null = rejects_null / 1000

    # Crear tabla de percentiles
    percentiles_alpha = create_stats_table(
        custom=True,
        custom_title="Comparación de nivel de signficancia",
        custom_dict_varnames={
            "Valor simulado": f"{percentage_rejects_null:.4f}",
            "Valor real": "0.05",
            "Diferencia": f"{percentage_rejects_null-0.05:.4f}",
        },
        parameter_name=r"$\alpha$",
    )

    return fig_epsilon, fig_t_dist, fig_scatter, percentiles_alpha

############ RUN SERVER #################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true', help='Run in server mode')
    args = parser.parse_args()
    
    try:
        if args.server:
            print("Running on server mode")
            app.run(debug=True, host="0.0.0.0", port=8050, processes=6, threaded=False)
        else:
            print("Running on local mode")
            app.run_server(debug=True, port=8066, processes=6, threaded=False)
    except Exception as e:
        print(f"Application error: {e}") 