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
import sys
import argparse

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

############### FUNCIONES PROPIAS DE ESTA SIMULACIÓN ################

# Funciones para la lógica de la aplicación
def generate_sample(sample_size, beta_0, beta_1, x_mean, x_std, error_std):
    X = np.random.normal(loc=x_mean, scale=x_std, size=sample_size)
    epsilon = np.random.normal(loc=0, scale=error_std, size=sample_size)
    Y = beta_0 + beta_1 * X + epsilon
    return X, Y


# Estimar con OLS
def estimate_ols(X, Y):
    X_with_constant = sm.add_constant(X)
    model = sm.OLS(exog=X_with_constant, endog=Y)
    results = model.fit()
    return results.params


# Función de Simulación
def run_simulation(n_samples, sample_size, beta_0, beta_1, x_mean, x_std, error_std):
    estimates = [
        estimate_ols(
            *generate_sample(sample_size, beta_0, beta_1, x_mean, x_std, error_std)
        )
        for _ in range(n_samples)
    ]
    return np.array(estimates)


# Crear histograma
def create_histogram(data, range_x, true_value, x_label):
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
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=30, b=30))
    return fig


# Crear scatter
def create_scatter_plot(X, Y):
    fig = px.scatter(
        x=X,
        y=Y,
        labels={"x": "X", "y": "Y"},
        range_x=[-150, 150],
        range_y=[-410, 410],
    )
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=30, b=30))
    return fig


def calculate_stats(beta_0_list, beta_1_list):
    return {
        "mean_beta_0": np.mean(beta_0_list),
        "var_beta_0": np.var(beta_0_list),
        "mean_beta_1": np.mean(beta_1_list),
        "var_beta_1": np.var(beta_1_list),
    }


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
                                        dmc.Title("Simulación: Insesgadez y Varianza de Estimadores OLS", order=1, style=title_style),
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
                                                Esta simulación demuestra la insesgadez y los determinantes de la varianza de los estimadores OLS de $\\beta_0$ y $\\beta_1$ en el modelo poblacional:
                                                $$
                                                Y_i = \\beta_0 + \\beta_1 X_i + \\varepsilon_i
                                                $$

                                                En particular, supondremos que $\\beta_0 = 10$ y $\\beta_1 = 3$.

                                                $$
                                                Y_i = 10 + 3 X_i + \\varepsilon_i
                                                $$
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
                                            html.Span("Demostración de Insesgadez", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
                                                La propiedad de insesgadez de los estimadores OLS dice que, en valor esperado, las estimaciones son iguales a los verdaderos valores poblacionales. En este caso, esto sería:

                                                $$
                                                E[\\hat{\\beta}_0] = 10
                                                $$
                                                $$
                                                E[\\hat{\\beta}_1] = 3
                                                $$

                                                Para demostrar numéricamente que esta propiedad se cumple, mostramos que la simulación de múltiples estimaciones OLS cumplen que :

                                                1. La distribución de las estimación de los respectivos coeficientes se centra cerca de los valores verdaderos $(\\beta_0 = 10, \\beta_1 = 3)$.
                                                2. El promedio de las estimaciones es cercano a los valores verdaderos.
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
                                            html.Span("Determinantes de la Varianza de la Estimación", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
                                                La varianza de los estimadores OLS ( $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$ está determinada por dos factores:

                                                1. **Varianza del término de error:** 
                                                       - Un aumento en el error del modelo se traduce en mayor imprecisión de la estimación del modelo.
                                                       - Esto se refleja en histogramas más anchos para $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$.
                                                       - Mayor varianza estimada de las estimaciones realizadas (Ver valor en la tabla)).

                                                2. **Varianza de $X$:**
                                                       - Ceteris paribus, una  mayor dispersión en los valores de la variable explicativa $X$ reduce la imprecisión de la estimación $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$ .
                                                       - Esto se refleja en histogramas más estrecho para $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$.
                                                       - Menor varianza estimada de las estimaciones realizadas (Ver valor en la tabla)).
                                                """,
                                                mathjax=True
                                            )
                                        ),
                                    ],
                                    value="variance"
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
                                                       - Cada muestra contiene 250 observaciones.

                                                2. **Modelado de variables para cada muestra:**
                                                       - X: Generada con distribución normal
                                                         * Media = 10
                                                         * Desviación estándar = ajustable por el usuario (slider_value_x)
                                                       - $\\epsilon$ (error): Generada con distribución normal
                                                         * Media = 0
                                                         * Desviación estándar = ajustable por el usuario (slider_value)
                                                       - Y: Calculada usando el modelo poblacional supuesto:
                                                         * $Y = 10 + 3X + \\varepsilon $

                                                3. **Estimación:**
                                                       - Para cada muestra, se estiman $\\beta_0$ y $\\beta_1$ usando OLS.
                                                       - Se recopilan todas las estimaciones para crear los histogramas.

                                              Ajusta los sliders para observar cómo la variabilidad en $X$ y $\\epsilon$ afecta las estimaciones de $\\beta_0$ y $\\beta_1$.
                                              """,
                                                mathjax=True
                                            )
                                        ),
                                    ],
                                    value="methodology"
                                ),
                            ],
                            value="intro"
                        ),
                        dmc.Space(h=20),
                        
                        # Main content in a grid layout
                        dmc.Grid(
                            [
                                # First row: Sliders and Scatter plot
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white", "height": "400px", "display": "flex", "flexDirection": "column"},
                                            children=[
                                                dmc.Title("Ajuste de parámetros", order=3, style=title_style),
                                                dmc.Divider(my="sm"),
                                                dcc.Markdown(
                                                    "*Desviación estándar del error* $(\\sigma_{\\varepsilon})$", mathjax=True
                                                ),
                                                dmc.Slider(
                                                    id="std_error",
                                                    min=1,
                                                    max=50,
                                                    marks=[{"value": 1, "label": "1"}] + [{"value": x, "label": f"{x}"} for x in range(10, 51, 10)],
                                                    value=10,
                                                    color="blue",
                                                    size="md",
                                                    style={"width": "80%"},
                                                ),
                                                dmc.Space(h=10),
                                                dcc.Markdown("Desviación estándar de $X$ $(\\sigma_{X})$", mathjax=True),
                                                dmc.Slider(
                                                    id="std_error_x",
                                                    min=1,
                                                    max=50,
                                                    marks=[{"value": 1, "label": "1"}] + [{"value": x, "label": f"{x}"} for x in range(10, 51, 10)],
                                                    value=10,
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
                                                dmc.Title("Gráfico de dispersión", order=3, style=title_style),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="scatter",
                                                    figure=create_scatter_plot(
                                                        *generate_sample(250, 10, 3, 10, 10, 10)
                                                    )
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                
                                # Second row: Histograms
                                dmc.GridCol(
                                    span=6,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white"},
                                            children=[
                                                dmc.Title("Distribución de estimaciones de β₀", order=3, style=title_style),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="histogram_estimator_beta_0",
                                                    figure=create_histogram(
                                                        run_simulation(500, 250, 10, 3, 10, 10, 10)[:, 0],
                                                        [0, 20],
                                                        10,
                                                        "Estimación de β₀"
                                                    )
                                                ),
                                                dmc.Space(h=10),
                                                dmc.Alert(
                                                    title="Estadísticas",
                                                    color="blue",
                                                    variant="light",
                                                    style={"backgroundColor": "white"},
                                                    children=[
                                                        html.Div(id="beta_0_stats")
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
                                                dmc.Title("Distribución de estimaciones de β₁", order=3, style=title_style),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="histogram_estimator_beta_1",
                                                    figure=create_histogram(
                                                        run_simulation(500, 250, 10, 3, 10, 10, 10)[:, 1],
                                                        [0, 6],
                                                        3,
                                                        "Estimación de β₁"
                                                    )
                                                ),
                                                dmc.Space(h=10),
                                                dmc.Alert(
                                                    title="Estadísticas",
                                                    color="blue",
                                                    variant="light",
                                                    style={"backgroundColor": "white"},
                                                    children=[
                                                        html.Div(id="beta_1_stats")
                                                    ]
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
    Output("histogram_estimator_beta_0", "figure"),
    Output("histogram_estimator_beta_1", "figure"),
    Output("scatter", "figure"),
    Output("beta_0_stats", "children"),
    Output("beta_1_stats", "children"),
    Input("std_error", "value"),
    Input("std_error_x", "value"),
)
def update_histogram(error_std, x_std):
    # Parámetros de la simulación
    n_samples = 500
    sample_size = 250
    beta_0, beta_1 = 10, 3
    x_mean = 10

    # Ejecutar la simulación
    estimates = run_simulation(
        n_samples, sample_size, beta_0, beta_1, x_mean, x_std, error_std
    )

    # Crear histogramas
    fig_histogram_0 = create_histogram(
        estimates[:, 0], [0, 20], beta_0, "Estimación de β₀"
    )
    fig_histogram_1 = create_histogram(
        estimates[:, 1], [0, 6], beta_1, "Estimación de β₁"
    )

    # Crear gráfico de dispersión
    X, Y = generate_sample(sample_size, beta_0, beta_1, x_mean, x_std, error_std)
    fig_scatter = create_scatter_plot(X, Y)

    # Calcular estadísticas
    beta_0_stats = {"mean": np.mean(estimates[:, 0]), "std": np.std(estimates[:, 0])}
    beta_1_stats = {"mean": np.mean(estimates[:, 1]), "std": np.std(estimates[:, 1])}

    # Crear tablas de estadísticas con Mantine components
    beta_0_table = dmc.Table(
        striped=True,
        highlightOnHover=True,
        withTableBorder=True,
        withColumnBorders=True,
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("Estadística"),
                        html.Th("Valor"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Media"),
                            html.Td(f"{beta_0_stats['mean']:.4f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Desviación estándar"),
                            html.Td(f"{beta_0_stats['std']:.4f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Valor verdadero"),
                            html.Td(f"{beta_0}"),
                        ]
                    ),
                ]
            ),
        ]
    )

    beta_1_table = dmc.Table(
        striped=True,
        highlightOnHover=True,
        withTableBorder=True,
        withColumnBorders=True,
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("Estadística"),
                        html.Th("Valor"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td("Media"),
                            html.Td(f"{beta_1_stats['mean']:.4f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Desviación estándar"),
                            html.Td(f"{beta_1_stats['std']:.4f}"),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td("Valor verdadero"),
                            html.Td(f"{beta_1}"),
                        ]
                    ),
                ]
            ),
        ]
    )

    return fig_histogram_0, fig_histogram_1, fig_scatter, beta_0_table, beta_1_table

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