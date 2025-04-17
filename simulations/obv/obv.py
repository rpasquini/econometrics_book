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
def calculate_stats(beta_1_list):
    return {
        "mean_beta_1": np.mean(beta_1_list),
        "var_beta_1": np.var(beta_1_list),
    }


# Funciones para la lógica de la aplicación
def generate_sample(
    sample_size, beta_0, beta_1, beta_2, mean: list, correlation: float
):
    cov_matrix = np.array([[10**2, correlation * 10**2], [correlation * 10**2, 10**2]])
    X_1_U = np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=sample_size)
    X_1 = X_1_U[:, 0]
    U = X_1_U[:, 1]
    epsilon = np.random.normal(loc=0, scale=10, size=sample_size)
    Y = beta_0 + beta_1 * X_1 + beta_2 * U + epsilon
    return X_1, U, Y


# Estimar con OLS
def estimate_ols(X1, U, Y):
    X_with_constant = sm.add_constant(X1)
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
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=30, b=30))
    return fig


# Crear scatter
def create_scatter_matrix_plot(X1, U, Y):
    fig = px.scatter_matrix(
        data_frame=pd.DataFrame({"X1": X1, "U": U, "Y": Y}),
        dimensions=["X1", "U", "Y"],
        color_discrete_sequence=["indianred"],
    )
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=30, b=30))
    return fig


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
app = Dash(__name__, suppress_callback_exceptions=True, url_base_pathname='/obv/')

# Add Google Translate script to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script type="text/javascript">
            function googleTranslateElementInit() {
                new google.translate.TranslateElement({
                    pageLanguage: 'es',
                    includedLanguages: 'en, fr, de, it, pt, ru, es, zh-CN',
                    layout: google.translate.TranslateElement.InlineLayout.SIMPLE
                }, 'google_translate_element');
            }
        </script>
        <script type="text/javascript" src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
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
                                        dmc.Title("Simulación OLS: Sesgo por Variable Omitida", order=1, style=title_style),
                                        # Add Google Translate element
                                        html.Div(
                                            id="google_translate_element",
                                            style={"marginTop": "10px"}
                                        ),
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
                                                Esta simulación demuestra el efecto del sesgo por variable omitida en el estimador OLS $\\beta_1$ en el modelo:
                                                $$
                                                y = \\beta_0 + \\beta_1 x_1 + \\beta_2 u + \\epsilon
                                                $$
                                                $$
                                                \\beta_0 = 10, \\beta_1 = 3, \\beta_2 = 5
                                                $$
                                                donde $x_1$ es la variable explicativa incluida y $u$ es la variable omitida.
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
                                                       - Cada muestra contiene 250 observaciones.

                                                2. **Modelado de variables para cada muestra:**
                                                       - X1: Generada con distribución normal
                                                         * Media = 10
                                                         * Desviación estándar = 10
                                                       - U: Generada con distribución normal
                                                         * Media = 10
                                                         * Desviación estándar = 10
                                                       - $\\epsilon$ (error): Generada con distribución normal
                                                         * Media = 0
                                                         * Desviación estándar = 10
                                                       - Y: Calculada usando la ecuación del modelo
                                                         * $Y = 10 + 3X_1 + 5U + \\epsilon$

                                                3. **Estimación:**
                                                       - Se estima $\\beta_1$ usando OLS con el modelo incompleto $y = \\beta_0 + \\beta_1 x_1 + \\epsilon$
                                                       - Se recopilan todas las estimaciones para crear el histograma.

                                                Ajusta el slider para observar cómo la correlación entre $X_1$ y $U$ afecta las estimaciones de $\\beta_1$.
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
                                                El histograma debe centrarse cerca del valor verdadero $(\\beta_1 = 3)$, demostrando insesgadez.
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
                                            html.Span("Determinantes del Sesgo", style={"fontWeight": "bold", "fontSize": "18px"}),
                                        ),
                                        dmc.AccordionPanel(
                                            dcc.Markdown(
                                                """
                                                1. **Correlación entre $X_1$ y $U$:**
                                                       - Aumentar la correlación entre $X_1$ y $U$ crea un sesgo en la estimación de $\\beta_1$.
                                                       - El histograma de $\\beta_1$ se desplaza hacia la derecha si $\\beta_2$ es positivo, y hacia la izquierda si $\\beta_2$ es negativo.
                                                       - El sesgo aumenta a medida que aumenta la correlación.
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
                                                create_title_with_hovercard(
                                                    "Ajuste de parámetros",
                                                    "Estos parámetros cambian la correlación entre la variable explicativa X₁ y la variable omitida U."
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Markdown(
                                                    "Correlación entre $X_1$ y $U$", mathjax=True
                                                ),
                                                dmc.Slider(
                                                    id="correlation",
                                                    min=-1,
                                                    max=1,
                                                    step=0.05,
                                                    marks=[{"value": x, "label": f"{x}"} for x in np.arange(-1, 1.01, 0.25)],
                                                    value=0,
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
                                                    "Gráfico de dispersión",
                                                    "Muestra la relación entre las variables X₁, U e Y en una muestra de datos."
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="scatter",
                                                    figure=create_scatter_matrix_plot(
                                                        *generate_sample(250, 10, 3, 5, [10, 10], 0)
                                                    )
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                
                                # Second row: Histogram for beta_1
                                dmc.GridCol(
                                    span=12,
                                    children=[
                                        dmc.Paper(
                                            p="md",
                                            withBorder=True,
                                            radius="md",
                                            style={"backgroundColor": "white"},
                                            children=[
                                                create_title_with_hovercard(
                                                    "Distribución de estimaciones de β₁",
                                                    "El histograma muestra la distribución de las estimaciones simuladas de β₁"
                                                ),
                                                dmc.Divider(my="sm"),
                                                dcc.Graph(
                                                    id="histogram_estimator_beta_1",
                                                    figure=create_histogram(
                                                        run_simulation(500, 250, 10, 3, 5, [10, 10], 0)[:, 1],
                                                        [0, 6],
                                                        3,
                                                        "Estimación de β₁"
                                                    )
                                                ),
                                                dmc.Space(h=10),
                                                dmc.Alert(
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
    Output("histogram_estimator_beta_1", "figure"),
    Output("scatter", "figure"),
    Output("beta_1_stats", "children"),
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

    # Muestra para el scatter
    X_1, U, Y = generate_sample(
        sample_size=sample_size,
        beta_0=beta_0,
        beta_1=beta_1,
        beta_2=beta_2,
        mean=[10, 10],
        correlation=correlation,
    )

    # Create figures
    fig_histogram_1 = create_histogram(
        data=np.array(betas_1_estimated), 
        range_x=[0, 6], 
        true_value=beta_1,
        x_label="Estimación de β₁"
    )
    fig_scatter = create_scatter_matrix_plot(X1=X_1, U=U, Y=Y)

    # Calcular estadísticas
    beta_1_stats = {
        "mean": np.mean(betas_1_estimated),
        "std": np.std(betas_1_estimated),
    }

    # Create table with centered text
    beta_1_table = create_stats_table(beta_1_stats, beta_1, "β₁")

    return fig_histogram_1, fig_scatter, beta_1_table

############ RUN SERVER #################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true', help='Run in server mode')
    args = parser.parse_args()
    
    # Get port from environment variable or use default
    port = int(os.environ.get("DASH_PORT", 8066))
    
    try:
        if args.server:
            print(f"Running on server mode on port {port}")
            app.run(debug=True, host="0.0.0.0", port=port, processes=6, threaded=False)
        else:
            print(f"Running on local mode on port {port}")
            app.run_server(debug=True, port=port, processes=6, threaded=False)
    except Exception as e:
        print(f"Application error: {e}") 