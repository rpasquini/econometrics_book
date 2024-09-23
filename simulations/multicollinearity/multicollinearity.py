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

######## CONFIGURACIÓN DE ESTILO  ##############
dash._dash_renderer._set_react_version("18.2.0")

_VOLT_BASE = (
    "https://cdn.jsdelivr.net/gh/stevej2608/volt-bootstrap-5-dashboard@1.4.2/dist/css/"
)

VOLT = _VOLT_BASE + "volt.min.css"

VOLT_BOOTSTRAP = _VOLT_BASE + "volt.bootstrap.min.css"

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/chartist/0.11.4/chartist.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.2.0/css/all.min.css",
    "https://cdn.jsdelivr.net/npm/notyf@3/notyf.min.css",
    VOLT,
]

style = {
    # "height": 100,
    "width": "80%",
    # "border": f"1px solid {dmc.DEFAULT_THEME['colors']['indigo'][4]}",
    "marginTop": 20,
    "marginBottom": 20,
}

# Definir un estilo común para los títulos
title_style = {
    "marginBottom": 0,
    "marginLeft": 40,
    "marginTop": 20,
    "textAlign": "left",
}


COMPONENT_HEIGHT = "400px"

############### INSTANCIA DE LA APP ###############3
app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
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
markdown_description = dmc.Accordion(
    disableChevronRotation=True,
    children=[
        dmc.AccordionItem(
            [
                # Titulo
                dmc.AccordionControl(dmc.Text("Introducción", fw=600)),
                # Contenido
                dmc.AccordionPanel(dcc.Markdown(intro_content, mathjax=True)),
            ],
            value="intro",
        ),
        dmc.AccordionItem(
            [
                # Titulo
                dmc.AccordionControl(dmc.Text("Metodología", fw=600)),
                # Contenido
                dmc.AccordionPanel(dcc.Markdown(simulation_content, mathjax=True)),
            ],
            value="simulation",
        ),
        dmc.AccordionItem(
            [
                # Titulo
                dmc.AccordionControl(dmc.Text("Demostración de Insesgadez", fw=600)),
                # Contenido
                dmc.AccordionPanel(dcc.Markdown(unbiasedness_content, mathjax=True)),
            ],
            value="unbiasedness",
        ),
        dmc.AccordionItem(
            [
                # Titulo
                dmc.AccordionControl(dmc.Text("Determinantes de la Varianza", fw=600)),
                # Contenido
                dmc.AccordionPanel(dcc.Markdown(variance_content, mathjax=True)),
            ],
            value="variance",
        ),
    ],
)


############ 2) GRÁFICOS ##############3

# scatter
scatter_content = dmc.Stack(
    [
        dmc.Text("Gráficos de dispersión", fw=800),
        dcc.Graph(id="scatter", style={"height": "100%", "width": "100%"}),
    ],
    # align="center",
    # gap="xs",
    # spacing=0,
    style={"height": "100%"},
    justify="flex-start",
    gap=0,
)

# histogram of beta 0
histogram_beta_0_content = dmc.Stack(
    [
        dmc.Text("Histograma de β₀", fw=800),
        dcc.Graph(
            id="histogram_estimator_beta_0", style={"height": "100%", "width": "100%"}
        ),
    ],
    style={"height": "100%"},
    justify="flex-start",
    gap="xs",
)

# histogram of beta 1
histogram_beta_1_content = dmc.Stack(
    [
        dmc.Text("Histograma de β₁", fw=800),
        dcc.Graph(
            id="histogram_estimator_beta_1", style={"height": "100%", "width": "100%"}
        ),
    ],
    style={"height": "100%"},
    justify="flex-start",
    gap="xs",
)

# histogram of beta 2
histogram_beta_2_content = dmc.Stack(
    [
        dmc.Text("Histograma de β₂", fw=800),
        dcc.Graph(
            id="histogram_estimator_beta_2", style={"height": "100%", "width": "100%"}
        ),
    ],
    style={"height": "100%"},
    justify="flex-start",
    gap="xs",
)

######### 3) SLIDERS ##############3

sliders_stack = dmc.Stack(
    [
        dmc.Text("Ajuste de parámetros", fw=800),
        dmc.Text("Correlación entre X₁ y X₂", fw=600),
        dmc.Slider(
            id="correlation",
            updatemode="drag",
            min=-1,
            max=1,
            step=0.1,
            marks=[{"value": x, "label": f"{x:.1f}"} for x in np.arange(-1, 1.1, 0.2)],
            value=0,
            style={"width": "80%"},
        ),
    ],
    justify="center",
    gap="xl",
)


resumen_content = dmc.Stack(
    [
        markdown_description,
        dmc.Grid(
            children=[
                dmc.GridCol(
                    dmc.Paper(
                        children=[sliders_stack],
                        p="xl",
                        shadow="xs",
                        withBorder=True,
                        style={
                            "height": COMPONENT_HEIGHT,
                            "borderRadius": "10px",
                            # "backgroundColor": "#f8f9fa",
                        },
                    ),
                    span=6,
                ),
                dmc.GridCol(
                    dmc.Paper(
                        children=[scatter_content],
                        p="xl",
                        shadow="xs",
                        withBorder=True,
                        style={
                            "height": COMPONENT_HEIGHT,
                            "borderRadius": "10px",
                        },
                    ),
                    span=6,
                ),
            ],
            gutter="xl",
            align="stretch",
        ),
        # dmc.Space(h=30),
        dmc.Grid(
            children=[
                dmc.GridCol(
                    children=[
                        dmc.Paper(
                            children=[histogram_beta_0_content],
                            p="xl",
                            shadow="xs",
                            withBorder=True,
                            style={
                                "height": COMPONENT_HEIGHT,
                                "borderRadius": "10px",
                            },
                        ),
                        dmc.Space(h=10),
                        dmc.Paper(
                            children=[html.Div(id="beta_0_stats")],
                            p="sm",
                            shadow="xs",
                            withBorder=True,
                            style={"borderRadius": "10px"},
                        ),
                    ],
                    span=4,
                ),
                dmc.GridCol(
                    children=[
                        dmc.Paper(
                            children=[histogram_beta_1_content],
                            className="isolated-paper",
                            p="xl",
                            shadow="xs",
                            withBorder=True,
                            style={
                                "height": COMPONENT_HEIGHT,
                                "borderRadius": "10px",
                            },
                        ),
                        dmc.Space(h=10),
                        dmc.Paper(
                            children=[html.Div(id="beta_1_stats")],
                            className="isolated-paper",
                            p="sm",
                            shadow="xs",
                            withBorder=True,
                            style={"borderRadius": "10px"},
                        ),
                    ],
                    span=4,
                ),
                dmc.GridCol(
                    children=[
                        dmc.Paper(
                            children=[histogram_beta_2_content],
                            className="isolated-paper",
                            p="xl",
                            shadow="xs",
                            withBorder=True,
                            style={
                                "height": COMPONENT_HEIGHT,
                                "borderRadius": "10px",
                            },
                        ),
                        dmc.Space(h=10),
                        dmc.Paper(
                            children=[html.Div(id="beta_2_stats")],
                            className="isolated-paper",
                            p="sm",
                            shadow="xs",
                            withBorder=True,
                            style={"borderRadius": "10px"},
                        ),
                    ],
                    span=4,
                ),
            ],
            gutter="xl",
            align="stretch",
        ),
    ],
    gap="lg",
)


def calculate_stats(beta_0_list, beta_1_list, beta_2_list):
    return {
        "mean_beta_0": np.mean(beta_0_list),
        "var_beta_0": np.var(beta_0_list),
        "mean_beta_1": np.mean(beta_1_list),
        "var_beta_1": np.var(beta_1_list),
        "mean_beta_2": np.mean(beta_2_list),
        "var_beta_2": np.var(beta_2_list),
    }


def create_stats_table(stats, true_value, parameter_name):
    return dmc.Table(
        striped=True,
        highlightOnHover=True,
        # withTableBorder=True,
        withColumnBorders=True,
        data={
            # "caption": "Estadísticas de los Estimadores",
            "head": ["Estadística", parameter_name],
            "body": [
                ["Media", f"{stats['mean']:.4f}"],
                ["Varianza", f"{stats['var']:.4f}"],
                ["Valor verdadero", f"{true_value:.4f}"],
            ],
        },
    )


@app.callback(
    Output("histogram_estimator_beta_0", "figure"),
    Output("histogram_estimator_beta_1", "figure"),
    Output("histogram_estimator_beta_2", "figure"),
    Output("scatter", "figure"),
    Output("beta_0_stats", "children"),
    Output("beta_1_stats", "children"),
    Output("beta_2_stats", "children"),
    Input("correlation", "value"),
)
def update_histogram(correlation):
    sample_size = 250
    amount_samples = 500
    beta_0 = 10
    beta_1 = 3
    beta_2 = 5

    beta_1_list = []
    beta_0_list = []
    beta_2_list = []

    for n_sample in range(amount_samples):
        # Generate correlated variables
        cov_matrix = np.array(
            [[10**2, correlation * 10**2], [correlation * 10**2, 10**2]]
        )
        X_1_2 = np.random.multivariate_normal(
            mean=[10, 10], cov=cov_matrix, size=sample_size
        )
        X_1 = X_1_2[:, 0]
        X_2 = X_1_2[:, 1]
        epsilon = np.random.normal(loc=0, scale=10, size=sample_size)
        Y = beta_0 + beta_1 * X_1 + beta_2 * X_2 + epsilon

        X_with_constant = sm.add_constant(np.column_stack((X_1, X_2)))
        model = sm.OLS(exog=X_with_constant, endog=Y)
        results = model.fit()

        coefficients = results.params

        beta_0_estimated = coefficients[0]
        beta_0_list.append(beta_0_estimated)

        beta_1_estimated = coefficients[1]
        beta_1_list.append(beta_1_estimated)

        beta_2_estimated = coefficients[2]
        beta_2_list.append(beta_2_estimated)

    fig_histogram_0 = px.histogram(
        x=np.array(beta_0_list),
        color_discrete_sequence=["indianred"],
        range_x=[0, 20],
        # title="Histograma de β₀",
    )
    fig_histogram_0.add_vline(
        x=beta_0,
        line_dash="dash",
        line_color="black",
        annotation_text="Valor verdadero",
    )

    fig_histogram_1 = px.histogram(
        x=np.array(beta_1_list),
        color_discrete_sequence=["indianred"],
        range_x=[0, 6],
        # title="Histograma de β₁",
    )
    fig_histogram_1.add_vline(
        x=beta_1,
        line_dash="dash",
        line_color="black",
        annotation_text="Valor verdadero",
    )

    fig_histogram_2 = px.histogram(
        x=np.array(beta_2_list),
        color_discrete_sequence=["indianred"],
        range_x=[0, 10],
        # title="Histograma de β₂",
    )
    fig_histogram_2.add_vline(
        x=beta_2,
        line_dash="dash",
        line_color="black",
        annotation_text="Valor verdadero",
    )

    fig_scatter = px.scatter_matrix(
        pd.DataFrame({"X1": X_1, "X2": X_2, "Y": Y}),
        dimensions=["X1", "X2", "Y"],
        color_discrete_sequence=["indianred"],
    )

    for fig in [fig_histogram_0, fig_histogram_1, fig_histogram_2, fig_scatter]:
        fig.update_layout(
            height=300,
            margin=dict(l=35, r=35, t=30, b=30),
            modebar_remove=[
                # "zoom",
                # "pan",
                "select",
                "lasso",
                "zoomIn",
                "zoomOut",
                "autoScale",
                # "resetScale",
                "hover",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
        )

    beta_0_stats = {"mean": np.mean(beta_0_list), "var": np.var(beta_0_list)}
    beta_1_stats = {"mean": np.mean(beta_1_list), "var": np.var(beta_1_list)}
    beta_2_stats = {"mean": np.mean(beta_2_list), "var": np.var(beta_2_list)}

    beta_0_table = create_stats_table(beta_0_stats, beta_0, "β₀")
    beta_1_table = create_stats_table(beta_1_stats, beta_1, "β₁")
    beta_2_table = create_stats_table(beta_2_stats, beta_2, "β₂")

    return (
        fig_histogram_0,
        fig_histogram_1,
        fig_histogram_2,
        fig_scatter,
        beta_0_table,
        beta_1_table,
        beta_2_table,
    )


app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    children=[
        dmc.Container(
            fluid=True,
            # size="lg",
            style=style,
            children=[
                dmc.Title("Simulación OLS: Multicolinealidad y sus Efectos", order=2),
                dmc.Space(h=20),
                resumen_content,
            ],
        )
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT"))
    app.run_server(debug=False, port=port)
    # app.run_server(debug=False, port=8070)
