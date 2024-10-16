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
    return fig


# Crear scatter
def create_scatter_plot(X, Y):
    return px.scatter(
        x=X,
        y=Y,
        labels={"x": "X", "y": "Y"},
        range_x=[-150, 150],
        range_y=[-410, 410],
    )


def calculate_stats(beta_0_list, beta_1_list):
    return {
        "mean_beta_0": np.mean(beta_0_list),
        "var_beta_0": np.var(beta_0_list),
        "mean_beta_1": np.mean(beta_1_list),
        "var_beta_1": np.var(beta_1_list),
    }


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
Esta simulación demuestra la insesgadez y los determinantes de la varianza de los estimadores OLS de $\\beta_0$ y $\\beta_1$ en el modelo poblacional:
$$
Y_i = \\beta_0 + \\beta_1 X_i + \\varepsilon_i
$$

En particular, supondremos que $\\beta_0 = 10$ y $\\beta_1 = 3$.

$$
Y_i = 10 + 3 X_i + \\varepsilon_i
$$

"""

simulation_content = """
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
"""

unbiasedness_content = """
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

"""

variance_content = """
La varianza de los estimadores OLS ( $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$ está determinada por dos factores:

1. **Varianza del término de error:** 
       - Un aumento en el error del modelo se traduce en mayor imprecisión de la estimación del modelo.
       - Esto se refleja en histogramas más anchos para $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$.
       - Mayor varianza estimada de las estimaciones realizadas (Ver valor en la tabla)).

2. **Varianza de $X$:**
       - Ceteris paribus, una  mayor dispersión en los valores de la variable explicativa $X$ reduce la imprecisión de la estimación $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$ .
       - Esto se refleja en histogramas más estrecho para $\\hat{\\beta_0}$ y $\\hat{\\beta_1}$.
       - Menor varianza estimada de las estimaciones realizadas (Ver valor en la tabla)).

"""

# Creamos el componente Accordion (las cajitas) con las secciones

markdown_description = create_menu(
    600,
    {"Introducción": intro_content},
    {"Demostración de Insesgadez": unbiasedness_content},
    {"Determinantes de la Varianza de la Estimación": variance_content},
    {"Metodología": simulation_content},
)


############ 2) GRÁFICOS ##############3

# scatter
scatter_content = create_graph(
    id="scatter",
    title="Gráfico de dispersión",
    information="Incluye una muestra de datos obtenidos de la población. Cada punto es un dato de la muestra: un par de X e Y",
)

# histogram of beta 0
histogram_beta_0_content = create_graph(
    "histogram_estimator_beta_0",
    "Distribución de estimaciones de β₀",
    "El histograma muestra la distribución de las estimaciones simuladas de β₀",
)

# histogram of beta 1
histogram_beta_1_content = create_graph(
    "histogram_estimator_beta_1",
    "Distribución de estimaciones de β₁",
    "El histograma muestra la distribución de las estimaciones simuladas de β₁",
)


######### 3) SLIDERS ##############3

sliders_stack = dmc.Stack(
    [
        card_title_w_information_hovercard(
            title="Ajuste de parámetros",
            information="Estos parámetros cambian los supuestos del modelo poblacional a ser estimado.",
        ),
        # dmc.Text("Desviación estándar del error (Sigma ε)", fw=600),
        dcc.Markdown(
            "*Desviación estándar del error* $(\\sigma_{\\varepsilon})$", mathjax=True
        ),
        dmc.Slider(
            id="std_error",
            updatemode="drag",
            min=1,
            max=50,
            marks=[{"value": 1, "label": "1"}]
            + [{"value": x, "label": f"{x}"} for x in range(10, 51, 10)],
            value=10,
            style={"width": "80%"},
        ),
        dmc.Space(h=2),
        # dmc.Text("Desviación estándar de X (Sigma X)", fw=600),
        dcc.Markdown("Desviación estándar de $X$ $(\\sigma_{X})$", mathjax=True),
        dmc.Slider(
            id="std_error_x",
            updatemode="drag",
            min=1,
            max=50,
            marks=[{"value": 1, "label": "1"}]
            + [{"value": x, "label": f"{x}"} for x in range(10, 51, 10)],
            value=10,
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
    ],
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

    # Actualizar el diseño de los gráficos
    for fig in [fig_histogram_0, fig_histogram_1, fig_scatter]:
        update_layout(fig)

    # Calcular estadísticas
    beta_0_stats = {"mean": np.mean(estimates[:, 0]), "std": np.std(estimates[:, 0])}
    beta_1_stats = {"mean": np.mean(estimates[:, 1]), "std": np.std(estimates[:, 1])}

    # Crear tablas de estadísticas
    beta_0_table = create_stats_table(beta_0_stats, beta_0, r"$$\hat{\beta_0}$$")
    beta_1_table = create_stats_table(beta_1_stats, beta_1, r"$$\hat{\beta_1}$$")

    return fig_histogram_0, fig_histogram_1, fig_scatter, beta_0_table, beta_1_table


############ LAYOUT ##################
app.layout = build_layout(
    title="Simulación: Insesgadez y Varianza de Estimadores OLS",
    content=resumen_content,
)

############ RUN SERVER #################
if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT"))
    app.run_server(host="0.0.0.0", debug=False, port=port)
    # app.run_server(debug=False, port=8070)

