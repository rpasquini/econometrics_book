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
    return fig


# Crear scatter
def create_scatter_matrix_plot(X1, U, Y):
    return px.scatter_matrix(
        data_frame=pd.DataFrame({"X1": X1, "U": U, "Y": Y}),
        dimensions=["X1", "U", "Y"],
        color_discrete_sequence=["indianred"],
    )


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
Esta simulación demuestra el efecto del sesgo por variable omitida en el estimador OLS $\\beta_1$ en el modelo:
$$
y = \\beta_0 + \\beta_1 x_1 + \\beta_2 u + \\epsilon
$$
$$
\\beta_0 = 10, \\beta_1 = 3, \\beta_2 = 5
$$
donde $x_1$ es la variable explicativa incluida y $u$ es la variable omitida.
"""

simulation_content = """
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
"""

unbiasedness_content = """
El histograma debe centrarse cerca del valor verdadero $(\\beta_1 = 3)$, demostrando insesgadez.
"""

variance_content = """
1. **Correlación entre $X_1$ y $U$:**
       - Aumentar la correlación entre $X_1$ y $U$ crea un sesgo en la estimación de $\\beta_1$.
       - El histograma de $\\beta_1$ se desplaza hacia la derecha si $\\beta_2$ es positivo, y hacia la izquierda si $\\beta_2$ es negativo.
       - El sesgo aumenta a medida que aumenta la correlación.
"""

# Creamos el componente Accordion (las cajitas) con las secciones
markdown_description = create_menu(
    600,
    {"Introducción": intro_content},
    {"Metodología": simulation_content},
    {"Demostración de Insesgadez": unbiasedness_content},
    {"Determinantes del Sesgo": variance_content},
)


############ 2) GRÁFICOS ##############3

# scatter
scatter_content = create_graph(id="scatter", title="Gráfico de dispersión")

# histogram of beta 1
histogram_beta_1_content = create_graph(
    "histogram_estimator_beta_1", "Histograma de estimación β₁"
)

######### 3) SLIDERS ##############3

sliders_stack = dmc.Stack(
    [
        dmc.Text("Ajuste de parámetros", fw=800),
        dcc.Markdown("Correlación entre $X_1$ y $U$", mathjax=True),
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

########## ESTRUCTURA ##################3
resumen_content = main_structure(
    menu=markdown_description,
    structure=[
        # Fila 1
        [[paper(sliders_stack)], [paper(scatter_content)]],
        # Fila 2
        [
            [
                paper(histogram_beta_1_content),
                dmc.Space(h=10),
                paper(html.Div(id="beta_1_stats"), ""),
            ],
            [],
        ],
    ],
)


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

    # fmt: off
    fig_histogram_1=create_histogram(data=np.array(betas_1_estimated), range_x=[0,6], true_value=beta_1 )
    fig_scatter = create_scatter_matrix_plot(X1=X_1, U=U, Y=Y)
    # fmt: on

    for fig in [fig_histogram_1, fig_scatter]:
        update_layout(fig)

    beta_1_stats = {
        "mean": np.mean(betas_1_estimated),
        "std": np.std(betas_1_estimated),
    }

    beta_1_table = create_stats_table(beta_1_stats, beta_1, "β₁")

    return fig_histogram_1, fig_scatter, beta_1_table


############ LAYOUT ##################
app.layout = build_layout(
    title="Simulación OLS: Sesgo por Variable Omitida",
    content=resumen_content,
)


############ RUN SERVER #################
if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT"))
    app.run_server(host="0.0.0.0", debug=False, port=port)
    # app.run_server(debug=False, port=8080)

# print("hola soy octi")
