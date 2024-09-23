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


######## CONFIGURACIÓN DE ESTILO  ##############

dash._dash_renderer._set_react_version("18.2.0")


# Configuración de estilo
def get_external_stylesheets():
    _VOLT_BASE = "https://cdn.jsdelivr.net/gh/stevej2608/volt-bootstrap-5-dashboard@1.4.2/dist/css/"
    VOLT = _VOLT_BASE + "volt.min.css"
    return [
        "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/chartist/0.11.4/chartist.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.2.0/css/all.min.css",
        "https://cdn.jsdelivr.net/npm/notyf@3/notyf.min.css",
        VOLT,
    ]


style = {
    # "height": 100,
    "width": "100%",
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

####### FUNCIONES #############


# Create Menu
def create_menu(font_weight=600, *args: dict) -> dmc.Accordion:
    """
    Create a Dash Mantine Accordion menu with multiple sections.

    This function generates an accordion menu where each section consists of a title and content.
    The content is rendered using Markdown with LaTeX math support enabled.

    Parameters:
    -----------
    font_weight : int, optional
        Font weight for the section titles in the accordion (default is 600).
    *args : dict
        Variable number of dictionaries representing each section of the accordion.
        Each dictionary should have a single key-value pair:
        - key (str): The title of the section
        - value (str): The content of the section in Markdown format

    Returns:
    --------
    dmc.Accordion
        A Dash Mantine Accordion component with the specified sections and content.

    Examples:
    ---------
    >>> section_1 = {"Introduction": "This is the **introduction** to the menu."}
    >>> section_2 = {"Math Section": "Here is some math: $E=mc^2$."}
    >>> menu = create_menu(500, section_1, section_2)
    >>>
    >>> # The returned 'menu' can then be used as part of a Dash layout:
    >>> app.layout = dmc.Container(menu)

    Notes:
    ------
    - The function uses `dmc.Accordion` and `dcc.Markdown` components from the Dash Mantine library.
    - LaTeX math support is enabled in the Markdown content using `mathjax=True`.
    - The accordion is configured with `disableChevronRotation=True`.

    See Also:
    ---------
    dash_mantine_components.Accordion : For more details on the Accordion component.
    dash_core_components.Markdown : For more information on Markdown rendering in Dash.
    """
    children_accordion = []
    for dict_a in args:
        title_section = list(dict_a.keys())[0]
        content_section = list(dict_a.values())[0]
        children_accordion.append(
            dmc.AccordionItem(
                [
                    # Titulo
                    dmc.AccordionControl(dmc.Text(title_section, fw=font_weight)),
                    # Contenido
                    dmc.AccordionPanel(dcc.Markdown(content_section, mathjax=True)),
                ],
                value=title_section,
            ),
        )
    accordion = dmc.Accordion(disableChevronRotation=True, children=children_accordion)
    return accordion


# Función par crear gráfico
def create_graph(id, title):
    return dmc.Stack(
        [
            dmc.Text(title, fw=800),
            dcc.Graph(id=id, style={"height": "100%", "width": "100%"}),
        ],
        style={"height": "100%"},
        justify="flex-start",
        gap="xs",
    )


# Función para crear la tabla de estadísticas
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


# Actualizar gráfico y fig
def update_layout(fig):
    fig.update_layout(
        height=300,
        margin=dict(l=35, r=35, t=30, b=30),
        modebar_remove=[
            "select",
            "lasso",
            "zoomIn",
            "zoomOut",
            "autoScale",
            "hover",
            "toggleSpikelines",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
        ],
    )
    return fig


def paper(content, height=COMPONENT_HEIGHT):
    # fmt: off
    return dmc.Paper(
            children=[content],
            p="xl",
            shadow="xs",
            withBorder=True,
            style={
                "height": height,
                "borderRadius": "10px",
                # "backgroundColor": "#f8f9fa",
            },
        )
    # fmt: on


def calculate_stats(beta_0_list, beta_1_list):
    return {
        "mean_beta_0": np.mean(beta_0_list),
        "var_beta_0": np.var(beta_0_list),
        "mean_beta_1": np.mean(beta_1_list),
        "var_beta_1": np.var(beta_1_list),
    }


def main_structure(menu, structure: list[list]) -> dmc.Stack:
    """
    Create a Dash Mantine layout with a menu and a grid-based structure.

    This function generates a layout where a menu is placed at the top, followed by
    a grid structure with rows. Each row consists of exactly two columns, and each
    column contains one or more Dash Mantine components.

    Parameters:
    -----------
    menu : dmc.Component
        The component representing the menu to be displayed at the top of the layout.
    structure : list[list]
        A list of lists where each sublist represents a row containing exactly two
        columns. Each column is a list of one or more Dash Mantine components to
        display within that column.

    Returns:
    --------
    dmc.Stack
        A Dash Mantine Stack component that contains the menu and a grid-based layout
        with the specified rows and columns.

    Examples:
    ---------
    >>> structure = [
    >>>    [[component1, component2], [component3]],  # First row with two columns
    >>>    [[component4], [component5, component6]]   # Second row with two columns
    >>> ]
    >>> layout = main_structure(menu_component, structure)
    >>>
    >>> # The returned 'layout' can then be used as part of a Dash layout:
    >>> app.layout = dmc.Container(layout)

    Notes:
    ------
    - Each row must contain exactly two columns, and each column can contain
      one or more components.
    - The function uses `dmc.Grid` and `dmc.GridCol` from the Dash Mantine library.
    - A `ValueError` is raised if any row does not contain exactly two columns.

    See Also:
    ---------
    dash_mantine_components.Grid : For more details on the Grid component.
    dash_mantine_components.GridCol : For more information on how columns are handled.
    """

    content_grids = []

    # Verificar que cada fila tenga exactamente dos elementos y que esos elementos sean listas
    for fila in structure:
        if len(fila) != 2:
            raise ValueError("Cada fila debe contener exactamente dos elementos.")
        if not all(isinstance(element, list) for element in fila):
            raise ValueError("Cada elemento dentro de una fila debe ser una lista.")

        # Crear las columnas (GridCol) de la fila
        col_grids = [dmc.GridCol(children=element, span=6) for element in fila]

        # Crear el grid de la fila
        grid = dmc.Grid(
            children=col_grids,
            gutter="md",
            align="stretch",
        )

        content_grids.append(grid)

    # Crear la estructura principal con el menú y las filas
    main_structure = dmc.Stack(
        children=[
            menu,  # Menú en la parte superior
            *content_grids,  # Filas con sus respectivas columnas
        ],
        gap="lg",
    )
    pprint.pp(main_structure)

    return main_structure


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
Esta simulación demuestra la insesgadez y los determinantes de la varianza de los estimadores OLS $\\beta_0$ y $\\beta_1$ en el modelo:
$$
y = \\beta_0 + \\beta_1 x + \\epsilon
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
       - Y: Calculada usando la ecuación del modelo
         * $Y = 10 + 3X + \\epsilon (\\beta_0 = 10, \\beta_1 = 3)$

3. **Estimación:**
       - Para cada muestra, se estiman $\\beta_0$ y $\\beta_1$ usando OLS.
       - Se recopilan todas las estimaciones para crear los histogramas.

Ajusta los sliders para observar cómo la variabilidad en $X$ y $\\epsilon$ afecta las estimaciones de $\\beta_0$ y $\\beta_1$.
"""

unbiasedness_content = """
Los histogramas deben centrarse cerca de los valores verdaderos $(\\beta_0 = 10, \\beta_1 = 3)$, demostrando insesgadez.
"""

variance_content = """
1. **Varianza del término de error:** 
       - Aumentarla incrementa la dispersión de Y alrededor de la línea de regresión.
       - Resulta en histogramas más anchos para $\\beta_0$ y $\\beta_1$.

2. **Varianza de $X$:**
       - Aumentarla dispersa los valores de $X$.
       - Resulta en un histograma más estrecho para $\\beta_1$ y para $\\beta_0$.
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
scatter_content = create_graph("scatter", "Gráfico de dispersión")

# histogram of beta 0
histogram_beta_0_content = create_graph(
    "histogram_estimator_beta_0", "Histograma de estimación β₀"
)

# histogram of beta 1
histogram_beta_1_content = create_graph(
    "histogram_estimator_beta_1", "Histograma de estimación β₁"
)


######### 3) SLIDERS ##############3

sliders_stack = dmc.Stack(
    [
        dmc.Text("Ajuste de parámetros", fw=800),
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
    fig_histogram_0 = create_histogram(estimates[:, 0], [0, 20], beta_0, "$\\beta_0$")
    fig_histogram_1 = create_histogram(estimates[:, 1], [0, 6], beta_1, "$\\beta_1$")

    # Crear gráfico de dispersión
    X, Y = generate_sample(sample_size, beta_0, beta_1, x_mean, x_std, error_std)
    fig_scatter = create_scatter_plot(X, Y)

    # Actualizar el diseño de los gráficos
    for fig in [fig_histogram_0, fig_histogram_1, fig_scatter]:
        update_layout(fig)

    # Calcular estadísticas
    beta_0_stats = {"mean": np.mean(estimates[:, 0]), "var": np.var(estimates[:, 0])}
    beta_1_stats = {"mean": np.mean(estimates[:, 1]), "var": np.var(estimates[:, 1])}

    # Crear tablas de estadísticas
    beta_0_table = create_stats_table(beta_0_stats, beta_0, "β₀")
    beta_1_table = create_stats_table(beta_1_stats, beta_1, "β₁")

    return fig_histogram_0, fig_histogram_1, fig_scatter, beta_0_table, beta_1_table


app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    children=[
        dmc.Container(
            fluid=True,
            # size="lg",
            style=style,
            children=[
                dmc.Title(
                    "Simulación OLS: Insesgadez y Varianza de Estimadores", order=2
                ),
                dmc.Space(h=20),
                resumen_content,
            ],
        )
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT"))
    app.run_server(debug=False, port=port)
