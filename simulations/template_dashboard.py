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
""" def create_graph(id, title):
    return dmc.Stack(
        [
            dmc.Text(title, fw=800),
            dcc.Graph(id=id, style={"height": "100%", "width": "100%"}),
        ],
        style={"height": "100%"},
        justify="flex-start",
        gap="xs",
    ) """


def create_graph(id, title, information=""):
    return dmc.Stack(
        [
            dmc.Group(
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
                # spacing="xs",
            ),
            dcc.Graph(id=id, style={"height": "100%", "width": "100%"}),
        ],
        style={"height": "100%"},
        justify="flex-start",
        gap="xs",
    )


def title_with_hovercard(title:str, information:str=""):
    """
    Creates a styled title with a hoverable icon that displays additional information when hovered.

    This function generates a group consisting of:
    - A title displayed in bold font.
    - A hoverable question mark icon next to the title. When hovered over, a card with additional information is shown.

    Parameters:
    -----------
    title : str
        The main title text to display with bold styling.
    information : str, optional
        The content to display in the hovercard when the question mark icon is hovered. 
        Supports Markdown and math formatting. Defaults to an empty string.

    Returns:
    --------
    dmc.Group
        A group component with the title and hovercard, structured using Mantine's Group and HoverCard components.
    """
    return dmc.Group(
                [
                 # Esto es el titulo
                    dmc.Text(title, fw=800), 
                 # Esto es el hovercard (signo de pregunta)  
                    dmc.HoverCard(
                        withArrow=True,
                        width=400,
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
                                html.Div(
                                    dcc.Markdown(
                                        children=information,
                                        mathjax=True,
                                        # size="sm",
                                        style={
                                            # "white-space": "pre-line",
                                            # "overflow-x": "scroll",
                                            "font-weight": "normal",
                                            "font-size": "14px",
                                            "text-align": "justify",
                                            "width": "100%"
                                        },
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
                # spacing="xs",
            )
        
        
def paper(content, height=COMPONENT_HEIGHT):
    # fmt: off
    print(*content)
    return dmc.Paper(
            children=[x for x in content],
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

    return main_structure


def build_layout(title: str, content):
    return dmc.MantineProvider(
        theme={"colorScheme": "light"},
        children=[
            dmc.Container(
                fluid=True,
                # size="lg",
                style=style,
                children=[
                    dmc.Title(title, order=2),
                    dmc.Space(h=20),
                    content,
                ],
            )
        ],
    )


def create_stats_table(
    stats={"mean": 0, "std": 0},
    true_value=0,
    parameter_name="unknown",
    custom=False,
    custom_title="",
    custom_dict_varnames=dict(),
):
    default = {
        # "caption": "Estadísticas de los Estimadores",
        "head": [
            "Estadísticas de los valores simulados",
            dcc.Markdown(
                parameter_name,
                mathjax=True,
            ),
        ],
        "body": [
            ["Valor verdadero (poblacional)", f"{true_value:.4f}"],
            ["Media", f"{stats['mean']:.4f}"],
            ["Desvío Estándar", f"{stats['std']:.4f}"],
        ],
    }
    if custom == True:
        names_var = [[name, value] for name, value in custom_dict_varnames.items()]
        dict_data = {
            # "caption": "Estadísticas de los Estimadores",
            "head": [
                custom_title,
                dcc.Markdown(
                    parameter_name,
                    mathjax=True,
                ),
            ],
            "body": names_var,
        }

    else:
        dict_data = default

    return dmc.Table(
        striped=True,
        highlightOnHover=True,
        # withTableBorder=True,
        withColumnBorders=True,
        data=dict_data,
    )


def card_title_w_information_hovercard(title: str, information: str):

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
        # spacing="xs",
    )
