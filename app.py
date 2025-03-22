# ================= Import les packages ================= #

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dash_table
from calendar import month_name, month_abbr
import plotly.graph_objects as go


# ================= Initialiser l'application ================= #

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Ajout de cette ligne pour Render



# ================= Import les données ================= #

df = pd.read_csv("transactions.csv", encoding="utf-8")
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
df["Month"] = df["Transaction_Date"].dt.month
df["Total_price"] = df["Quantity"] * df["Avg_Price"] * (1 - (df["Discount_pct"] / 100))
df["Transaction_Date_Display"] = df["Transaction_Date"].dt.date


# ================= Fonctions d'analyse ================= #

def calculer_chiffre_affaire(data):
    return data["Total_price"].sum()


def frequence_meilleure_vente(data, top=10, ascending=False):
    resultat = (
        pd.crosstab(
            [data["Gender"], data["Product_Category"]],
            "Total vente",
            values=data["Total_price"],
            aggfunc=lambda x: len(x),
            rownames=["Sexe", "Categorie du produit"],
            colnames=[""],
        )
        .reset_index()
        .groupby(["Sexe"], as_index=False, group_keys=True)
        .apply(
            lambda x: x.sort_values("Total vente", ascending=ascending).iloc[:top, :]
        )
        .reset_index(drop=True)
        .set_index(["Sexe", "Categorie du produit"])
    )

    return resultat


def indicateur_du_mois(data, current_month=12, freq=True, abbr=False):
    previous_month = current_month - 1 if current_month > 1 else 12
    if freq:
        resultat = data["Month"][
            (data["Month"] == current_month) | (data["Month"] == previous_month)
        ].value_counts()
        # sort by index
        resultat = resultat.sort_index()
        resultat.index = [
            (month_abbr[i] if abbr else month_name[i]) for i in resultat.index
        ]
        return resultat
    else:
        resultat = (
            data[(data["Month"] == current_month) | (data["Month"] == previous_month)]
            .groupby("Month")
            .apply(calculer_chiffre_affaire)
        )
        resultat.index = [
            (month_abbr[i] if abbr else month_name[i]) for i in resultat.index
        ]
        return resultat


# Top 10 des ventes
def barplot_top_10_ventes(data):
    df_plot = frequence_meilleure_vente(data, ascending=True)
    graph = px.bar(
        df_plot,
        x="Total vente",
        y=df_plot.index.get_level_values(1),
        color=df_plot.index.get_level_values(0),
        barmode="group",
        title="Frequence des 10 meilleures ventes",
        labels={"x": "Fréquence", "y": "Categorie du produit", "color": "Sexe"},
        color_discrete_map={"F": "#636EFA", "M": "#EF553B"},
    ).update_layout(margin=dict(t=60), height=600)
    return graph


# Evolution chiffre d'affaire
def plot_evolution_chiffre_affaire(data):
    df_plot = data.groupby(pd.Grouper(key="Transaction_Date", freq="W")).apply(
        calculer_chiffre_affaire
    )[:-1]
    chiffre_evolution = px.line(
        x=df_plot.index,
        y=df_plot,
        title="Evolution du chiffre d'affaire par semaine",
        labels={"x": "Semaine", "y": "Chiffre d'affaire"},
    ).update_layout(
        margin=dict(t=60, b=0), height=300  # Ajustement de la hauteur
    )
    return chiffre_evolution


# Chiffre d'affaire du mois
def plot_chiffre_affaire_mois(data):
    df_plot = indicateur_du_mois(data, freq=False)
    indicateur = go.Figure(
        go.Indicator(
            mode="number+delta",
            value=df_plot[1],
            delta={
                "reference": df_plot[0],
                "position": "bottom",
                "increasing": {"color": "green"},
                "decreasing": {"color": "red"},
            },
            domain={"row": 0, "column": 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout(
        margin=dict(l=0, r=0, t=25, b=0), font=dict(size=16)
    )  # Réduction de la taille de la police
    return indicateur


# Ventes du mois
def plot_vente_mois(data, abbr=False):
    df_plot = indicateur_du_mois(data, freq=True, abbr=abbr)
    indicateur = go.Figure(
        go.Indicator(
            mode="number+delta",
            value=df_plot[1],
            delta={
                "reference": df_plot[0],
                "position": "bottom",
                "increasing": {"color": "green"},
                "decreasing": {"color": "red"},
            },
            domain={"row": 0, "column": 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout(margin=dict(l=0, r=0, t=25, b=0), font=dict(size=16))
    return indicateur


# Tableau
sales_table = dash_table.DataTable(
    id="sales-table",
    data=df.sort_values(by="Transaction_Date", ascending=False)
    .head(100)
    .to_dict("records"),
    columns=[
        {"name": "Date", "id": "Transaction_Date_Display"},
        {"name": "Gender", "id": "Gender"},
        {"name": "Location", "id": "Location"},
        {"name": "Product Category", "id": "Product_Category"},
        {"name": "Quantity", "id": "Quantity"},
        {"name": "Avg Price", "id": "Avg_Price"},
        {"name": "Discount Pct", "id": "Discount_pct"},
    ],
    page_size=10,
    filter_action="native",
    sort_action="native",
    sort_by=[{"column_id": "Transaction_Date", "direction": "desc"}],
    style_table={"height": "300px", "overflowY": "auto"},
    style_cell={
        "textAlign": "right",
        "fontSize": 13,
        "padding": "2px",
        "margin": "0px",
        "lineHeight": "1",
        "whiteSpace": "normal",
        "height": "25px",
        "fontFamily": "'Fira Code', monospace",  # Font with slashed zero
        "fontWeight": 500,  # Adjusted for thicker font style
    },
    style_header={
        "backgroundColor": "rgb(230, 230, 230)",
        "fontWeight": 600,  # Adjusted for thicker font style
        "fontSize": 13,
        "padding": "2px",
        "margin": "0px",
        "lineHeight": "1",
        "fontFamily": "'Fira Code', monospace",  # Font with slashed zero
    },
    style_data={
        "minWidth": "auto",
        "maxWidth": "auto",
        "whiteSpace": "normal",
    },
    style_header_conditional=[
        {
            "if": {"column_id": col["id"]},
            "minWidth": f"{len(col['name']) + 2}ch",
            "maxWidth": f"{len(col['name']) + 2}ch",
        }
        for col in [
            {"name": "Date", "id": "Transaction_Date_Display"},
            {"name": "Gender", "id": "Gender"},
            {"name": "Location", "id": "Location"},
            {"name": "Product Category", "id": "Product_Category"},
            {"name": "Quantity", "id": "Quantity"},
            {"name": "Avg Price", "id": "Avg_Price"},
            {"name": "Discount Pct", "id": "Discount_pct"},
        ]
    ],
)


# ================= Layout de l'Application ================= #

app.layout = dbc.Container(
    fluid=True,
    style={"height": "100vh", "overflow": "hidden"},
    children=[
        dbc.Row(
            [
                dbc.Col(
                    html.H3("ECAP Store", className="text-left text-dark"),
                    md=6,
                    style={"height": "7vh"},
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="zone-dropdown",
                        options=[{"label": "Toutes les zones", "value": "all"}]
                        + [
                            {"label": loc, "value": loc}
                            for loc in df["Location"].unique()
                        ],
                        value=None,
                        placeholder="Choisissez des zones",
                        style={"width": "80%"},
                    ),
                    md=6,
                    style={"height": "7vh"},
                ),
            ],
            className="p-2",
            style={"backgroundColor": "#ADD8E6"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="chiffre-affaire-mois",
                                        figure=plot_chiffre_affaire_mois(df),
                                        style={"height": "100%"},
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="vente-mois",
                                        figure=plot_vente_mois(df),
                                        style={"height": "100%"},
                                    ),
                                    md=6,
                                ),
                            ],
                            style={
                                "height": "120px",
                                "marginTop": "20px",
                                "marginBottom": "20px",
                            },
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="top-sales-graph",
                                        figure=barplot_top_10_ventes(df),
                                        style={"height": "100%"},
                                    ),
                                    style={
                                        "height": "500px",
                                    },
                                )
                            ]
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(
                                        id="sales-trend-graph",
                                        figure=plot_evolution_chiffre_affaire(df),
                                        style={"height": "100%"},
                                    ),
                                    style={"height": "300px"},
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5(
                                            "Table des 100 dernières ventes",
                                            className="text-dark",
                                        ),
                                        sales_table,
                                    ],
                                    style={"height": "300px"},
                                )
                            ]
                        ),
                    ],
                    md=7, 
                ),
            ],
            align="stretch",
            style={"height": "86vh"},
        ),
    ],
)


# ================= Callback ================= #

@app.callback(
    [
        dash.dependencies.Output("top-sales-graph", "figure"),
        dash.dependencies.Output("sales-trend-graph", "figure"),
        dash.dependencies.Output("sales-table", "data"),
        dash.dependencies.Output("chiffre-affaire-mois", "figure"),
        dash.dependencies.Output("vente-mois", "figure"),
    ],
    [dash.dependencies.Input("zone-dropdown", "value")],
)
def update_dashboard(selected_zone):
    # Traiter None comme "all"
    if selected_zone is None:
        selected_zone = "all"

    filtered_df = df if selected_zone == "all" else df[df["Location"] == selected_zone].copy()
    filtered_df["Transaction_Date"] = pd.to_datetime(filtered_df["Transaction_Date"])
    filtered_df["Month"] = filtered_df["Transaction_Date"].dt.month

    top_sales_data = frequence_meilleure_vente(filtered_df, ascending=True)

    if "Total vente" not in top_sales_data.columns:
        raise ValueError("La colonne 'Total vente' n'existe pas dans le DataFrame.")

    updated_top_sales = px.bar(
        top_sales_data,
        x="Total vente",
        y=top_sales_data.index.get_level_values(1),
        color=top_sales_data.index.get_level_values(0),
        orientation="h",
        title="Frequence des 10 meilleures ventes",
        barmode="group",
        labels={"x": "Fréquence", "y": "Categorie du produit", "color": "Sexe"},
        color_discrete_map={"F": "#636EFA", "M": "#EF553B"},
    ).update_layout(margin=dict(t=60), height=500)

    updated_sales_trend = px.line(
        filtered_df.groupby(pd.Grouper(key="Transaction_Date", freq="W"))["Total_price"]
        .sum()
        .reset_index(),
        x="Transaction_Date",
        y="Total_price",
        title="Évolution du chiffre d'affaire par semaine",
        labels={
            "Transaction_Date": "Semaine",
            "Total_price": "Chiffre d'affaire",
        },
    ).update_layout(margin=dict(t=60, b=0), height=300)

    updated_table_data = (
        filtered_df.sort_values(by="Transaction_Date", ascending=False)
        .head(100)
        .to_dict("records")
    )

    updated_chiffre_affaire_mois = plot_chiffre_affaire_mois(filtered_df)
    updated_vente_mois = plot_vente_mois(filtered_df)

    return (
        updated_top_sales,
        updated_sales_trend,
        updated_table_data,
        updated_chiffre_affaire_mois,
        updated_vente_mois,
    )


# ================= Lancer l'application ================= #

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8080)