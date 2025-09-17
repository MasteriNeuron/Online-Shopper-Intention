from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from src.logger.logs import setup_logger

logger = setup_logger()

def create_dash_app(server):
    logger.info("Initializing Dash app at /eda/")
    dash_app = Dash(__name__, server=server, url_base_pathname='/eda/', suppress_callback_exceptions=True)
    
    # Load Data
    try:
        df = pd.read_csv(r'C:\Users\drsnc\Desktop\PW-Projects\online_shoppers_intention\datasets\processed\clean.csv')
        logger.info("Successfully loaded clean.csv")
    except Exception as e:
        logger.error(f"Failed to load clean.csv: {str(e)}")
        raise
    
    # Figures
    numeric_cols = [col for col in [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
        "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType"
    ] if col in df.columns]

    fig_revenue_dist = px.histogram(
        df, x="Revenue", color="Revenue",
        color_discrete_map={True: "#2ca02c", False: "#d62728"},
        title="Revenue Distribution",
        labels={"Revenue": "Revenue (True=1, False=0)"}
    )

    corr_matrix = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        title="Correlation Heatmap of Numerical Features",
        width=1000, height=800,
        zmin=-1, zmax=1
    )

    fig_bounce_exit = px.scatter(
        df, x="BounceRates", y="ExitRates",
        color="Revenue", trendline="ols",
        color_discrete_map={True: "#ff7f0e", False: "#1f77b4"},
        title="Bounce Rate vs Exit Rate"
    )

    visitor_counts = df['VisitorType'].value_counts().reset_index()
    visitor_counts.columns = ['VisitorType', 'Count']
    fig_visitor_type = px.pie(
        visitor_counts, names="VisitorType", values="Count",
        title="Visitor Type Distribution", hole=0.3
    )

    crosstab_vt = pd.crosstab(df['VisitorType'], df['Revenue'], normalize='index').reset_index()
    crosstab_vt_melt = crosstab_vt.melt(id_vars='VisitorType', var_name='Revenue', value_name='Proportion')
    fig_vt_revenue = px.bar(
        crosstab_vt_melt, x="VisitorType", y="Proportion",
        color="Revenue", barmode="stack",
        color_discrete_map={False: 'red', True: 'orange'},
        title="Visitor Type vs Revenue"
    )

    counts = df.groupby(['TrafficType', 'Revenue']).size().reset_index(name='count')
    fig_traffic = px.bar(
        counts, x="TrafficType", y="count", color="Revenue", barmode="stack",
        color_discrete_map={False: "#1f77b4", True: "#ff7f0e"},
        title="Revenue by Traffic Type",
        labels={"count": "Number of Sessions", "TrafficType": "Traffic Type"}
    )

    fig_month_sessions = px.histogram(
        df, x="Month", color="Revenue", barmode="group",
        title="Sessions by Month",
        color_discrete_map={True: "#ff7f0e", False: "#1f77b4"}
    )

    fig_exit_month = px.box(
        df, x="Month", y="ExitRates", color="Revenue",
        title="Exit Rate Trend by Month",
        color_discrete_map={False: "#1f77b4", True: "#ff7f0e"}
    )

    if "Engagement_Score" in df.columns:
        fig_engagement = px.violin(
            df, y="Engagement_Score", x="Revenue",
            box=True, points="all", title="Engagement Score vs Revenue",
            color="Revenue", color_discrete_map={True: "#2ca02c", False: "#d62728"}
        )
    else:
        fig_engagement = px.scatter(title="Engagement Score not available")

    # Dash Layout
    dash_app.layout = html.Div([
        html.H1("E-Commerce Revenue Dashboard", 
                style={"textAlign": "center", "color": "#C72C48", "padding": "15px"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(label="Revenue Overview", children=[
                    dcc.Graph(figure=fig_revenue_dist, style={"width": "100%"}),
                    dcc.Graph(figure=fig_corr, style={"width": "100%"}),
                    dcc.Graph(figure=fig_bounce_exit, style={"width": "100%"})
                ]),
                dcc.Tab(label="Visitor Insights", children=[
                    dcc.Graph(figure=fig_visitor_type, style={"width": "100%"}),
                    dcc.Graph(figure=fig_vt_revenue, style={"width": "100%"})
                ]),
                dcc.Tab(label="Traffic Analysis", children=[
                    dcc.Graph(figure=fig_traffic, style={"width": "100%"})
                ]),
                dcc.Tab(label="Seasonality Trends", children=[
                    dcc.Graph(figure=fig_month_sessions, style={"width": "100%"}),
                    dcc.Graph(figure=fig_exit_month, style={"width": "100%"})
                ]),
                dcc.Tab(label="Engagement Features", children=[
                    dcc.Graph(figure=fig_engagement, style={"width": "100%"})
                ])
            ],
            style={"backgroundColor": "#F8C8DC"},
            colors={"border": "#C72C48", "primary": "#C72C48", "background": "#F8C8DC"}
        )
    ], 
    style={
        "backgroundColor": "#F8C8DC", 
        "minHeight": "100vh", 
        "padding": "10px"
    })

    logger.info("Dash app layout set up")
    return dash_app