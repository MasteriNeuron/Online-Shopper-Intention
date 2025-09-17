import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from src.logger.logs import setup_logger

logger = setup_logger()

def perform_eda(df, output_dir='static/plots'):
    logger.info("Starting EDA aligned with masterof_eda.py")
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify input data
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Revenue Distribution
        logger.info("Generating revenue distribution plot")
        revenue_counts = df['Revenue'].value_counts().reset_index()
        revenue_counts.columns = ['Revenue', 'Count']
        fig = px.bar(revenue_counts, x='Revenue', y='Count', color='Revenue',
                     color_discrete_map={True: "#2ca02c", False: "#d62728"},
                     title='Revenue Distribution')
        with open(os.path.join(output_dir, 'revenue_distribution.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # Correlation Heatmap
        logger.info("Generating correlation heatmap")
        numeric_cols = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
                        "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType"]
        corr_matrix = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=numeric_cols, y=numeric_cols,
            colorscale='coolwarm', zmin=-1, zmax=1, text=corr_matrix.values.round(2),
            texttemplate="%{text}", showscale=True
        ))
        fig.update_layout(title='Correlation Heatmap of Numerical Features')
        with open(os.path.join(output_dir, 'correlation_heatmap.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # Bounce Rate vs Exit Rate
        logger.info("Generating bounce vs exit rate scatter plot")
        fig = px.scatter(df, x='BounceRates', y='ExitRates', color='Revenue',
                         color_discrete_map={True: "#2ca02c", False: "#d62728"},
                         title='Bounce Rate vs Exit Rate')
        with open(os.path.join(output_dir, 'bounce_vs_exit.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # Visitor Type Distribution
        logger.info("Generating visitor type pie chart")
        visitor_counts = df['VisitorType'].value_counts().reset_index()
        visitor_counts.columns = ['VisitorType', 'Count']
        fig = px.pie(visitor_counts, values='Count', names='VisitorType',
                     color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
                     title="Visitor Type Distribution")
        with open(os.path.join(output_dir, 'visitor_type_pie.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # Visitor Type vs Revenue
        logger.info("Generating VisitorType vs Revenue bar plot")
        visitor_revenue = df.groupby('VisitorType')['Revenue'].mean().reset_index()
        fig = px.bar(visitor_revenue, x='VisitorType', y='Revenue',
                     color_discrete_sequence=["#2ca02c"],
                     title='Average Revenue by Visitor Type')
        with open(os.path.join(output_dir, 'visitor_vs_revenue.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # Numerical Feature Distributions
        logger.info("Generating histograms for numerical features")
        for col in numeric_cols:
            fig = px.histogram(df, x=col, color='Revenue',
                               color_discrete_map={True: "#2ca02c", False: "#d62728"},
                               title=f'Distribution of {col} by Revenue',
                               barmode='overlay', histnorm='probability density')
            with open(os.path.join(output_dir, f'{col.lower()}_histogram.json'), 'w') as f:
                json.dump(fig.to_dict(), f)
        
        # Categorical Feature Analysis
        logger.info("Generating categorical feature vs Revenue bar plots")
        categorical_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
        for col in categorical_cols:
            cat_revenue = df.groupby(col)['Revenue'].mean().reset_index()
            fig = px.bar(cat_revenue, x=col, y='Revenue',
                         color_discrete_sequence=["#2ca02c"],
                         title=f'Average Revenue by {col}')
            with open(os.path.join(output_dir, f'{col.lower()}_vs_revenue.json'), 'w') as f:
                json.dump(fig.to_dict(), f)
        
        # Weekend vs Revenue
        logger.info("Generating Weekend vs Revenue bar plot")
        weekend_revenue = df.groupby('Weekend')['Revenue'].mean().reset_index()
        fig = px.bar(weekend_revenue, x='Weekend', y='Revenue',
                     color_discrete_sequence=["#2ca02c"],
                     title='Average Revenue by Weekend')
        with open(os.path.join(output_dir, 'weekend_vs_revenue.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # Month vs Revenue
        logger.info("Generating Month vs Revenue bar plot")
        month_revenue = df.groupby('Month')['Revenue'].mean().reset_index()
        fig = px.bar(month_revenue, x='Month', y='Revenue',
                     color_discrete_sequence=["#2ca02c"],
                     title='Average Revenue by Month')
        with open(os.path.join(output_dir, 'month_vs_revenue.json'), 'w') as f:
            json.dump(fig.to_dict(), f)
        
        # PCA Scatter Plot
        logger.info("Generating PCA scatter plot with clusters")
        try:
            preprocessed_df = pd.read_csv('datasets/processed/preprocessed_data.csv')
            fig = px.scatter(preprocessed_df, x='PC1', y='PC2', color='Revenue', symbol='Cluster',
                             color_discrete_map={True: "#2ca02c", False: "#d62728"},
                             title='PCA Components with K-Means Clusters')
            with open(os.path.join(output_dir, 'pca_scatter.json'), 'w') as f:
                json.dump(fig.to_dict(), f)
        except FileNotFoundError:
            logger.warning("Could not generate PCA scatter plot: preprocessed_data.csv not found")
        
        logger.info("All EDA plots generated successfully")
    
    except Exception as e:
        logger.error(f"EDA plot generation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        df = pd.read_csv('datasets/processed/clean.csv')
        perform_eda(df)
    except FileNotFoundError:
        logger.error("clean.csv not found in datasets/processed")
        raise