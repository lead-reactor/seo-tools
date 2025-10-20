"""
Visualization module for semantic analysis results.
Handles charts, tables, and word clouds.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List
import streamlit as st


class SemanticVisualizer:
    """Create visualizations for semantic analysis."""

    def __init__(self):
        self.color_palette = px.colors.qualitative.Set2

    def display_dataframe(self, df: pd.DataFrame, title: str = None, height: int = 400):
        """
        Display a formatted DataFrame in Streamlit.

        Args:
            df: DataFrame to display
            title: Optional title
            height: Table height in pixels
        """
        if title:
            st.subheader(title)

        if df.empty:
            st.info("Aucune donnée à afficher")
            return

        st.dataframe(df, height=height, use_container_width=True)

    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                         title: str, top_n: int = 20, color: str = None) -> go.Figure:
        """
        Create horizontal bar chart.

        Args:
            df: DataFrame with data
            x_col: Column for x-axis (values)
            y_col: Column for y-axis (categories)
            title: Chart title
            top_n: Number of top items to show
            color: Bar color

        Returns:
            Plotly figure
        """
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Take top N
        plot_df = df.head(top_n).copy()

        # Sort by value for better visualization
        plot_df = plot_df.sort_values(x_col, ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df[x_col],
            y=plot_df[y_col],
            orientation='h',
            marker=dict(
                color=color or self.color_palette[0],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=plot_df[x_col].round(4),
            textposition='auto',
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title="",
            height=max(400, top_n * 25),
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    def create_comparison_chart(self, data_dict: Dict[str, pd.DataFrame],
                                value_col: str, title: str, top_n: int = 15) -> go.Figure:
        """
        Create grouped bar chart comparing multiple sources.

        Args:
            data_dict: Dictionary mapping source name to DataFrame
            value_col: Column name for values
            title: Chart title
            top_n: Number of top terms to show

        Returns:
            Plotly figure
        """
        if not data_dict:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Collect all terms and their scores
        all_terms = set()
        for df in data_dict.values():
            if not df.empty and 'term' in df.columns:
                all_terms.update(df.head(top_n)['term'].tolist())

        if not all_terms:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucun terme trouvé",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Limit to top terms overall
        all_terms = list(all_terms)[:top_n]

        fig = go.Figure()

        for idx, (source_name, df) in enumerate(data_dict.items()):
            if df.empty:
                continue

            # Get scores for each term
            scores = []
            for term in all_terms:
                term_df = df[df['term'] == term]
                if not term_df.empty:
                    scores.append(term_df[value_col].values[0])
                else:
                    scores.append(0)

            fig.add_trace(go.Bar(
                name=source_name,
                x=all_terms,
                y=scores,
                marker_color=self.color_palette[idx % len(self.color_palette)]
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Termes",
            yaxis_title=value_col.replace('_', ' ').title(),
            barmode='group',
            height=500,
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=40, b=100),
        )

        return fig

    def create_wordcloud(self, terms_dict: Dict[str, float], title: str = "Nuage de mots",
                        max_words: int = 100, width: int = 800, height: int = 400):
        """
        Create and display word cloud.

        Args:
            terms_dict: Dictionary mapping term to score/frequency
            title: Title for the word cloud
            max_words: Maximum number of words in cloud
            width: Image width
            height: Image height
        """
        if not terms_dict:
            st.info("Aucun terme disponible pour le nuage de mots")
            return

        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.5,
            colormap='viridis',
            min_font_size=10
        ).generate_from_frequencies(terms_dict)

        # Display
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)

        st.pyplot(fig)
        plt.close()

    def create_density_chart(self, density_dict: Dict[str, float], title: str,
                            top_n: int = 20) -> go.Figure:
        """
        Create bar chart for keyword density.

        Args:
            density_dict: Dictionary mapping term to density percentage
            title: Chart title
            top_n: Number of top terms

        Returns:
            Plotly figure
        """
        if not density_dict:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Convert to DataFrame
        df = pd.DataFrame(list(density_dict.items()), columns=['term', 'density'])
        df = df.head(top_n).sort_values('density', ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['density'],
            y=df['term'],
            orientation='h',
            marker=dict(
                color=df['density'],
                colorscale='Blues',
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=df['density'].round(2).astype(str) + '%',
            textposition='auto',
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Densité (%)",
            yaxis_title="",
            height=max(400, top_n * 25),
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    def display_metrics(self, metrics: Dict[str, any]):
        """
        Display metrics in Streamlit columns.

        Args:
            metrics: Dictionary of metric_name: value
        """
        cols = st.columns(len(metrics))

        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                if isinstance(value, float):
                    st.metric(label, f"{value:.2f}")
                else:
                    st.metric(label, value)

    def create_gap_analysis_chart(self, gap_df: pd.DataFrame, title: str = "Termes manquants",
                                  top_n: int = 20) -> go.Figure:
        """
        Create chart showing terms present in competitors but missing from target.

        Args:
            gap_df: DataFrame with 'term' and score columns
            title: Chart title
            top_n: Number of top terms

        Returns:
            Plotly figure
        """
        if gap_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucun écart détecté",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Get score column name
        score_col = 'avg_tfidf_score' if 'avg_tfidf_score' in gap_df.columns else 'tfidf_score'

        plot_df = gap_df.head(top_n).sort_values(score_col, ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df[score_col],
            y=plot_df['term'],
            orientation='h',
            marker=dict(
                color='coral',
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=plot_df[score_col].round(4),
            textposition='auto',
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Score TF-IDF moyen",
            yaxis_title="",
            height=max(400, top_n * 25),
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    def create_coverage_gauge(self, coverage_percentage: float) -> go.Figure:
        """
        Create gauge chart for semantic coverage.

        Args:
            coverage_percentage: Coverage percentage (0-100)

        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=coverage_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Couverture sémantique"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 75], 'color': "lightyellow"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(height=300)

        return fig
