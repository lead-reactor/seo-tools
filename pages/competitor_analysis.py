"""
Competitor Semantic Analysis - Streamlit Page

Analyzes Top 10 Google results for a bucket query using:
- Custom Search JSON API (URLs)
- Cloud Natural Language API (entities/categories)
- Vertex AI Text Embeddings (clustering/gaps)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Analyse Concurrentielle Sémantique",
    page_icon="🔍",
    layout="wide",
)


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required Google Cloud packages are installed."""
    missing = []

    try:
        import google.cloud.language_v1
    except ImportError:
        missing.append("google-cloud-language")

    try:
        import vertexai
    except ImportError:
        missing.append("google-cloud-aiplatform")

    return len(missing) == 0, missing


def show_config_help():
    """Display configuration help in sidebar."""
    with st.sidebar:
        st.header("Configuration")

        st.markdown("""
        ### Variables d'environnement requises

        **Custom Search API (obligatoire):**
        ```
        GOOGLE_CUSTOM_SEARCH_API_KEY=xxx
        GOOGLE_CUSTOM_SEARCH_ENGINE_ID=xxx
        ```

        **Cloud NLP & Vertex AI:**
        ```
        GOOGLE_CLOUD_PROJECT=your-project
        GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
        ```

        ### Obtenir les clés API

        1. **Custom Search API**
           - [Console API](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)
           - [Créer un moteur](https://programmablesearchengine.google.com/)

        2. **Cloud Natural Language**
           - [Activer l'API](https://console.cloud.google.com/apis/library/language.googleapis.com)

        3. **Vertex AI**
           - [Activer l'API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)
        """)


def display_search_results(pages: list):
    """Display Top 10 search results."""
    st.subheader("Top 10 - Résultats de recherche")

    df = pd.DataFrame([
        {
            "Position": p.position,
            "Titre": p.title[:60] + "..." if len(p.title) > 60 else p.title,
            "URL": p.url,
            "Mots": p.word_count if p.scrape_success else "N/A",
            "Statut": "✅" if p.scrape_success else "❌",
        }
        for p in pages
    ])

    st.dataframe(
        df,
        column_config={
            "URL": st.column_config.LinkColumn("URL"),
        },
        hide_index=True,
        use_container_width=True,
    )


def display_entity_analysis(result):
    """Display entity frequency analysis."""
    st.subheader("Analyse des Entités")

    top_entities = result.get_top_entities(30)

    if not top_entities:
        st.warning("Aucune entité extraite. Vérifiez la configuration Cloud NLP.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Entity frequency chart
        df = pd.DataFrame([
            {
                "Entité": e.name,
                "Pages": e.pages_count,
                "Mentions": e.total_mentions,
                "Type": e.type,
                "Salience": round(e.avg_salience, 3),
            }
            for e in top_entities[:20]
        ])

        fig = px.bar(
            df,
            x="Pages",
            y="Entité",
            orientation="h",
            color="Type",
            title="Entités les plus fréquentes (par nombre de pages)",
            hover_data=["Mentions", "Salience"],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Entity type distribution
        type_counts = {}
        for e in top_entities:
            type_counts[e.type] = type_counts.get(e.type, 0) + 1

        fig = px.pie(
            names=list(type_counts.keys()),
            values=list(type_counts.values()),
            title="Distribution par type d'entité",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Coverage threshold filter
        st.markdown("### Entités à fort coverage")
        min_coverage = st.slider("Minimum de pages", 3, 10, 5)
        high_coverage = [e for e in top_entities if e.pages_count >= min_coverage]

        if high_coverage:
            st.markdown("**Entités présentes dans {}+ pages:**".format(min_coverage))
            for e in high_coverage[:10]:
                st.markdown(f"- **{e.name}** ({e.pages_count} pages, {e.type})")
        else:
            st.info("Aucune entité n'atteint ce seuil.")


def display_category_analysis(result):
    """Display content category distribution."""
    st.subheader("Catégories de contenu")

    if not result.category_distribution:
        st.warning("Aucune catégorie extraite.")
        return

    df = pd.DataFrame([
        {"Catégorie": k, "Occurrences": v}
        for k, v in sorted(result.category_distribution.items(), key=lambda x: -x[1])
    ])

    fig = px.bar(
        df,
        x="Occurrences",
        y="Catégorie",
        orientation="h",
        title="Distribution des catégories thématiques",
        color="Occurrences",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)


def display_clusters(result):
    """Display semantic clusters."""
    st.subheader("Clusters Sémantiques")

    if not result.content_clusters:
        st.info("Clustering non disponible (nécessite Vertex AI).")
        return

    for cluster in result.content_clusters:
        with st.expander(f"Cluster {cluster.cluster_id + 1} ({len(cluster.urls)} pages)"):
            st.markdown("**URLs:**")
            for url in cluster.urls:
                st.markdown(f"- {url}")

            if cluster.sample_headings:
                st.markdown("**Thèmes dominants (headings):**")
                for h in cluster.sample_headings[:5]:
                    st.markdown(f"- {h}")


def display_gaps(result):
    """Display semantic gaps."""
    st.subheader("Gaps Sémantiques")

    if not result.semantic_gaps:
        st.info("Analyse des gaps non disponible. Spécifiez une URL cible.")
        return

    st.markdown("**Sujets traités par les concurrents mais absents/faibles chez vous:**")

    df = pd.DataFrame([
        {"Sujet": gap[0][:100], "Similarité": round(gap[1], 3)}
        for gap in result.semantic_gaps[:20]
    ])

    st.dataframe(df, hide_index=True, use_container_width=True)


def display_statistics(result):
    """Display summary statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pages analysées", f"{result.successful_pages}/10")

    with col2:
        st.metric("Mots moyens", f"{int(result.avg_word_count)}")

    with col3:
        st.metric("Entités uniques", result.total_unique_entities)

    with col4:
        st.metric("Catégories", len(result.category_distribution))


def main():
    st.title("🔍 Analyse Concurrentielle Sémantique")
    st.markdown("Analysez la sémantique du Top 10 Google pour un bucket de mots-clés.")

    # Check dependencies
    deps_ok, missing_deps = check_dependencies()

    if not deps_ok:
        st.error(f"Packages manquants: {', '.join(missing_deps)}")
        st.code(f"pip install {' '.join(missing_deps)}")

    # Show config help
    show_config_help()

    # Import after dependency check to avoid errors
    try:
        from utils.google_config import GoogleAPIConfig
        from utils.competitor_analyzer import CompetitorAnalyzer
    except ImportError as e:
        st.error(f"Erreur d'import: {e}")
        return

    # Load config
    config = GoogleAPIConfig.from_env()
    missing_configs = config.get_missing_configs()

    if missing_configs:
        st.warning("Configuration incomplète:")
        for cfg in missing_configs:
            st.markdown(f"- {cfg}")

    # Main form
    with st.form("analysis_form"):
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "Requête (bucket)",
                placeholder="ex: assurance auto pas cher",
                help="La requête principale représentant votre bucket sémantique"
            )

        with col2:
            num_results = st.slider("Nombre de résultats", 5, 10, 10)

        col3, col4 = st.columns(2)

        with col3:
            target_url = st.text_input(
                "URL cible (optionnel)",
                placeholder="https://votre-site.com/page",
                help="Votre page à comparer pour l'analyse de gaps"
            )

        with col4:
            language = st.selectbox("Langue", ["fr", "en", "de", "es", "it"], index=0)

        # Options
        col5, col6 = st.columns(2)
        with col5:
            enable_nlp = st.checkbox("Activer Cloud NLP (entités/catégories)", value=True)
        with col6:
            enable_embeddings = st.checkbox("Activer Vertex AI (clustering/gaps)", value=True)

        submitted = st.form_submit_button("Analyser le Top 10", type="primary")

    if submitted and query:
        if not config.validate_custom_search():
            st.error("Custom Search API non configurée. Voir la sidebar.")
            return

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(step: int, total: int, message: str):
            progress_bar.progress(step / total)
            status_text.text(message)

        try:
            analyzer = CompetitorAnalyzer(
                config=config,
                enable_nlp=enable_nlp,
                enable_embeddings=enable_embeddings,
            )

            result = analyzer.analyze_bucket(
                query=query,
                num_results=num_results,
                language=language,
                target_url=target_url if target_url else None,
                progress_callback=update_progress,
            )

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            # Display results
            st.success(f"Analyse terminée pour: **{query}**")

            # Statistics
            display_statistics(result)

            st.divider()

            # Tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Top 10",
                "🏷️ Entités",
                "📁 Catégories",
                "🔗 Clusters",
                "⚠️ Gaps"
            ])

            with tab1:
                display_search_results(result.pages)

            with tab2:
                display_entity_analysis(result)

            with tab3:
                display_category_analysis(result)

            with tab4:
                display_clusters(result)

            with tab5:
                display_gaps(result)

            # Export
            st.divider()
            st.subheader("Export")

            col1, col2 = st.columns(2)

            with col1:
                # Export entities as CSV
                if result.common_entities:
                    entities_df = pd.DataFrame([e.to_dict() for e in result.common_entities])
                    csv = entities_df.to_csv(index=False)
                    st.download_button(
                        "Télécharger les entités (CSV)",
                        csv,
                        f"entities_{query.replace(' ', '_')}.csv",
                        "text/csv",
                    )

            with col2:
                # Export full results as JSON
                import json
                json_data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
                st.download_button(
                    "Télécharger l'analyse complète (JSON)",
                    json_data,
                    f"analysis_{query.replace(' ', '_')}.json",
                    "application/json",
                )

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Erreur lors de l'analyse: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
