"""
Application Streamlit d'analyse sémantique TF-IDF.
Analyse une ou plusieurs URLs pour identifier les opportunités SEO.
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.scraper import WebScraper
from utils.text_processor import TextProcessor
from utils.tfidf_analyzer import TFIDFAnalyzer
from utils.visualizations import SemanticVisualizer

# Configuration de la page
st.set_page_config(
    page_title="Analyse Sémantique TF-IDF",
    page_icon="🔍",
    layout="wide"
)

# Initialisation des objets
@st.cache_resource
def init_tools():
    """Initialize analysis tools."""
    return {
        'scraper': WebScraper(),
        'processor': TextProcessor(),
        'analyzer': TFIDFAnalyzer(),
        'visualizer': SemanticVisualizer()
    }

tools = init_tools()

# Titre et description
st.title("🔍 Analyse Sémantique TF-IDF")
st.markdown("""
Analysez la sémantique d'une page web ou comparez jusqu'à 5 URLs concurrentes.
L'analyse inclut : TF-IDF, densité des mots-clés, termes manquants, visualisations.
""")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

analysis_mode = st.sidebar.radio(
    "Mode d'analyse",
    ["Simple (1 URL)", "Comparaison (2-5 URLs)"],
    help="Analysez une seule URL ou comparez plusieurs URLs"
)

st.sidebar.divider()

# Paramètres d'analyse
st.sidebar.subheader("Paramètres")
top_n_terms = st.sidebar.slider("Nombre de termes à afficher", 10, 100, 30, 5)
show_unigrams = st.sidebar.checkbox("Uni-grammes", value=True)
show_bigrams = st.sidebar.checkbox("Bi-grammes", value=True)
show_trigrams = st.sidebar.checkbox("Tri-grammes", value=True)
show_wordcloud = st.sidebar.checkbox("Afficher les nuages de mots", value=True)

st.sidebar.divider()
st.sidebar.info("User-Agent: **LeadFactorBot**")

# Main content
if analysis_mode == "Simple (1 URL)":
    st.header("📊 Analyse Simple")

    url = st.text_input(
        "URL à analyser",
        placeholder="https://exemple.com/page",
        help="Entrez l'URL complète de la page à analyser"
    )

    if st.button("🚀 Lancer l'analyse", type="primary"):
        if not url or not url.startswith('http'):
            st.error("❌ Veuillez entrer une URL valide commençant par http:// ou https://")
        else:
            with st.spinner("🔄 Extraction du contenu..."):
                # Extract content
                content = tools['scraper'].extract_seo_content(url)

                if 'error' in content:
                    st.error(f"❌ Erreur lors de l'extraction : {content['error']}")
                else:
                    text = content['all_text']

                    if not text:
                        st.warning("⚠️ Aucun contenu textuel trouvé sur cette page")
                    else:
                        # Display page info
                        st.success(f"✅ Contenu extrait : {content['text_length']} caractères")

                        with st.expander("ℹ️ Informations de la page"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Titre:**", content['title'])
                                st.write("**H1:**", ", ".join(content['h1']) if content['h1'] else "Aucun")
                            with col2:
                                st.write("**Meta Description:**", content['meta_description'])
                                st.write("**Nombre de H2:**", len(content['h2']))

                        st.divider()

                        # Process text
                        with st.spinner("📝 Analyse du texte..."):
                            ngrams = tools['processor'].extract_all_ngrams(text, remove_stopwords=True)

                        # Tabs for different analyses
                        tab1, tab2, tab3 = st.tabs(["📈 TF-IDF", "📊 Densité", "☁️ Nuages de mots"])

                        with tab1:
                            st.subheader("Analyse TF-IDF")
                            st.info("Pour une seule URL, les scores représentent la fréquence relative des termes.")

                            # Unigrams
                            if show_unigrams:
                                st.markdown("### Uni-grammes")
                                col1, col2 = st.columns([1, 1])

                                with col1:
                                    unigram_freq = tools['processor'].get_top_terms(ngrams['unigrams'], top_n_terms)
                                    if unigram_freq:
                                        import pandas as pd
                                        df_uni = pd.DataFrame(unigram_freq, columns=['Terme', 'Fréquence'])
                                        st.dataframe(df_uni, height=400, use_container_width=True)

                                with col2:
                                    if unigram_freq:
                                        import pandas as pd
                                        df_uni = pd.DataFrame(unigram_freq, columns=['term', 'frequency'])
                                        fig = tools['visualizer'].create_bar_chart(
                                            df_uni, 'frequency', 'term',
                                            'Top Uni-grammes', top_n=20
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                            # Bigrams
                            if show_bigrams:
                                st.markdown("### Bi-grammes")
                                col1, col2 = st.columns([1, 1])

                                with col1:
                                    bigram_freq = tools['processor'].get_top_terms(ngrams['bigrams'], top_n_terms)
                                    if bigram_freq:
                                        import pandas as pd
                                        df_bi = pd.DataFrame(bigram_freq, columns=['Terme', 'Fréquence'])
                                        st.dataframe(df_bi, height=400, use_container_width=True)

                                with col2:
                                    if bigram_freq:
                                        import pandas as pd
                                        df_bi = pd.DataFrame(bigram_freq, columns=['term', 'frequency'])
                                        fig = tools['visualizer'].create_bar_chart(
                                            df_bi, 'frequency', 'term',
                                            'Top Bi-grammes', top_n=20, color='lightcoral'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                            # Trigrams
                            if show_trigrams:
                                st.markdown("### Tri-grammes")
                                col1, col2 = st.columns([1, 1])

                                with col1:
                                    trigram_freq = tools['processor'].get_top_terms(ngrams['trigrams'], top_n_terms)
                                    if trigram_freq:
                                        import pandas as pd
                                        df_tri = pd.DataFrame(trigram_freq, columns=['Terme', 'Fréquence'])
                                        st.dataframe(df_tri, height=400, use_container_width=True)

                                with col2:
                                    if trigram_freq:
                                        import pandas as pd
                                        df_tri = pd.DataFrame(trigram_freq, columns=['term', 'frequency'])
                                        fig = tools['visualizer'].create_bar_chart(
                                            df_tri, 'frequency', 'term',
                                            'Top Tri-grammes', top_n=20, color='lightgreen'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                        with tab2:
                            st.subheader("Densité des mots-clés")

                            if show_unigrams:
                                density = tools['processor'].calculate_density(ngrams['unigrams'], top_n=top_n_terms)
                                if density:
                                    col1, col2 = st.columns([1, 1])

                                    with col1:
                                        import pandas as pd
                                        df_density = pd.DataFrame(list(density.items()), columns=['Terme', 'Densité (%)'])
                                        df_density['Densité (%)'] = df_density['Densité (%)'].round(2)
                                        st.dataframe(df_density, height=400, use_container_width=True)

                                    with col2:
                                        fig = tools['visualizer'].create_density_chart(
                                            density,
                                            "Densité des mots-clés (%)",
                                            top_n=20
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                        with tab3:
                            if show_wordcloud:
                                st.subheader("Nuages de mots")

                                if show_unigrams and ngrams['unigrams']:
                                    st.markdown("#### Uni-grammes")
                                    freq_dict = dict(tools['processor'].get_top_terms(ngrams['unigrams'], 100))
                                    tools['visualizer'].create_wordcloud(freq_dict, "Nuage de mots - Uni-grammes")

                                if show_bigrams and ngrams['bigrams']:
                                    st.markdown("#### Bi-grammes")
                                    freq_dict = dict(tools['processor'].get_top_terms(ngrams['bigrams'], 100))
                                    tools['visualizer'].create_wordcloud(freq_dict, "Nuage de mots - Bi-grammes")

                                if show_trigrams and ngrams['trigrams']:
                                    st.markdown("#### Tri-grammes")
                                    freq_dict = dict(tools['processor'].get_top_terms(ngrams['trigrams'], 100))
                                    tools['visualizer'].create_wordcloud(freq_dict, "Nuage de mots - Tri-grammes")

else:  # Comparison mode
    st.header("🔄 Analyse Comparative")

    # URL inputs
    st.markdown("**URL Cible (votre site)**")
    target_url = st.text_input(
        "Votre URL",
        placeholder="https://votre-site.com/page",
        key="target"
    )

    st.markdown("**URLs Concurrentes (1-4 URLs)**")
    competitor_urls = []
    num_competitors = st.number_input("Nombre de concurrents", min_value=1, max_value=4, value=2)

    for i in range(num_competitors):
        url = st.text_input(
            f"Concurrent {i+1}",
            placeholder=f"https://concurrent-{i+1}.com/page",
            key=f"competitor_{i}"
        )
        if url:
            competitor_urls.append(url)

    if st.button("🚀 Lancer la comparaison", type="primary"):
        if not target_url or not target_url.startswith('http'):
            st.error("❌ Veuillez entrer une URL cible valide")
        elif len(competitor_urls) == 0:
            st.error("❌ Veuillez entrer au moins une URL concurrente")
        elif not all(url.startswith('http') for url in competitor_urls):
            st.error("❌ Toutes les URLs concurrentes doivent être valides")
        else:
            all_urls = [target_url] + competitor_urls

            with st.spinner(f"🔄 Extraction du contenu de {len(all_urls)} URLs..."):
                # Extract all contents
                contents = {}
                errors = []

                progress_bar = st.progress(0)
                for idx, url in enumerate(all_urls):
                    content = tools['scraper'].extract_seo_content(url)
                    if 'error' in content:
                        errors.append(f"{url}: {content['error']}")
                    else:
                        contents[url] = content
                    progress_bar.progress((idx + 1) / len(all_urls))

                progress_bar.empty()

                if errors:
                    st.warning(f"⚠️ Erreurs lors de l'extraction :\n" + "\n".join(errors))

                if len(contents) < 2:
                    st.error("❌ Au moins 2 URLs doivent être extraites avec succès pour la comparaison")
                else:
                    st.success(f"✅ {len(contents)} URLs extraites avec succès")

                    # Prepare documents for analysis
                    documents = {url: content['all_text'] for url, content in contents.items()}

                    # Identify target and competitors
                    target_name = "🎯 Votre site"
                    competitor_names = [f"🔵 Concurrent {i+1}" for i in range(len(competitor_urls))]

                    documents_named = {target_name: documents.get(target_url, '')}
                    for i, url in enumerate(competitor_urls):
                        if url in documents:
                            documents_named[competitor_names[i]] = documents[url]

                    st.divider()

                    # Display metrics
                    st.subheader("📊 Métriques générales")
                    metrics = {
                        "URLs analysées": len(documents),
                        "Mots cible": len(tools['processor'].tokenize(documents.get(target_url, ''))),
                    }
                    total_competitor_words = sum(
                        len(tools['processor'].tokenize(documents[url]))
                        for url in competitor_urls if url in documents
                    )
                    metrics["Mots concurrents"] = total_competitor_words

                    tools['visualizer'].display_metrics(metrics)

                    st.divider()

                    # Tabs for different analyses
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 Comparaison TF-IDF",
                        "🎯 Termes manquants",
                        "📊 Densité",
                        "☁️ Nuages de mots"
                    ])

                    with tab1:
                        st.subheader("Comparaison TF-IDF")

                        # Unigrams comparison
                        if show_unigrams:
                            st.markdown("### Uni-grammes")
                            with st.spinner("Analyse des uni-grammes..."):
                                unigram_results = tools['analyzer'].analyze_ngrams(
                                    documents_named, (1, 1), max_features=top_n_terms
                                )

                                if unigram_results:
                                    # Table comparison
                                    st.markdown("#### Tableau comparatif")
                                    for name, df in unigram_results.items():
                                        with st.expander(f"Voir les termes de {name}"):
                                            st.dataframe(df.head(30), use_container_width=True)

                                    # Chart
                                    st.markdown("#### Graphique comparatif")
                                    fig = tools['visualizer'].create_comparison_chart(
                                        unigram_results, 'tfidf_score',
                                        'Comparaison TF-IDF - Uni-grammes', top_n=15
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                        # Bigrams
                        if show_bigrams:
                            st.markdown("### Bi-grammes")
                            with st.spinner("Analyse des bi-grammes..."):
                                bigram_results = tools['analyzer'].analyze_ngrams(
                                    documents_named, (2, 2), max_features=top_n_terms
                                )

                                if bigram_results:
                                    for name, df in bigram_results.items():
                                        with st.expander(f"Voir les bi-grammes de {name}"):
                                            st.dataframe(df.head(30), use_container_width=True)

                                    fig = tools['visualizer'].create_comparison_chart(
                                        bigram_results, 'tfidf_score',
                                        'Comparaison TF-IDF - Bi-grammes', top_n=15
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                        # Trigrams
                        if show_trigrams:
                            st.markdown("### Tri-grammes")
                            with st.spinner("Analyse des tri-grammes..."):
                                trigram_results = tools['analyzer'].analyze_ngrams(
                                    documents_named, (3, 3), max_features=top_n_terms
                                )

                                if trigram_results:
                                    for name, df in trigram_results.items():
                                        with st.expander(f"Voir les tri-grammes de {name}"):
                                            st.dataframe(df.head(30), use_container_width=True)

                                    fig = tools['visualizer'].create_comparison_chart(
                                        trigram_results, 'tfidf_score',
                                        'Comparaison TF-IDF - Tri-grammes', top_n=15
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.subheader("🎯 Opportunités sémantiques")
                        st.markdown("Termes présents chez vos concurrents mais absents de votre page")

                        with st.spinner("Analyse des gaps sémantiques..."):
                            # Get target and competitor texts
                            target_text = documents.get(target_url, '')
                            competitor_texts = [documents[url] for url in competitor_urls if url in documents]

                            # Extract ngrams
                            target_ngrams = tools['processor'].extract_all_ngrams(target_text, remove_stopwords=True)
                            competitor_ngrams_list = [
                                tools['processor'].extract_all_ngrams(text, remove_stopwords=True)
                                for text in competitor_texts
                            ]

                            # Unigrams gaps
                            if show_unigrams:
                                st.markdown("### Uni-grammes manquants")
                                all_competitor_unigrams = []
                                for ngrams in competitor_ngrams_list:
                                    all_competitor_unigrams.extend(ngrams['unigrams'])

                                gaps = tools['processor'].find_unique_terms(
                                    target_ngrams['unigrams'],
                                    all_competitor_unigrams,
                                    top_n=50
                                )

                                if gaps:
                                    import pandas as pd
                                    df_gaps = pd.DataFrame(gaps, columns=['term', 'frequency'])

                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        st.dataframe(df_gaps.head(30), height=400, use_container_width=True)

                                    with col2:
                                        fig = tools['visualizer'].create_gap_analysis_chart(
                                            df_gaps.rename(columns={'frequency': 'avg_tfidf_score'}),
                                            "Opportunités - Uni-grammes",
                                            top_n=20
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Aucun uni-gramme manquant détecté")

                            # Bigrams gaps
                            if show_bigrams:
                                st.markdown("### Bi-grammes manquants")
                                all_competitor_bigrams = []
                                for ngrams in competitor_ngrams_list:
                                    all_competitor_bigrams.extend(ngrams['bigrams'])

                                gaps = tools['processor'].find_unique_terms(
                                    target_ngrams['bigrams'],
                                    all_competitor_bigrams,
                                    top_n=50
                                )

                                if gaps:
                                    import pandas as pd
                                    df_gaps = pd.DataFrame(gaps, columns=['term', 'frequency'])

                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        st.dataframe(df_gaps.head(30), height=400, use_container_width=True)

                                    with col2:
                                        fig = tools['visualizer'].create_gap_analysis_chart(
                                            df_gaps.rename(columns={'frequency': 'avg_tfidf_score'}),
                                            "Opportunités - Bi-grammes",
                                            top_n=20
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Aucun bi-gramme manquant détecté")

                            # Trigrams gaps
                            if show_trigrams:
                                st.markdown("### Tri-grammes manquants")
                                all_competitor_trigrams = []
                                for ngrams in competitor_ngrams_list:
                                    all_competitor_trigrams.extend(ngrams['trigrams'])

                                gaps = tools['processor'].find_unique_terms(
                                    target_ngrams['trigrams'],
                                    all_competitor_trigrams,
                                    top_n=50
                                )

                                if gaps:
                                    import pandas as pd
                                    df_gaps = pd.DataFrame(gaps, columns=['term', 'frequency'])

                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        st.dataframe(df_gaps.head(30), height=400, use_container_width=True)

                                    with col2:
                                        fig = tools['visualizer'].create_gap_analysis_chart(
                                            df_gaps.rename(columns={'frequency': 'avg_tfidf_score'}),
                                            "Opportunités - Tri-grammes",
                                            top_n=20
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Aucun tri-gramme manquant détecté")

                    with tab3:
                        st.subheader("📊 Densité des mots-clés")

                        # Show density for each URL
                        for url, text in documents.items():
                            name = target_name if url == target_url else f"🔵 Concurrent {competitor_urls.index(url) + 1}"

                            with st.expander(f"Densité - {name}"):
                                ngrams = tools['processor'].extract_all_ngrams(text, remove_stopwords=True)
                                density = tools['processor'].calculate_density(ngrams['unigrams'], top_n=30)

                                if density:
                                    col1, col2 = st.columns([1, 1])

                                    with col1:
                                        import pandas as pd
                                        df_density = pd.DataFrame(list(density.items()), columns=['Terme', 'Densité (%)'])
                                        df_density['Densité (%)'] = df_density['Densité (%)'].round(2)
                                        st.dataframe(df_density, height=400, use_container_width=True)

                                    with col2:
                                        fig = tools['visualizer'].create_density_chart(
                                            density,
                                            f"Densité - {name}",
                                            top_n=20
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        if show_wordcloud:
                            st.subheader("☁️ Nuages de mots comparatifs")

                            for url, text in documents.items():
                                name = target_name if url == target_url else f"🔵 Concurrent {competitor_urls.index(url) + 1}"

                                st.markdown(f"### {name}")
                                ngrams = tools['processor'].extract_all_ngrams(text, remove_stopwords=True)

                                if show_unigrams and ngrams['unigrams']:
                                    freq_dict = dict(tools['processor'].get_top_terms(ngrams['unigrams'], 100))
                                    tools['visualizer'].create_wordcloud(freq_dict, f"Uni-grammes - {name}")

                                st.divider()

# Footer
st.divider()
st.caption("🤖 Analyse sémantique TF-IDF pour le SEO | User-Agent: LeadFactorBot")
