# SEO Tools - Analyse Sémantique TF-IDF

Application Streamlit d'analyse sémantique pour le SEO, permettant d'analyser une ou plusieurs URLs et identifier les opportunités d'optimisation.

## Fonctionnalités

### Analyse Simple (1 URL)
- Extraction du contenu SEO (title, meta, headings, body, alt images)
- Analyse TF-IDF des uni-grammes, bi-grammes et tri-grammes
- Calcul de la densité des mots-clés
- Visualisations : tableaux, graphiques en barres, nuages de mots

### Analyse Comparative (2-5 URLs)
- Comparaison de votre URL avec jusqu'à 4 concurrents
- Analyse TF-IDF comparative
- Identification des termes manquants (gap analysis)
- Opportunités sémantiques basées sur la concurrence
- Densité comparative des mots-clés
- Visualisations comparatives

## Structure du projet

```
seo-tools/
├── pages/
│   └── semantic_analysis.py      # Application Streamlit
├── utils/
│   ├── __init__.py
│   ├── scraper.py                # Web scraping (User-Agent: LeadFactorBot)
│   ├── text_processor.py         # Traitement du texte, stopwords, n-grammes
│   ├── tfidf_analyzer.py         # Analyse TF-IDF
│   └── visualizations.py         # Graphiques et visualisations
├── data/
│   └── stopwords_fr.txt          # Stopwords français
├── requirements.txt              # Dépendances Python
└── README.md
```

## Installation

1. Cloner le repository ou télécharger les fichiers

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Lancer l'application Streamlit :

```bash
streamlit run pages/semantic_analysis.py
```

L'application s'ouvrira dans votre navigateur par défaut.

### Mode Simple

1. Sélectionnez "Simple (1 URL)" dans la barre latérale
2. Entrez l'URL à analyser
3. Configurez les paramètres (uni/bi/tri-grammes, nombre de termes)
4. Cliquez sur "Lancer l'analyse"

### Mode Comparaison

1. Sélectionnez "Comparaison (2-5 URLs)" dans la barre latérale
2. Entrez votre URL cible
3. Entrez les URLs concurrentes (1 à 4)
4. Configurez les paramètres
5. Cliquez sur "Lancer la comparaison"

## Caractéristiques techniques

### Web Scraping
- User-Agent : **LeadFactorBot**
- Extraction SEO optimisée
- Exclusion des éléments non pertinents (nav, header, footer, scripts)
- Support des contenus en français

### Traitement du texte
- Stopwords français détaillés (600+ mots)
- Nettoyage et normalisation du texte
- Génération d'uni-grammes, bi-grammes et tri-grammes
- Calcul de densité en pourcentage

### Analyse TF-IDF
- Scikit-learn TfidfVectorizer
- Support des n-grammes personnalisables
- Analyse comparative multi-documents
- Gap analysis (termes manquants)

### Visualisations
- Tableaux interactifs (Streamlit)
- Graphiques en barres (Plotly)
- Nuages de mots (WordCloud)
- Graphiques comparatifs

## Dépendances principales

- **Streamlit** : Interface web interactive
- **BeautifulSoup4** : Parsing HTML
- **Requests** : HTTP requests
- **Scikit-learn** : TF-IDF et NLP
- **Plotly** : Graphiques interactifs
- **WordCloud** : Nuages de mots
- **Pandas** : Manipulation de données

## Notes

- Le scraping utilise le User-Agent "LeadFactorBot"
- Les stopwords français peuvent être personnalisés dans `data/stopwords_fr.txt`
- L'analyse est optimisée pour le contenu en français
- Timeout de 30 secondes par requête HTTP