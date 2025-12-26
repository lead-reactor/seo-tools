# SEO Tools - Analyse Sémantique

Suite d'outils Streamlit pour l'analyse sémantique SEO, incluant l'analyse TF-IDF classique et l'analyse concurrentielle avancée avec les APIs Google Cloud.

## Fonctionnalités

### 1. Analyse Sémantique TF-IDF (`pages/semantic_analysis.py`)

#### Mode Simple (1 URL)
- Extraction du contenu SEO (title, meta, headings, body, alt images)
- Analyse TF-IDF des uni-grammes, bi-grammes et tri-grammes
- Calcul de la densité des mots-clés
- Visualisations : tableaux, graphiques en barres, nuages de mots

#### Mode Comparaison (2-5 URLs)
- Comparaison de votre URL avec jusqu'à 4 concurrents
- Analyse TF-IDF comparative
- Identification des termes manquants (gap analysis)
- Opportunités sémantiques basées sur la concurrence

---

### 2. Analyse Concurrentielle Sémantique (`pages/competitor_analysis.py`)

Analyse avancée du Top 10 Google pour un bucket de mots-clés, utilisant les APIs Google Cloud.

#### Pipeline d'analyse

```
Requête bucket ("assurance auto pas cher")
         │
         ▼
┌─────────────────────────────────────┐
│ 1. Custom Search JSON API           │
│    Récupère les 10 premières URLs   │
│    du SERP Google                   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 2. Web Scraper                      │
│    Extrait: title, meta, headings,  │
│    body text, images alt            │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 3. Cloud Natural Language API       │
│    - Entités nommées (+ Knowledge   │
│      Graph MID si disponible)       │
│    - Catégories thématiques         │
│    - Salience (importance)          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 4. Vertex AI Text Embeddings        │
│    - Vecteurs 768 dimensions        │
│    - Clustering sémantique          │
│    - Détection de gaps              │
└─────────────────────────────────────┘
         │
         ▼
    CompetitorAnalysisResult
```

#### Outputs générés

| Output | Description | Usage SEO |
|--------|-------------|-----------|
| `common_entities` | Entités fréquentes cross-pages (nom, type, salience, pages_count) | Identifier les entités "obligatoires" à couvrir |
| `category_distribution` | Répartition des catégories thématiques | Comprendre le positionnement thématique du SERP |
| `content_clusters` | Groupes de pages sémantiquement similaires | Identifier les angles éditoriaux dominants |
| `semantic_gaps` | Sujets présents chez les concurrents mais absents chez vous | Prioriser la création de contenu |
| `avg_word_count` | Nombre moyen de mots sur le Top 10 | Benchmark de longueur de contenu |

---

## Structure du projet

```
seo-tools/
├── pages/
│   ├── semantic_analysis.py      # Analyse TF-IDF classique
│   └── competitor_analysis.py    # Analyse concurrentielle (Google APIs)
├── utils/
│   ├── __init__.py
│   ├── scraper.py                # Web scraping (User-Agent: LeadFactorBot)
│   ├── text_processor.py         # Traitement du texte, stopwords, n-grammes
│   ├── tfidf_analyzer.py         # Analyse TF-IDF
│   ├── visualizations.py         # Graphiques et visualisations
│   ├── google_config.py          # Configuration APIs Google
│   ├── google_search.py          # Custom Search JSON API
│   ├── google_nlp.py             # Cloud Natural Language API
│   ├── google_embeddings.py      # Vertex AI Text Embeddings
│   └── competitor_analyzer.py    # Orchestrateur analyse concurrentielle
├── data/
│   └── stopwords_fr.txt          # Stopwords français
├── examples/
│   └── competitor_analysis_output.json  # Exemple d'output
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Configurer les APIs Google (pour l'analyse concurrentielle)

#### Variables d'environnement requises

```bash
# Custom Search API (obligatoire pour le Top 10)
export GOOGLE_CUSTOM_SEARCH_API_KEY="votre-api-key"
export GOOGLE_CUSTOM_SEARCH_ENGINE_ID="votre-cx-id"

# Cloud NLP + Vertex AI
export GOOGLE_CLOUD_PROJECT="votre-projet-gcp"
export GOOGLE_APPLICATION_CREDENTIALS="/chemin/vers/service-account.json"

# Optionnel
export GOOGLE_CLOUD_LOCATION="europe-west1"  # défaut: us-central1
export VERTEX_EMBEDDING_MODEL="text-multilingual-embedding-002"
```

#### Obtenir les clés API

1. **Custom Search API**
   - Activer l'API : [Console GCP](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)
   - Créer un moteur de recherche : [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Récupérer le `cx` (Search Engine ID) et la clé API

2. **Cloud Natural Language API**
   - Activer : [Console GCP](https://console.cloud.google.com/apis/library/language.googleapis.com)

3. **Vertex AI**
   - Activer : [Console GCP](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)
   - Créer un service account avec les rôles `Vertex AI User` et `Cloud Natural Language API User`

---

## Utilisation

### Lancer l'application

```bash
# Analyse TF-IDF classique
streamlit run pages/semantic_analysis.py

# Analyse concurrentielle (Google APIs)
streamlit run pages/competitor_analysis.py
```

### Usage programmatique

```python
from utils import CompetitorAnalyzer, GoogleAPIConfig

# Charger la configuration depuis les variables d'environnement
config = GoogleAPIConfig.from_env()

# Initialiser l'analyseur
analyzer = CompetitorAnalyzer(
    config=config,
    enable_nlp=True,        # Cloud NLP (entités/catégories)
    enable_embeddings=True  # Vertex AI (clustering/gaps)
)

# Analyser un bucket
result = analyzer.analyze_bucket(
    query="assurance auto pas cher",
    num_results=10,
    language="fr",
    target_url="https://mon-site.fr/assurance-auto"  # pour gap analysis
)

# Accéder aux résultats
print(f"Pages analysées: {result.successful_pages}/10")
print(f"Mots moyens: {result.avg_word_count:.0f}")

# Top entités (présentes dans 6+ pages)
for entity in result.get_entities_coverage(min_pages=6):
    print(f"- {entity.name} ({entity.type}): {entity.pages_count} pages")

# Distribution des catégories
for category, count in result.category_distribution.items():
    print(f"- {category}: {count} pages")

# Clusters sémantiques
for cluster in result.content_clusters:
    print(f"Cluster {cluster.cluster_id}: {len(cluster.urls)} pages")
    print(f"  Thèmes: {cluster.sample_headings[:3]}")

# Gaps sémantiques (sujets à traiter)
for topic, similarity in result.semantic_gaps[:5]:
    print(f"Gap: {topic} (sim: {similarity:.2f})")

# Export JSON
import json
with open("analysis.json", "w") as f:
    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
```

---

## Exemple d'output

Voir le fichier complet : [`examples/competitor_analysis_output.json`](examples/competitor_analysis_output.json)

### Résumé de l'output

```json
{
  "bucket_query": "assurance auto pas cher",
  "successful_pages": 10,
  "avg_word_count": 2156.4,
  "total_unique_entities": 156,

  "common_entities": [
    {"name": "assurance auto", "pages_count": 10, "total_mentions": 287, "avg_salience": 0.412},
    {"name": "véhicule", "pages_count": 10, "total_mentions": 156, "avg_salience": 0.089},
    {"name": "garantie", "pages_count": 10, "total_mentions": 134, "avg_salience": 0.078},
    {"name": "bonus-malus", "pages_count": 9, "total_mentions": 76, "avg_salience": 0.048},
    {"name": "franchise", "pages_count": 8, "total_mentions": 72, "avg_salience": 0.045}
  ],

  "category_distribution": {
    "Finance": 10,
    "Insurance": 10,
    "Autos & Vehicles": 8
  },

  "content_clusters": [
    {
      "cluster_id": 0,
      "urls": ["lesfurets.com", "lelynx.fr", "assurland.com"],
      "sample_headings": ["Comparez les assurances", "Comment ça marche ?"]
    },
    {
      "cluster_id": 1,
      "urls": ["maaf.fr", "axa.fr", "allianz.fr", "macif.fr"],
      "sample_headings": ["Nos formules", "Les garanties incluses"]
    },
    {
      "cluster_id": 2,
      "urls": ["direct-assurance.fr", "olivier-assurance.com"],
      "sample_headings": ["100% en ligne", "Souscription rapide"]
    }
  ],

  "semantic_gaps": [
    {"topic": "Résiliation loi Hamon", "similarity": 0.34},
    {"topic": "Assurance véhicule électrique", "similarity": 0.28},
    {"topic": "Procédure constat amiable", "similarity": 0.31}
  ]
}
```

### Interprétation SEO

| Insight | Action |
|---------|--------|
| **Entités à 10/10 pages** : assurance auto, véhicule, garantie | Ces entités sont **obligatoires** dans votre contenu |
| **Entités à 6-8 pages** : bonus-malus, franchise, bris de glace | À intégrer pour une couverture complète |
| **3 clusters identifiés** : comparateurs, assureurs traditionnels, 100% en ligne | Choisir son angle éditorial ou couvrir les 3 |
| **Gaps** : loi Hamon, véhicule électrique, constat amiable | Créer ces contenus pour se différencier |
| **Moyenne 2156 mots** | Viser au moins 2000 mots pour être compétitif |

---

## Schéma de données

### CompetitorAnalysisResult

```python
@dataclass
class CompetitorAnalysisResult:
    bucket_query: str                    # Requête analysée
    search_response: SearchResponse      # Résultats Custom Search
    pages: list[PageAnalysis]            # Analyse détaillée de chaque page

    # Agrégations
    common_entities: list[EntityFrequency]   # Entités cross-pages
    category_distribution: dict[str, int]    # Catégories par fréquence
    content_clusters: list[SemanticCluster]  # Groupes sémantiques
    semantic_gaps: list[tuple[str, float]]   # Sujets manquants

    # Statistiques
    avg_word_count: float
    total_unique_entities: int
    successful_pages: int
```

### EntityFrequency

```python
@dataclass
class EntityFrequency:
    name: str           # "assurance auto"
    type: str           # "OTHER", "ORGANIZATION", "PERSON", "LOCATION"...
    total_mentions: int # Nombre total de mentions cross-pages
    pages_count: int    # Nombre de pages contenant cette entité
    avg_salience: float # Importance moyenne (0-1)
    mid: str | None     # Knowledge Graph ID ("/m/0xyz")
```

### PageAnalysis

```python
@dataclass
class PageAnalysis:
    url: str
    position: int                # Rang SERP (1-10)

    # Contenu extrait
    page_title: str
    meta_description: str
    headings: list[str]
    body_text: str
    word_count: int

    # Analyse NLP
    entities: list[Entity]       # Entités nommées
    categories: list[Category]   # Catégories thématiques

    # Embedding (768 dimensions)
    embedding: list[float]

    # Statuts
    scrape_success: bool
    nlp_success: bool
    embedding_success: bool
```

---

## APIs Google utilisées

| API | Endpoint | Usage |
|-----|----------|-------|
| **Custom Search JSON** | `customsearch.googleapis.com/v1` | Récupérer le Top 10 URLs |
| **Cloud Natural Language** | `language.googleapis.com` | Entités, catégories, syntaxe |
| **Vertex AI Embeddings** | `aiplatform.googleapis.com` | Vecteurs sémantiques 768-dim |
| **Knowledge Graph** (optionnel) | `kgsearch.googleapis.com` | Désambiguïsation d'entités |

### Modèles d'embedding supportés

| Modèle | Dimensions | Usage |
|--------|------------|-------|
| `text-multilingual-embedding-002` | 768 | **Recommandé** pour FR/multilingue |
| `text-embedding-004` | 768 | Optimisé pour l'anglais |
| `textembedding-gecko@003` | 768 | Legacy, toujours supporté |

---

## Coûts estimés (Google Cloud)

| API | Tarif | Pour 10 pages |
|-----|-------|---------------|
| Custom Search | $5 / 1000 requêtes | ~$0.005 |
| Cloud NLP - Entities | $1 / 1000 unités (1000 chars) | ~$0.05 |
| Cloud NLP - Classification | $2 / 1000 unités | ~$0.04 |
| Vertex AI Embeddings | $0.0001 / 1000 chars | ~$0.002 |
| **Total par analyse** | | **~$0.10** |

---

## Dépendances

### Core
- **Streamlit** >= 1.28.0 : Interface web
- **BeautifulSoup4** >= 4.12.0 : Parsing HTML
- **Scikit-learn** >= 1.3.0 : TF-IDF et clustering
- **Pandas** >= 2.0.0 : Manipulation de données
- **Plotly** >= 5.17.0 : Visualisations

### Google Cloud
- **google-cloud-language** >= 2.11.0 : Cloud NLP
- **google-cloud-aiplatform** >= 1.38.0 : Vertex AI

---

## Notes techniques

- **User-Agent** : `LeadFactorBot`
- **Timeout** : 30 secondes par requête HTTP
- **Stopwords** : 600+ mots français (`data/stopwords_fr.txt`)
- **Limite Custom Search** : 10 résultats max par requête
- **Limite Vertex AI** : 250 textes par batch d'embedding
- **Classification NLP** : Requiert minimum 20 mots

---

## Licence

MIT
