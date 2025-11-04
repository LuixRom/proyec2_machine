import os
import re
import cv2
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# CONFIGURACIÓN DE LA APP
# ==========================
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# ESTILOS
# ==========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .cluster-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
        color: white;
    }
    .movie-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        background: white;
    }
    .movie-card:hover {
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
        transform: translateY(-5px);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { padding: 1rem 2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==========================
# DIAGNÓSTICO (siempre visible)
# ==========================
with st.sidebar.expander("🔧 Diagnóstico (click para abrir)", expanded=True):
    st.write("**Working dir**:", os.getcwd())
    try:
        files = os.listdir()
        st.write("**Archivos en la carpeta**:", files)
        found = []
        for fname in [
            "combined_features_clustered.csv",
            "pca_features_clustered.csv",
            "umap_features_clustered.csv",
        ]:
            if os.path.exists(fname):
                found.append(fname)
        if found:
            st.success(f"Encontrado(s): {found}")
        else:
            st.error("No se encontró ningún archivo *_clustered.csv")
            st.info("Coloca junto a app.py alguno de: "
                    "`combined_features_clustered.csv`, "
                    "`pca_features_clustered.csv`, "
                    "`umap_features_clustered.csv`.")
    except Exception as e:
        st.error(f"Error listando archivos: {e}")

# ==========================
# CONSTANTES / UTILS
# ==========================
PLACEHOLDER = "https://via.placeholder.com/150x225/667eea/ffffff?text=Movie+Poster"
POSTER_FOLDER = "Posterss"  # Carpeta local con pósters

def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", str(text)).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)

def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")

def _is_valid_http_url(s: str) -> bool:
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or s.lower() in {"none", "nan", "null", "0", "-", "na"}:
        return False
    return s.startswith("http://") or s.startswith("https://")

def _is_tmdb_path(s: str) -> bool:
    return isinstance(s, str) and s.strip().startswith("/") and s.strip().lower().endswith((".jpg", ".png", ".jpeg"))

def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return None

@st.cache_data(show_spinner=False)
def _poster_index():
    """Indexa una vez los archivos de Posterss para búsqueda flexible."""
    base = Path(__file__).resolve().parent / POSTER_FOLDER
    files = []
    if base.exists():
        for p in base.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                files.append(p)
    names_ci = [p.stem.casefold() for p in files]
    return base, files, names_ci

def get_poster_source(row: dict) -> str:
    """
    Prioridad:
    1) URL directa (poster_url, image_url)
    2) TMDb poster_path -> URL
    3) Archivo local en Posterss/ probando: movieId, título EXACTO, ASCII, slug y búsqueda contains.
    4) Placeholder
    """
    # 1) URLs directas
    for col in ("poster_url", "image_url"):
        if col in row and _is_valid_http_url(row[col]):
            return row[col].strip()

    # 2) TMDb path
    if "poster_path" in row and _is_tmdb_path(row["poster_path"]):
        return f"https://image.tmdb.org/t/p/w342{row['poster_path'].strip()}"

    # 3) Local
    base, files, names_ci = _poster_index()
    if not base.exists():
        return PLACEHOLDER

    # 3.a) por movieId
    if "movieId" in row and pd.notna(row["movieId"]):
        try:
            mid = int(row["movieId"])
            hit = _first_existing([base / f"{mid}{ext}" for ext in (".jpg", ".png", ".jpeg")])
            if hit:
                return hit
        except Exception:
            pass

    # 3.b) por título (varias estrategias)
    title = None
    if "title" in row and pd.notna(row["title"]):
        title = str(row["title"]).strip()

    if title:
        # EXACTO
        hit = _first_existing([base / f"{title}{ext}" for ext in (".jpg", ".png", ".jpeg")])
        if hit:
            return hit

        # ASCII
        title_ascii = _ascii(title)
        hit = _first_existing([base / f"{title_ascii}{ext}" for ext in (".jpg", ".png", ".jpeg")])
        if hit:
            return hit

        # SLUG
        title_slug = _slugify(title)
        hit = _first_existing([base / f"{title_slug}{ext}" for ext in (".jpg", ".png", ".jpeg")])
        if hit:
            return hit

        # CONTAINS (case-insensitive, sin tildes)
        q = _ascii(title).casefold()
        candidates = [(len(n), i) for i, n in enumerate(names_ci) if q in _ascii(n).casefold()]
        if candidates:
            _, best_i = max(candidates)  # prioriza coincidencia más larga
            return str(files[best_i])

    # 4) Fallback
    return PLACEHOLDER

# --- helper robusto de géneros ---
def _genres_set(s):
    if pd.isna(s):
        return set()
    return set([g.strip() for g in str(s).split('|') if g.strip()])

# ==========================
# CARGA DE DATOS
# ==========================
@st.cache_data(show_spinner=False)
def load_movie_metadata():
    """Intenta cargar metadatos opcionales."""
    for candidate in ["movies_train.csv", "movies.csv", "metadata.csv"]:
        if os.path.exists(candidate):
            try:
                md = pd.read_csv(candidate)
                return md
            except Exception:
                pass
    return None

@st.cache_data(show_spinner=False)
def load_clustered_data():
    """Carga el dataset clusterizado disponible y devuelve df, feature_cols, método, archivo."""
    candidates = [
        ("best_method_clustered.csv", "UMAP"),
    ]
    chosen = None
    for fname, method in candidates:
        if os.path.exists(fname):
            chosen = (fname, method)
            break

    if chosen is None:
        return None, None, None, None

    df = pd.read_csv(chosen[0])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {'cluster', 'movieId', 'id', 'index', 'year'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if 'cluster' not in df.columns:
        return None, None, None, chosen[0]

    return df, feature_cols, chosen[1], chosen[0]

# ==========================
# EXTRACCIÓN DE FEATURES (IMAGEN SUBIDA) - DEMO
# ==========================
def extract_features_from_image(image: Image.Image):
    """DEMO simple: histograma RGB + stats (para producción replica tu pipeline real)."""
    img_array = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img_array, (100, 150))
    features = []
    for channel in range(3):
        hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
        features.extend(hist.flatten())
    features.extend([img_resized.mean(), img_resized.std(), img_resized.min(), img_resized.max()])
    return np.array(features, dtype=float)

# ==========================
# LÓGICA DE RECOMENDACIONES
# ==========================
def find_similar_movies(df_for_similarity, feature_cols, query_features, top_k=10):
    """
    Estandariza X del universo candidato y la query, luego usa cosine similarity.
    df_for_similarity es el universo sobre el que se hace el top-k.
    """
    X = df_for_similarity[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    query_scaled = scaler.transform(query_features.reshape(1, -1))
    similarities = cosine_similarity(query_scaled, X_scaled)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df_for_similarity.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    return results

def get_cluster_representatives(df, feature_cols, n_per_cluster=5):
    reps = []
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        centroid = cluster_data[feature_cols].mean().values
        distances = np.linalg.norm(cluster_data[feature_cols].values - centroid, axis=1)
        closest_indices = np.argsort(distances)[:n_per_cluster]
        cluster_reps = cluster_data.iloc[closest_indices].copy()
        cluster_reps['distance_to_centroid'] = distances[closest_indices]
        reps.append(cluster_reps)
    return pd.concat(reps, ignore_index=True) if reps else pd.DataFrame(columns=list(df.columns)+['distance_to_centroid'])

# ==========================
# PLOTS (NUEVOS: ejes con nombres reales)
# ==========================
def _pretty_axis(col_name: str) -> str:
    """Mapa de nombres legibles para ejes."""
    m = {
        "year": "Año",
        "age": "Edad",
        "runtime": "Duración (min)",
        "vote_average": "Puntaje",
        "vote_count": "Nº de votos",
        "popularity": "Popularidad",
        "budget": "Presupuesto",
        "revenue": "Recaudación",
    }
    key = str(col_name).lower()
    if key in m:
        return m[key]
    return str(col_name).replace("_", " ").title()

def plot_scatter_by_columns(df, x_col: str, y_col: str, color_by='cluster', title_suffix=''):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_by,
        hover_data=[c for c in ['title', 'genres', 'year', 'movieId'] if c in df.columns],
        title=f"Distribución de Películas {f'({title_suffix})' if title_suffix else ''}",
        height=600,
        labels={x_col: _pretty_axis(x_col), y_col: _pretty_axis(y_col)}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.80, line=dict(width=0.5, color='white')))
    fig.update_layout(
        plot_bgcolor='rgba(240, 242, 246, 0.6)',
        paper_bgcolor='white',
        font=dict(size=14, color='black'),
        title=dict(font=dict(size=20, color='black')),
        xaxis=dict(title_font=dict(size=16, color='black'), tickfont=dict(size=12, color='black'),
                   showgrid=True, gridcolor='lightgray', zeroline=False),
        yaxis=dict(title_font=dict(size=16, color='black'), tickfont=dict(size=12, color='black'),
                   showgrid=True, gridcolor='lightgray', zeroline=False),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    return fig

def plot_cluster_distribution(df):
    counts = df['cluster'].value_counts().sort_index()
    colors = (px.colors.qualitative.Set3 * ((len(counts) // 10) + 1))[:len(counts)]
    fig = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in counts.index],
            y=counts.values,
            marker_color=colors,
            text=counts.values,
            textposition='auto',
        )
    ])
    fig.update_layout(
        title='Distribución de Películas por Cluster',
        xaxis_title='Cluster',
        yaxis_title='Número de Películas',
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(240, 242, 246, 0.5)',
        paper_bgcolor='white',
        font=dict(size=14, color='black'),
        xaxis=dict(title_font=dict(size=16, color='black'), tickfont=dict(size=12, color='black')),
        yaxis=dict(title_font=dict(size=16, color='black'), tickfont=dict(size=12, color='black')),
    )
    return fig

# ==========================
# UI DE PELÍCULA
# ==========================
def display_movie_card(movie_data, show_similarity=False):
    col1, col2 = st.columns([1, 3])
    with col1:
        img_src = get_poster_source(movie_data)
        st.image(img_src, use_container_width=True)
    with col2:
        title = movie_data.get('title', f"Movie ID: {movie_data.get('movieId', 'Unknown')}")
        st.markdown(f"**{title}**")
        palette = px.colors.qualitative.Set3
        color = palette[int(movie_data['cluster']) % len(palette)]
        st.markdown(
            f'<span class="cluster-badge" style="background-color: {color};">Cluster {int(movie_data["cluster"])}</span>',
            unsafe_allow_html=True
        )
        info_cols = st.columns(3)
        if 'genres' in movie_data and pd.notna(movie_data['genres']):
            info_cols[0].write(f"{movie_data['genres']}")
        if 'year' in movie_data and pd.notna(movie_data['year']):
            try:
                info_cols[1].write(f"Año: {int(movie_data['year'])}")
            except Exception:
                info_cols[1].write(f"Año: {movie_data['year']}")
        if show_similarity and 'similarity' in movie_data:
            info_cols[2].write(f"Similitud: {movie_data['similarity']:.2%}")

# ==========================
# APP
# ==========================
def main():
    st.markdown('<h1 class="main-header">Sistema de Recomendación de Películas</h1>', unsafe_allow_html=True)
    st.markdown("---")

    df, feature_cols, method, chosen_file = load_clustered_data()
    if df is None:
        st.warning("""
        ⚠️ No se pudieron cargar datasets clusterizados válidos.
        Asegúrate de tener alguno de estos archivos junto a `app.py` **con columna `cluster`**:
        - `combined_features_clustered.csv`
        - `pca_features_clustered.csv`
        - `umap_features_clustered.csv`
        """)
        if chosen_file:
            st.error(f"Se intentó usar `{chosen_file}`, pero no tiene columna `cluster`.")
        return

    metadata = load_movie_metadata()
    if metadata is not None and 'movieId' in df.columns and 'movieId' in metadata.columns:
        keep_cols = ['movieId', 'title', 'genres']
        if 'poster_url' in metadata.columns:
            keep_cols.append('poster_url')
        if 'poster_path' in metadata.columns:
            keep_cols.append('poster_path')
        if 'year' in metadata.columns:
            keep_cols.append('year')
        df = df.merge(metadata[keep_cols], on='movieId', how='left')

    # ===== Sidebar extra: Diagnóstico de pósters =====
    with st.sidebar.expander("🖼️ Diagnóstico de pósters", expanded=False):
        try:
            base, files, _ = _poster_index()
            st.write(f"Carpeta de pósters: {base}")
            st.write(f"Total archivos detectados: {len(files)}")
            if 'title' in df.columns and df['title'].notna().any():
                sample_title = st.selectbox("Probar con este título", df['title'].dropna().head(2000).unique())
                row = df[df['title'] == sample_title].iloc[0].to_dict()
                path = get_poster_source(row)
                st.caption(f"Ruta/URL resuelta: {path}")
                st.image(path, caption="Vista previa", use_container_width=True)
        except Exception as e:
            st.error(f"Diagnóstico falló: {e}")

    # ===== Sidebar =====
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Movie+Recommender", use_container_width=True)
        st.markdown("---")
        st.markdown("### Estadísticas del Dataset")
        st.metric("Total de Películas", len(df))
        st.metric("Número de Clusters", int(df['cluster'].nunique()))
        st.metric("Método de Reducción", method)
        st.metric("Dimensiones de Features", len(feature_cols))
        st.metric("Fuente de datos", chosen_file if chosen_file else "—")
        st.markdown("---")
        st.info("Este sistema agrupa películas basándose en características visuales extraídas de sus pósters usando técnicas de clustering.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Búsqueda por Similitud",
        "Clusters y Representantes",
        "Visualización 2D",
        "Filtros Avanzados"
    ])

    # ===== Tab 1: Búsqueda por Similitud =====
    with tab1:
        st.markdown('<div class="sub-header">Buscar Películas Similares</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        # ------------ Opción 1: Selección desde la base ------------
        with col1:
            st.markdown("#### Opción 1: Seleccionar de la Base de Datos")

            # 1) filtro por cluster
            filter_cluster = st.checkbox("Filtrar por cluster específico")
            if filter_cluster:
                selected_cluster = st.selectbox("Selecciona un cluster", sorted(df['cluster'].unique()))
                search_df = df[df['cluster'] == selected_cluster].copy()
            else:
                search_df = df.copy()

            # 2) filtro por género (previo a similitud)
            genre_filter_on = st.checkbox("Filtrar también por género", value=False)
            selected_genres_sim = []
            if genre_filter_on and ('genres' in df.columns and df['genres'].notna().any()):
                all_genres = df['genres'].str.split('|').explode().dropna().unique()
                selected_genres_sim = st.multiselect("Géneros a incluir", options=sorted(all_genres))
                if selected_genres_sim:
                    genre_set = set(selected_genres_sim)
                    search_df = search_df[search_df['genres'].apply(lambda s: len(_genres_set(s) & genre_set) > 0)]
                    if len(search_df) == 0:
                        st.warning("No hay películas que cumplan el/los géneros seleccionados.")
                        st.stop()

            # 3) búsqueda por teclado + selección simple o múltiple
            st.markdown("##### Buscar por título (teclea para filtrar)")
            all_titles = search_df['title'].dropna().astype(str).tolist() if 'title' in search_df.columns else []
            text_query = st.text_input("Escribe parte del título", placeholder="p. ej., 'matrix', 'toy story'").strip()
            filtered_titles = [t for t in all_titles if text_query.lower() in t.lower()] if text_query else all_titles

            multi_mode = st.checkbox("Usar varias películas como consulta (promedio de features)", value=False)
            if multi_mode:
                selected_movies = st.multiselect("Elige hasta 3 películas", options=filtered_titles)
                if len(selected_movies) > 3:
                    st.info("Se tomarán solo las 3 primeras seleccionadas.")
                    selected_movies = selected_movies[:3]
                has_selection = len(selected_movies) > 0
            else:
                selected_movie = st.selectbox("Selecciona una película", filtered_titles)
                selected_movies = [selected_movie] if selected_movie else []
                has_selection = bool(selected_movie)

            # 4) Previsualización
            if has_selection:
                st.markdown("### Película(s) base")
                cols_prev = st.columns(min(3, len(selected_movies)))
                for i, title in enumerate(selected_movies):
                    mrow = df[df['title'] == title].iloc[0]
                    with cols_prev[i % len(cols_prev)]:
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        display_movie_card(mrow, show_similarity=False)
                        st.markdown('</div>', unsafe_allow_html=True)

            # 5) Buscar similares
            if st.button("Buscar Similares", key="btn_search_db"):
                if not has_selection:
                    st.warning("Selecciona al menos una película.")
                    st.stop()

                candidate_df = search_df.copy()

                base_rows = df[df['title'].isin(selected_movies)]
                query_matrix = base_rows[feature_cols].values
                query_features = query_matrix.mean(axis=0)

                # Excluir base del resultado si hay movieId
                if 'movieId' in candidate_df.columns and 'movieId' in base_rows.columns:
                    base_ids = set(base_rows['movieId'].tolist())
                    candidate_df = candidate_df[~candidate_df['movieId'].isin(base_ids)]

                with st.spinner("Buscando películas similares..."):
                    similar_movies = find_similar_movies(
                        candidate_df, feature_cols, query_features, top_k=min(50, len(candidate_df))
                    )

                similar_movies = similar_movies.head(10)
                st.success(f"Se encontraron {len(similar_movies)} películas similares")
                st.markdown("### Películas Similares")
                for _, movie in similar_movies.iterrows():
                    with st.container():
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        display_movie_card(movie, show_similarity=True)
                        st.markdown('</div>', unsafe_allow_html=True)

        # ------------ Opción 2: Imagen subida ------------
        with col2:
            st.markdown("####  Opción 2: Subir una Imagen")
            st.info("⚠️ DEMO: usa el mismo pipeline de features que tu entrenamiento para resultados reales.")
            uploaded_file = st.file_uploader(
                "Sube un póster de película",
                type=['jpg', 'jpeg', 'png'],
                help="Sube una imagen de un póster para encontrar películas visualmente similares"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", use_container_width=True)
                if st.button("Buscar por Imagen", key="btn_search_img"):
                    st.warning(
                        "Para recomendaciones reales desde imagen, extrae features con el mismo pipeline del entrenamiento "
                        "(por ejemplo, embeddings de CLIP/ViT + normalización idéntica) y compara con cosine similarity."
                    )

    # ===== Tab 2: Clusters y Representantes =====
    with tab2:
        st.markdown('<div class="sub-header">Explorar Clusters de Películas</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            fig_dist = plot_cluster_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True, key="plot_cluster_dist")

        with col2:
            st.markdown("### Estadísticas por Cluster")
            for cluster_id in sorted(df['cluster'].unique()):
                size = (df['cluster'] == cluster_id).sum()
                st.metric(f"Cluster {cluster_id}", f"{size} películas", f"{(100*size/len(df)):.1f}%")

        st.markdown("---")
        st.markdown("### Películas Representativas de Cada Cluster")
        n_representatives = st.slider("Número de representantes por cluster", min_value=3, max_value=10, value=5)
        with st.spinner("Calculando películas representativas..."):
            representatives = get_cluster_representatives(df, feature_cols, n_representatives)

        if len(representatives) == 0:
            st.info("No hay representantes para mostrar.")
        else:
            for cluster_id in sorted(representatives['cluster'].unique()):
                with st.expander(f"🎬 Cluster {cluster_id}", expanded=True):
                    cluster_reps = representatives[representatives['cluster'] == cluster_id]
                    cols = st.columns(min(3, len(cluster_reps)))
                    for idx, (_, movie) in enumerate(cluster_reps.iterrows()):
                        with cols[idx % 3]:
                            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                            display_movie_card(movie, show_similarity=False)
                            if 'distance_to_centroid' in movie and pd.notna(movie['distance_to_centroid']):
                                st.caption(f"Distancia al centroide: {movie['distance_to_centroid']:.3f}")
                            st.markdown('</div>', unsafe_allow_html=True)

    # ===== Tab 3: Visualización 2D =====
    with tab3:
        st.markdown('<div class="sub-header">Visualización del Espacio de Características</div>', unsafe_allow_html=True)
        st.info("Esta visualización muestra la distribución de películas en un espacio 2D usando columnas reales (elige X e Y).")

        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown("### Opciones de Visualización")
            color_option = st.radio("Colorear por:", ["Cluster", "Año (si disponible)", "Género (si disponible)"])
            show_all = st.checkbox("Mostrar todas las películas", value=True)
            if not show_all:
                sample_size = st.slider("Número de películas a mostrar", min_value=100, max_value=min(5000, len(df)), value=min(1000, len(df)))
                plot_df = df.sample(n=sample_size, random_state=42)
            else:
                plot_df = df

        with col1:
            # Color
            if color_option == "Cluster":
                color_by = 'cluster'
            elif color_option == "Año (si disponible)" and 'year' in df.columns:
                color_by = 'year'
            elif color_option == "Género (si disponible)" and 'genres' in df.columns:
                plot_df = plot_df.copy()
                plot_df['main_genre'] = plot_df['genres'].str.split('|').str[0]
                color_by = 'main_genre'
            else:
                color_by = 'cluster'

            # Selectores de ejes
            numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
            # Sugerencias por defecto
            default_x = feature_cols[0] if feature_cols else (numeric_cols[0] if numeric_cols else None)
            default_y = feature_cols[1] if len(feature_cols) > 1 else (numeric_cols[1] if len(numeric_cols) > 1 else None)

            st.markdown("#### Elegir columnas para los ejes")
            cxa, cya = st.columns(2)
            with cxa:
                x_col = st.selectbox("Eje X", options=numeric_cols,
                                     index=max(0, numeric_cols.index(default_x)) if default_x in numeric_cols else 0,
                                     key="ax_x")
            with cya:
                y_col = st.selectbox("Eje Y", options=numeric_cols,
                                     index=max(1, numeric_cols.index(default_y)) if default_y in numeric_cols else (1 if len(numeric_cols) > 1 else 0),
                                     key="ax_y")

            fig_scatter = plot_scatter_by_columns(plot_df, x_col, y_col, color_by, title_suffix=method)
            st.plotly_chart(fig_scatter, use_container_width=True, key="plot_scatter_main")

        st.markdown("### Análisis Estadístico")
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Películas Visualizadas", len(plot_df))
        with stat_cols[1]:
            st.metric("Clusters Únicos", plot_df['cluster'].nunique())
        with stat_cols[2]:
            avg_cluster_size = len(plot_df) / max(1, plot_df['cluster'].nunique())
            st.metric("Tamaño Promedio de Cluster", f"{avg_cluster_size:.1f}")
        with stat_cols[3]:
            if 'genres' in plot_df.columns:
                unique_genres = plot_df['genres'].str.split('|').explode().nunique()
                st.metric("Géneros Únicos", unique_genres)

    # ===== Tab 4: Filtros Avanzados =====
    with tab4:
        st.markdown('<div class="sub-header">Búsqueda y Filtros Avanzados</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Filtrar por Cluster")
            options = sorted(df['cluster'].unique())
            selected_clusters = st.multiselect("Selecciona clusters", options=options, default=options)

        with col2:
            if 'year' in df.columns and df['year'].notna().any():
                st.markdown("#### Filtrar por Año")
                min_year = int(df['year'].min()) if pd.notna(df['year'].min()) else 1900
                max_year = int(df['year'].max()) if pd.notna(df['year'].max()) else 2025
                year_range = st.slider("Rango de años", min_value=min_year, max_value=max_year, value=(min_year, max_year))
            else:
                year_range = None

        with col3:
            if 'genres' in df.columns and df['genres'].notna().any():
                st.markdown("#### Filtrar por Género")
                all_genres = df['genres'].str.split('|').explode().dropna().unique()
                selected_genres = st.multiselect("Selecciona géneros", options=sorted(all_genres))
            else:
                selected_genres = []

        filtered_df = df[df['cluster'].isin(selected_clusters)].copy()
        if year_range and 'year' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
        if selected_genres and 'genres' in filtered_df.columns:
            gset = set(selected_genres)
            filtered_df = filtered_df[filtered_df['genres'].apply(lambda s: len(_genres_set(s) & gset) > 0)]

        st.markdown("---")
        st.markdown(f"### Resultados: {len(filtered_df)} películas encontradas")

        if len(filtered_df) > 0:
            view_option = st.radio("Vista:", ["Lista", "Tabla", "Gráfico"], horizontal=True)
            if view_option == "Lista":
                n_cols = 3
                rows = (len(filtered_df) + n_cols - 1) // n_cols
                max_rows = min(rows, 10)
                for row in range(max_rows):
                    cols = st.columns(n_cols)
                    for col_idx in range(n_cols):
                        idx = row * n_cols + col_idx
                        if idx < len(filtered_df):
                            with cols[col_idx]:
                                movie = filtered_df.iloc[idx]
                                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                                display_movie_card(movie)
                                st.markdown('</div>', unsafe_allow_html=True)

                if len(filtered_df) > (n_cols * max_rows):
                    st.info(f"Mostrando las primeras {n_cols * max_rows} de {len(filtered_df)} películas")

            elif view_option == "Tabla":
                display_cols = ['cluster']
                for c in ['title', 'genres', 'year', 'movieId']:
                    if c in filtered_df.columns:
                        display_cols.append(c)
                st.dataframe(filtered_df[display_cols].head(200), use_container_width=True, height=420)
                csv = filtered_df.to_csv(index=False)
                st.download_button("Descargar resultados (CSV)", data=csv, file_name="peliculas_filtradas.csv", mime="text/csv")

            else:
                # Gráfico con ejes elegibles también en esta pestaña
                st.markdown("#### Gráfico de dispersión de los filtrados")
                numeric_cols_f = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols_f) >= 2:
                    cx, cy = st.columns(2)
                    with cx:
                        x_f = st.selectbox("Eje X (filtrados)", options=numeric_cols_f, key="flt_x")
                    with cy:
                        # elegir distinta de x_f si se puede
                        default_y_idx = 1 if numeric_cols_f[0] == x_f and len(numeric_cols_f) > 1 else 0
                        y_f = st.selectbox("Eje Y (filtrados)", options=[c for c in numeric_cols_f if c != x_f] or numeric_cols_f,
                                           index=default_y_idx, key="flt_y")
                    fig = plot_scatter_by_columns(filtered_df, x_f, y_f, color_by='cluster', title_suffix=method)
                    st.plotly_chart(fig, use_container_width=True, key="plot_filtered_scatter")
                else:
                    st.info("No hay suficientes columnas numéricas para graficar.")
        else:
            st.warning("No se encontraron películas con los filtros seleccionados")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Sistema de Recomendación de Películas basado en Clustering Visual</p>
        <p>Desarrollado con Streamlit, Scikit-learn y Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
