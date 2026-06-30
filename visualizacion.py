from src.loader import final_load
from src.nlp.pipeline import process_bible
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.testament import Testament
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
from src.nlp.metrics import generate_similarity_matrix
from src.nlp.sentiment import analyze_sentiment

from src.nlp.tfidf import generate_tfidf_vectors

NT = Testament(1, "New Testament", [])
OT = Testament(0, "Old Testament", [])

books = {}
final_load(books, NT, OT)
ot_global_frequencies = process_bible(OT)
nt_global_frequencies = process_bible(NT)

# 1. CANTIDAD DE VERSICULOS POR LIBRO
versebook = {}
for book in books:
    for chapter in books[book].chapters:
        for verse in books[book].chapters[chapter].verses:
            if books[book].name not in versebook:
                versebook[books[book].name] = 1
            else:
                versebook[books[book].name] += 1

df_versebook = pd.DataFrame(
    list(versebook.items()), columns=["Libro", "Versiculos"]
)
fig1, ax1 = plt.subplots(figsize=(15, 6))
ax1.bar(
    df_versebook["Libro"],
    df_versebook["Versiculos"],
    color="#2b5c8f",
    edgecolor="black",
)
ax1.set_title(
    "Cantidad de Versículos por Libro", fontsize=16, fontweight="bold", pad=15
)
ax1.set_xlabel("Libros", fontsize=12, labelpad=10)
ax1.set_ylabel("Número de Versículos", fontsize=12, labelpad=10)

ax1.set_xticks(range(len(df_versebook["Libro"])))
ax1.set_xticklabels(df_versebook["Libro"], rotation=90, fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("versiculo_x_libro.png")
plt.close()


# 2. HEATMAP DE SIMILITUD (A nivel de Libro)
heatbook = {}
for book in books:
    for chapter in books[book].chapters:
        for verse in books[book].chapters[chapter].verses:
            tokens = books[book].chapters[chapter].verses[verse].tokens
            if books[book].name not in heatbook:
                heatbook[books[book].name] = list(tokens)
            else:
                heatbook[books[book].name].extend(tokens)

books_names = list(heatbook.keys())
corpus_libros = list(heatbook.values())

vectores_libros, vocab_libros, idf = generate_tfidf_vectors(corpus_libros)

matriz_similitud = generate_similarity_matrix(vectores_libros)

fig2, ax2 = plt.subplots(figsize=(12, 12))
im = ax2.imshow(matriz_similitud, cmap="YlGnBu", aspect="auto")
num_libros = len(books_names)
ax2.set_xticks(range(num_libros))
ax2.set_yticks(range(num_libros))

ax2.set_xticklabels(books_names, rotation=90, fontsize=8)
ax2.set_yticklabels(books_names, fontsize=8)

cbar = fig2.colorbar(im, ax=ax2, shrink=0.8)
cbar.set_label("Nivel de Similitud (Coseno)", fontsize=10)

ax2.set_title(
    "Matriz de Similitud entre Libros ($NxN$)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

plt.tight_layout()
plt.savefig("heatmap_libroxlibro.png")
plt.close()


# 3. NUBES DE PALABRAS
# ANTIGUO TESTAMENTO
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate_from_frequencies(ot_global_frequencies)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("nube_palabras_ot.png")
plt.close()

# NUEVO TESTAMENTO
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate_from_frequencies(nt_global_frequencies)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("nube_palabras_nt.png")
plt.close()

# NUEVO Y ANTIGUO TESTAMENTO JUNTOS
bible_frequencies = Counter(nt_global_frequencies) + \
    Counter(ot_global_frequencies)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis"
).generate_from_frequencies(bible_frequencies)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("nube_palabras_bible.png")
plt.close()


# 4. DISTRIBUCION DE LONGITUD DE VERSICULOS
longitudes = []

for book in books:
    for chapter in books[book].chapters:
        for verse in books[book].chapters[chapter].verses:
            tokens_limpios = books[book].chapters[chapter].verses[verse].tokens
            longitudes.append(len(tokens_limpios))

df_longitudes = pd.DataFrame(longitudes, columns=["Longitud"])

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.hist(df_longitudes["Longitud"], bins=40,
         color="#4a90e2", edgecolor="black", alpha=0.8)

ax3.set_title("Distribución de la Longitud de los Versículos",
              fontsize=16, fontweight="bold", pad=15)
ax3.set_xlabel("Cantidad de palabras por versículo (sin stopwords)",
               fontsize=12, labelpad=10)
ax3.set_ylabel("Frecuencia (Número de versículos)", fontsize=12, labelpad=10)

ax3.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("distribucion_longitud_versiculos.png", dpi=300)
plt.close(fig3)


# 5. PCA
corpus_versiculos = []
categorias_testamento = []

for book in books:
    for chapter in books[book].chapters:
        for verse in books[book].chapters[chapter].verses:
            tokens = books[book].chapters[chapter].verses[verse].tokens

            if tokens:
                corpus_versiculos.append(tokens)

                if books[book].testament == "0":
                    categorias_testamento.append("Antiguo Testamento")
                else:
                    categorias_testamento.append("Nuevo Testamento")

vectores_versiculos, vocab_versiculos, idf = generate_tfidf_vectors(
    corpus_versiculos)

pca = PCA(n_components=2)
matriz_densa = np.array(vectores_versiculos)
coordenadas_2d = pca.fit_transform(matriz_densa)

fig4, ax4 = plt.subplots(figsize=(11, 8))

coordenadas_2d = np.array(coordenadas_2d)
categorias_testamento = np.array(categorias_testamento)

colores_dict = {
    "Antiguo Testamento": "#d95f02",
    "Nuevo Testamento": "#7570b3"
}

for testamento in ["Antiguo Testamento", "Nuevo Testamento"]:
    indices = (categorias_testamento == testamento)

    ax4.scatter(
        coordenadas_2d[indices, 0],
        coordenadas_2d[indices, 1],
        label=testamento,
        color=colores_dict[testamento],
        alpha=0.5,
        edgecolors='none',
        s=10
    )

ax4.set_title("Reducción de Dimensionalidad de Versículos (PCA)",
              fontsize=16, fontweight="bold", pad=15)
ax4.set_xlabel("Primera Componente Principal ($PC_1$)",
               fontsize=12, labelpad=10)
ax4.set_ylabel("Segunda Componente Principal ($PC_2$)",
               fontsize=12, labelpad=10)

ax4.legend(loc="upper right", fontsize=11, frameon=True, shadow=True)
ax4.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("pca_versiculos_testamentos.png", dpi=300)
plt.close(fig4)

# ANALISIS DE SENTIMIENTO#
df_sentiments, book_sentiments, chapter_sentiments = analyze_sentiment(books)

fig5, ax5 = plt.subplots(figsize=(15, 6))
ax5.bar(
    book_sentiments["book"],
    book_sentiments["score"],
    color="#2b5c8f",
    edgecolor="black"
)
ax5.set_title("Average Sentiment Score per Book",
              fontsize=16, fontweight="bold", pad=15)
ax5.set_xlabel("Books", fontsize=12, labelpad=10)
ax5.set_ylabel("Sentiment Score (Compound)", fontsize=12, labelpad=10)
ax5.set_xticks(range(len(book_sentiments["book"])))
ax5.set_xticklabels(book_sentiments["book"], rotation=90, fontsize=9)
ax5.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("sentimiento_x_libro.png")
plt.close(fig5)

max_verse = df_sentiments.loc[df_sentiments["score"].idxmax()]
min_verse = df_sentiments.loc[df_sentiments["score"].idxmin()]

print("Most Positive Verse:")
print(max_verse["book"], max_verse["chapter"], max_verse["verse"])
print(max_verse["content"])
print("Score:", max_verse["score"])
print("-" * 20)
print("Most Negative Verse:")
print(min_verse["book"], min_verse["chapter"], min_verse["verse"])
print(min_verse["content"])
print("Score:", min_verse["score"])
