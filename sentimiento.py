import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.nlp.sentiment import analyze_sentiment
from src.testament import Testament
from src.loader import final_load


def test_sentiment_analysis(books):
    print("--- INICIANDO ANÁLISIS DE SENTIMIENTO ---")

    df_versiculos, df_libros, df_capitulos = analyze_sentiment(books)

    libros_ordenados = df_libros.sort_values(by="score", ascending=False)
    top_5_positivos_libros = libros_ordenados.head(5)
    top_5_negativos_libros = libros_ordenados.tail(5)

    print("\n[EXTREMOS] Top 5 Libros Más POSITIVOS:")
    print(top_5_positivos_libros.to_string(index=False))

    print("\n[EXTREMOS] Top 5 Libros Más NEGATIVOS:")
    print(top_5_negativos_libros.to_string(index=False))

    capitulos_ordenados = df_capitulos.sort_values(by="score", ascending=False)

    print("\n[EXTREMOS] Top 5 Capítulos Más POSITIVOS:")
    print(capitulos_ordenados.head(5).to_string(index=False))

    print("\n[EXTREMOS] Top 5 Capítulos Más NEGATIVOS:")
    print(capitulos_ordenados.tail(5).to_string(index=False))

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(15, 6))
    sns.barplot(x="book", y="score", data=df_libros, palette="vlag")
    plt.axhline(0, color='black', linewidth=1.2)  # Línea neutral
    plt.title("Evolución del Sentimiento Promedio por Libro Bíblico",
              fontsize=16, fontweight="bold")
    plt.xlabel("Libro", fontsize=12)
    plt.ylabel("Puntaje Compound Promedio (VADER)", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig("evolucion_sentimiento_libros.png", dpi=300)
    plt.close()

    libro_ejemplo = "Psalms"
    df_ejemplo = df_capitulos[df_capitulos["book"] == libro_ejemplo].copy()

    df_ejemplo["chapter"] = pd.to_numeric(df_ejemplo["chapter"])
    df_ejemplo = df_ejemplo.sort_values("chapter")

    if not df_ejemplo.empty:
        plt.figure(figsize=(12, 5))
        sns.lineplot(x="chapter", y="score", data=df_ejemplo,
                     marker="o", color="royalblue")
        plt.axhline(0, color='red', linestyle='--',
                    linewidth=1)
        plt.title(
            f"Evolución del Sentimiento por Capítulo en {libro_ejemplo}", fontsize=14, fontweight="bold")
        plt.xlabel("Número de Capítulo", fontsize=12)
        plt.ylabel("Sentimiento Promedio", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            f"evolucion_sentimiento_{libro_ejemplo.lower()}.png", dpi=300)
        plt.close()

    print("\n--- GRÁFICOS GENERADOS Y GUARDADOS CON ÉXITO ---")


NT = Testament(1, "New Testament", [])
OT = Testament(0, "Old Testament", [])

books = {}
final_load(books, NT, OT)
test_sentiment_analysis(books)
