from src.testament import Testament
from src.loader import final_load
from src.nlp.pipeline import process_bible
from src.nlp.classifier import train_and_evaluate_classifier
from src.nlp.generator import build_ngram_model, generate_text
from src.nlp.sentiment import analyze_sentiment
import matplotlib.pyplot as plt
import seaborn as sns

NT = Testament(1, "New Testament", [])
OT = Testament(0, "Old Testament", [])

books = {}
final_load(books, NT, OT)

ot_global_frequencies = process_bible(OT)
nt_global_frequencies = process_bible(NT)

print("--- TESTING CLASSIFIER ---")
model, vectorizer, accuracy, conf_matrix = train_and_evaluate_classifier(books)

# I let myself be seen by Abraham, Isaac, and Jacob, as God, the Ruler of all; but they had no knowledge of my name Yahweh.
custom_verse_tokens = ["Abraham", "Isaac", "Jacob",
                       "God", "Ruler", "knowledge", "Yahweh"]
# Jesus said these things; then, lifting his eyes to heaven, he said, Father, the time has now come; give glory to your Son, so that the Son may give glory to you
# custom_verse_tokens2 = ['Jesus', 'said', 'things', 'lifting', 'eyes', 'heaven', 'said',
#                       'Father', 'time', 'come', 'give', 'glory', 'Son', 'Son', 'may', 'give', 'glory']

transformed_verse = vectorizer.transform([custom_verse_tokens])
predicted_book = model.predict(transformed_verse)[0]
print("Input tokens:", custom_verse_tokens)
print("Predicted book:", predicted_book)

lista_clases = model.classes_.tolist()
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(conf_matrix, cmap="Blues", cbar=False,
            xticklabels=lista_clases, yticklabels=lista_clases)

plt.title("Matriz de Confusión - Clasificador de Versículos",
          fontsize=16, fontweight="bold")
plt.xlabel("Libro Predicho", fontsize=12)
plt.ylabel("Libro Real", fontsize=12)
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8)

plt.tight_layout()
plt.savefig("matriz_confusion_nb2.png", dpi=300)
plt.close()


print("--- TESTING GENERATOR ---")
start_word = "love"

print("1-Gram (Unigram):")
model_1 = build_ngram_model(books, n=1)
generated_1 = generate_text(model_1, n=1, start_word=start_word, max_length=20)
print(generated_1)
print("-" * 20)

print("2-Gram (Bigram):")
model_2 = build_ngram_model(books, n=2)
generated_2 = generate_text(model_2, n=2, start_word=start_word, max_length=20)
print(generated_2)
print("-" * 20)

print("3-Gram (Trigram):")
model_3 = build_ngram_model(books, n=3)
generated_3 = generate_text(model_3, n=3, start_word=start_word, max_length=20)
print(generated_3)
print("-" * 20)

print("4-Gram (Quadgram):")
model_4 = build_ngram_model(books, n=4)
generated_4 = generate_text(model_4, n=4, start_word=start_word, max_length=20)
print(generated_4)
print("-" * 30)


print("--- TESTING SENTIMENT ANALYSIS ---")

df_sentiments, book_sentiments, chapter_sentiments = analyze_sentiment(books)

print("\nMuestra de Sentimiento por Versículo (Primeros 5):")
print(df_sentiments[["book", "chapter", "verse", "score"]].head())
print("-" * 20)

print("\nMuestra de Sentimiento Promedio por Libro:")
print(book_sentiments.head())
print("-" * 20)

if not book_sentiments.empty:
    libro_mas_positivo = book_sentiments.loc[book_sentiments['score'].idxmax()]
    libro_mas_negativo = book_sentiments.loc[book_sentiments['score'].idxmin()]

    print("\nLibro más positivo:")
    print(
        f"{libro_mas_positivo['book']} (Score: {libro_mas_positivo['score']:.4f})")

    print("\nLibro más negativo:")
    print(
        f"{libro_mas_negativo['book']} (Score: {libro_mas_negativo['score']:.4f})")
print("-" * 30)
