from src.testament import Testament
from src.nlp.pipeline import process_bible
from src.loader import final_load
from src.nlp.tfidf import process_bible_objects_to_tfidf, vectorize_new_phrase
from src.nlp.cleaner import clean_word
from src.nlp.metrics import cosine_similarity_manual

NT = Testament(1, "New Testament", [])
OT = Testament(0, "Old Testament", [])

books = {}
final_load(books, NT, OT)
ot_global_frequencies = process_bible(OT)
nt_global_frequencies = process_bible(NT)

vector_bible, vocab, metadata, idf = process_bible_objects_to_tfidf([OT, NT])


def buscador():
    option = int(input("1)Frase \n2)Versiculo\n3)Salir"))
    while option:
        if option == 1:
            rawtext = input("Ingresar Frase: ")
            clean = clean_word(rawtext)
            word_vector = vectorize_new_phrase(
                clean, vocab, idf)

            resultados = []

            for i, vector_verso in enumerate(vector_bible):
                similitud = cosine_similarity_manual(word_vector, vector_verso)
                meta = metadata[i]

                book_id = None
                for bid, bobj in books.items():
                    if bobj.name == meta["book"]:
                        book_id = bid
                        break

                verso_obj = books[book_id].chapters[meta["chapter"]
                                                    ].verses[meta["verse"]]
                texto_original = verso_obj.content

                resultados.append({
                    "testamento": meta["testament"],
                    "libro": meta["book"],
                    "capitulo": meta["chapter"],
                    "verso": meta["verse"],
                    "texto": texto_original,
                    "similitud": similitud
                })

            resultados.sort(key=lambda x: x["similitud"], reverse=True)
            ranking = resultados[:5]

            for rank, res in enumerate(ranking, 1):
                print(f"Ranking: {rank}")
                print(f"Testamento: {res['testamento']}")
                print(f"Libro: {res['libro']}")
                print(f"Capitulo: {res['capitulo']}")
                print(f"Verso: {res['verso']}")
                print(f"Texto Original: {res['texto']}")
                print(f"Similitud: {res['similitud']:.4f}")
                print("-" * 20)

        if option == 2:
            optbook = input("Libro(0-66)")
            book = books[optbook]
            optchap = input("Capitulo")
            chapter = book.chapters[optchap]

            for verse in chapter.verses:
                print(verse, chapter.verses[verse].content)

            optverse = input("Seleccionar verso")
            cleanverse = chapter.verses[optverse].tokens
            word_vector = vectorize_new_phrase(cleanverse, vocab, idf)

            resultados = []

            for i, vector_verso in enumerate(vector_bible):
                meta = metadata[i]

                if (meta["book"] == book.name and
                    str(meta["chapter"]) == str(optchap) and
                        str(meta["verse"]) == str(optverse)):
                    continue

                similitud = cosine_similarity_manual(word_vector, vector_verso)

                book_id = None
                for bid, bobj in books.items():
                    if bobj.name == meta["book"]:
                        book_id = bid
                        break

                verso_obj = books[book_id].chapters[meta["chapter"]
                                                    ].verses[meta["verse"]]
                texto_original = verso_obj.content

                resultados.append({
                    "testamento": meta["testament"],
                    "libro": meta["book"],
                    "capitulo": meta["chapter"],
                    "verso": meta["verse"],
                    "texto": texto_original,
                    "similitud": similitud
                })

            resultados.sort(key=lambda x: x["similitud"], reverse=True)
            ranking = resultados[:5]

            for rank, res in enumerate(ranking, 1):
                print(f"Ranking: {rank}")
                print(f"Testamento: {res['testamento']}")
                print(f"Libro: {res['libro']}")
                print(f"Capitulo: {res['capitulo']}")
                print(f"Verso: {res['verso']}")
                print(f"Texto Original: {res['texto']}")
                print(f"Similitud: {res['similitud']:.4f}")
                print("-" * 20)

        if option == 3:
            break
        option = int(input("1)Frase \n2)Versiculo\n3)Salir"))


buscador()
