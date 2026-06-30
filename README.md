# Análisis de Texto Bíblico - Proyecto de Laboratorio de PLN

Un proyecto integral de Procesamiento de Lenguaje Natural (PLN) para analizar textos bíblicos (Antiguo y Nuevo Testamento) utilizando Python. Este proyecto realiza análisis de sentimientos, clasificación de textos, búsqueda semántica y visualizaciones avanzadas de PLN.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Módulos](#módulos)
- [Requisitos](#requisitos)
- [Salida](#salida)

## 🎯 Descripción General

Este es el **Laboratorio 2** de un curso de Programación Científica. El proyecto analiza el corpus completo de la Biblia (English Standard Version) utilizando técnicas modernas de PLN, incluyendo análisis de sentimientos con VADER, vectorización TF-IDF, clasificación con Naive Bayes y visualización mediante PCA.

## ✨ Características

- **Análisis de Sentimientos**: Analiza el tono emocional de todos los versículos, capítulos y libros.
- **Clasificación de Texto**: Predice a qué libro pertenece un versículo utilizando Naive Bayes.
- **Búsqueda Semántica**: Encuentra versículos similares a una frase dada mediante TF-IDF y similitud del coseno.
- **Visualización de Texto**: Genera nubes de palabras, matrices de confusión y gráficos de reducción de dimensionalidad.
- **Modelos N-gram**: Genera texto basado en patrones de frecuencia de palabras.
- **Estadísticas del Texto**: Calcula frecuencias de palabras y métricas a nivel de documento.
- **Análisis Multinivel**: Procesa los datos a nivel de versículo, capítulo y libro.

## 📁 Estructura del Proyecto

```text
.
├── README.md                 # Este archivo
├── UMLLAB2.drawio            # Diagrama UML de la arquitectura del proyecto
├── archive/                  # Archivos de datos (no incluidos en el repositorio)
│   ├── key_english.csv       # Metadatos de los libros de la Biblia
│   └── t_bbe.csv             # Corpus de versículos bíblicos
├── src/                      # Módulos principales
│   ├── book.py               # Definición de la clase Book
│   ├── chapter.py            # Definición de la clase Chapter
│   ├── genre.py              # Definición de la clase Genre
│   ├── loader.py             # Cargador de datos CSV
│   ├── testament.py          # Clase Testament (Antiguo/Nuevo)
│   ├── verse.py              # Definición de la clase Verse
│   └── nlp/                  # Módulos de procesamiento de PLN
│       ├── classifier.py     # Clasificador Naive Bayes
│       ├── cleaner.py        # Limpieza y preprocesamiento de texto
│       ├── generator.py      # Generación de texto mediante N-gram
│       ├── metrics.py        # Métricas de similitud/distancia
│       ├── pipeline.py       # Flujo principal de procesamiento
│       ├── sentiment.py      # Análisis de sentimientos con VADER
│       ├── stopwords.py      # Lista de stopwords
│       └── tfidf.py          # Vectorización TF-IDF
├── buscador.py               # Interfaz de búsqueda semántica
├── clasificador.py           # Entrenamiento y evaluación del clasificador
├── sentimiento.py            # Ejecución del análisis de sentimientos
└── visualizacion.py          # Generador de visualizaciones
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- Administrador de paquetes pip

### Configuración

1. **Clonar o descargar el repositorio**

   ```bash
   cd "Laboratorio 2"
   ```

2. **Crear un entorno virtual** (recomendado)

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar las dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Preparar los archivos de datos**

   Cree un directorio `archive/` en la raíz del proyecto y agregue:

   - `key_english.csv` - Metadatos de los libros de la Biblia (id, nombre, testamento, género).
   - `t_bbe.csv` - Versículos de la Biblia (id, id_libro, capítulo, versículo, texto).

## 📖 Uso

### 1. Análisis de Sentimientos

Analiza el tono emocional de los versículos a través de todos los libros y capítulos:

```bash
python sentimiento.py
```

**Salida:**

- Consola: Los 5 libros y capítulos más positivos y más negativos.
- `evolucion_sentimiento_libros.png` - Gráfico de barras con el sentimiento promedio por libro.
- `ejemplo_libro.png` - Evolución del sentimiento por capítulos para un libro seleccionado.
- Visualizaciones adicionales relacionadas con el análisis de sentimientos.

### 2. Clasificación de Texto

Entrena y evalúa un clasificador Naive Bayes para predecir a qué libro pertenece un versículo:

```bash
python clasificador.py
```

**Salida:**

- Consola: Resultados de las predicciones y métricas de precisión.
- `matriz_confusion_nb2.png` - Matriz de confusión que muestra el rendimiento del clasificador.

**Ejemplo de prueba:**

El script incluye una prueba predefinida con palabras de Éxodo 6:3 para demostrar las predicciones.

### 3. Búsqueda Semántica

Encuentra versículos semánticamente similares a una frase dada:

```bash
python buscador.py
```

**Menú interactivo:**

```text
1) Buscar por frase (TF-IDF + similitud del coseno)
2) Buscar por versículo
3) Salir
```

**Proceso:**

1. Ingrese una frase o consulta de versículo.
2. El sistema devuelve los resultados ordenados según su similitud.
3. Se muestra: testamento, libro, capítulo, versículo, texto y puntaje de similitud.

### 4. Visualizaciones

Genera un conjunto completo de visualizaciones del corpus de la Biblia:

```bash
python visualizacion.py
```

**Genera:**

- Cantidad de versículos por libro (gráfico de barras).
- Nubes de palabras para el Antiguo Testamento y el Nuevo Testamento.
- Visualización PCA de los vectores de los versículos.
- Mapa de calor de la matriz de similitud entre libros.
- Análisis de correlación de los puntajes de sentimiento.

## 📚 Módulos

### Clases Principales (`src/`)

- **Verse**: Representa un versículo individual con su ID, contenido y frecuencias de palabras.
- **Chapter**: Contenedor de los versículos pertenecientes a un capítulo.
- **Book**: Contenedor de capítulos con metadatos (nombre, género y testamento).
- **Testament**: Contenedor de todos los libros del Antiguo o Nuevo Testamento.

### Módulos de PLN (`src/nlp/`)

| Módulo | Propósito |
|---------|-----------|
| `pipeline.py` | Flujo principal de procesamiento para la tokenización y el conteo de frecuencias. |
| `cleaner.py` | Preprocesamiento del texto: conversión a minúsculas, eliminación de puntuación y filtrado de stopwords. |
| `tfidf.py` | Vectorización TF-IDF y cálculo de la similitud del coseno. |
| `sentiment.py` | Análisis de sentimientos mediante VADER con estadísticas agregadas. |
| `classifier.py` | Clasificación de texto utilizando Naive Bayes. |
| `generator.py` | Modelo N-gram para la generación de texto. |
| `metrics.py` | Cálculo de métricas de distancia y similitud. |
| `stopwords.py` | Lista de stopwords en inglés utilizadas para el filtrado. |

## 📦 Requisitos

- **pandas** - Manipulación y análisis de datos.
- **numpy** - Computación numérica.
- **scikit-learn** - Aprendizaje automático (PCA).
- **matplotlib** - Creación de gráficos y visualizaciones.
- **seaborn** - Visualización estadística de datos.
- **wordcloud** - Generación de nubes de palabras.
- **nltk** - Natural Language Toolkit (análisis de sentimientos con VADER).

Instale todas las dependencias con:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk
```

## 📊 Salida

Todos los scripts generan:

### 1. Salida por consola

- Estadísticas y resultados del análisis.
- Rankings de los cinco mejores resultados.
- Métricas de rendimiento.

### 2. Archivos de visualización (formato PNG, 300 DPI)

- `evolucion_sentimiento_libros.png`
- `matriz_confusion_nb2.png`
- `palabra_nubes_*.png`
- `pca_visualization_*.png`
- Visualizaciones adicionales del análisis.

### 3. Datos del análisis

- DataFrames impresos.
- Resultados ordenados.
- Reportes de clasificación.

## 🔬 Detalles Técnicos

### Análisis de Sentimientos

- **Método**: VADER (Valence Aware Dictionary and sEntiment Reasoner).
- **Rango de salida**: -1 (más negativo) a +1 (más positivo).
- **Nivel de análisis**: Agregación a nivel de versículo, capítulo y libro.

### Clasificación de Texto

- **Algoritmo**: Naive Bayes Multinomial.
- **Características**: Vectores de frecuencia de tokens (codificación one-hot después de la vectorización).
- **Clases**: Los 66 libros de la Biblia.
- **Métrica**: Precisión (*Accuracy*) y matriz de confusión.

### Búsqueda Semántica

- **Vectorización**: TF-IDF.
- **Similitud**: Similitud del coseno entre vectores.
- **Preprocesamiento**: Stopwords personalizadas en inglés, conversión a minúsculas y eliminación de signos de puntuación.

### Reducción de Dimensionalidad

- **Método**: Análisis de Componentes Principales (PCA).
- **Componentes**: 2 dimensiones para la visualización.
- **Propósito**: Visualizar las relaciones semánticas entre versículos y libros.

## 📝 Notas

- Todo el procesamiento del texto se realiza en **inglés**.
- El proyecto utiliza la versión **BBE (Bible in Basic English)** de la Biblia.
- La arquitectura UML está disponible en `LAB2UML.png` (editable con Draw.io).
- El informe del proyecto está disponible en `LAB2_PC.pdf`

