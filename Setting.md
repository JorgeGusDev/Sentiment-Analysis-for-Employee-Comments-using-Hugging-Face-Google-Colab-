# 📊 Sentiment Analysis for Employee Comments using Hugging Face (Google Colab)

Este repositorio contiene un flujo completo en **Python + Google Colab** para realizar **análisis de sentimiento en comentarios de empleados**, utilizando modelos preentrenados de **Hugging Face Transformers**.  

El objetivo es apoyar proyectos de **HR Analytics, NLP y SNA**, permitiendo analizar percepciones organizacionales y generar indicadores agregados por persona.

---

## 1. Hello World & Input Example

```python
print("Hello World")
print("¡Hola! Bienvenido a Python en Colab 🚀")

nombre = input("¿Cómo te llamas? ")
print(f"Mucho gusto, {nombre}!")
```

## 2. Basic Sentiment Analysis (English)

```python
# Instala la librería solo la primera vez
!pip install transformers

# Traemos el modelo de sentimiento
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")

# Comentarios de ejemplo de empleados
comments = [
    "My manager supports me.",
    "There is too much work.",
    "I feel valued.",
    "Communication is poor.",
    "Leadership does not listen."
]

# Analizamos los comentarios
for comment in comments:
    result = sentiment(comment)[0]  # resultado del comentario
    print(comment, "->", result["label"], round(result["score"], 2))

```


## 3. Multilingual Sentiment Analysis (English & Spanish)
```python
# Instalar librería (solo la primera vez)
!pip install transformers

# Importar pipeline de Hugging Face
from transformers import pipeline

# Crear el pipeline con el modelo multilingüe
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Comentarios de ejemplo (inglés y español)
comments = [
    "Mi jefe me apoya mucho.",
    "There is too much work.",
    "Me siento valorado.",
    "Communication is poor.",
    "El liderazgo no escucha."
]

# Analizamos los comentarios
for comment in comments:
    result = sentiment(comment)[0]
    print(comment, "->", result["label"])
```
## 4. Upload Dataset in Google Colab
```python
from google.colab import files

uploaded = files.upload()  # Esto abrirá un selector para subir archivos
```

## 5. Sentiment Analysis from Excel Dataset (HR Dataset)
```python
# 1️⃣ Instalar librerías
!pip install transformers pandas openpyxl

# 2️⃣ Importar librerías
from transformers import pipeline
import pandas as pd

# 3️⃣ Leer el archivo directamente desde la sesión de Colab
df = pd.read_excel('/content/2. DataSet_Chat.xlsx', header=None, names=['Raw'])

# 4️⃣ Separar número y comentario
df['Persona'] = df['Raw'].apply(lambda x: int(str(x).split(':')[0]))
df['Comentario'] = df['Raw'].apply(lambda x: ':'.join(str(x).split(':')[1:]).strip())

# 5️⃣ Crear pipeline de análisis de sentimiento
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 6️⃣ Función para convertir estrellas a número
def stars_to_score(label):
    return int(label[0])  # '3 stars' -> 3

# 7️⃣ Analizar comentarios línea por línea
df['Score'] = df['Comentario'].apply(lambda x: stars_to_score(sentiment(x)[0]['label']))

# 8️⃣ Convertir score a etiquetas HR-friendly línea por línea
def score_to_label(score):
    if score <= 2.7:
        return "NEGATIVE"
    elif score <= 3.5:
        return "NEUTRAL"
    else:
        return "POSITIVE"

df['Sentiment_Label'] = df['Score'].apply(score_to_label)

# 9️⃣ Promedio por persona
avg_sentiment = df.groupby('Persona')['Score'].mean().reset_index()
avg_sentiment.columns = ['Persona', 'Average_Sentiment']
avg_sentiment['Sentiment_Label'] = avg_sentiment['Average_Sentiment'].apply(lambda x: "NEGATIVE" if x<=2.7 else ("NEUTRAL" if x<=3.5 else "POSITIVE"))

# 🔟 Mostrar resultados
print("✅ Análisis línea por línea:\n")
print(df[['Persona', 'Comentario', 'Score', 'Sentiment_Label']])

print("\n✅ Promedio por persona:\n")
print(avg_sentiment)
```
















