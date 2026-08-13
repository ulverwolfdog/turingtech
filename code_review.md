# Revisión técnica del pipeline de ingesta y consulta RAG

---

Se presernta un código que implementa un pipeline RAG básico basado en OpenAI y Chroma: 
- Genera embeddings de documentos. 
- Almacena los vectores es una DB vectorial.
- Recupera los documentos más similares a una pregunta y utiliza un modelo conversacional para generar la respuesta.

Se pide analizar el código y realizar un análisis de errores, bugs, seguridad, mantenibilidad, diseño, rendimiento, etc ...

El código es el siguiente:

```bash
import openai
import json
import chromadb
 
API_KEY = "sk-proj-xxxxxxxxxxxxxxxx"

client = chromadb.Client()
collection = client.create_collection("docs")
 
def ingest_documents(docs: list[str]):
    for i, doc in enumerate(docs):
        embedding = openai.Embedding.create(
            input=doc,
            model="text-embedding-ada-002",
            api_key=API_KEY
        )["data"][0]["embedding"]
        collection.add(
            documents=[doc],
            embeddings=[embedding],
            ids=[str(i)]
        )
 
def ask(question: str, history: list) -> str:
    q_embedding = openai.Embedding.create(
        input=question,
        model="text-embedding-ada-002",
        api_key=API_KEY
    )["data"][0]["embedding"]
 
    results = collection.query(query_embeddings=[q_embedding], n_results=5)
    context = " ".join(results["documents"][0])
 
    messages = [{"role": "system", "content": "Responde usando: " + context}]
    for turn in history:
        messages.append({"role": "user", "content": turn[0]})
        messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": question})
 
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        api_key=API_KEY
    )
    answer = resp["choices"][0]["message"]["content"]
    history.append((question, answer))
    open("history.json", "w").write(json.dumps(history))
    return answer
```
 A continuación passamos a revisar las distintas partes del código y exponer las razones por las que debería ser modificado:

#### 1. La API key está en texto plano y "hardcodeada"

```bash
API_KEY = "sk-proj-xxxxxxxxxxxxxxxx"
```

Las credenciales, keys, secrets, ... no deben formar parte del código fuente. Esta práctica puede hacer que aparezcan en Git, Notebooks, ficheros de código, logs entre otros, creando un problema de seguridad al ser vulnerable. Se deben gestionar mediante un secret manager (por ejemplo Azure Keyvault), o como mínimo, mediante variables de entorno. En este caso sustituiremos el valor por una variable de entorno. 

#### 2. La base de datos vectorial (Chroma) no es persistente

```bash
client = chromadb.Client()
```

En el flujo de iungesta se usa la BD para almacenar los embbedings, pero emplear un cliente en memoria no ofrece las garactías y el almacenamiento a medio o largo plazo necesario. Es preciso que la DB se almacene en local o usar otro tipop de DB vactorial adaptado a la arquitectura de la aplicación global. Para el almacenamiento local de los índices podemos emplear el comando ``` PersistentClient ```.

### 3. Creación de la colección de documentos

```bash
client.create_collection("docs")
```


