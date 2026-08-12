###########################################################################################
##
##
##
##
##
###########################################################################################

'''

Load libraries 

'''

import os
import sys
from pathlib import Path
from typing import Optional, Literal
import requests

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field

'''

CONFIGURACIÓN Y PARAMETRIZACIÓN DEL CHATBOT

Asignación de variables y rutas

'''

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path(".") / "index"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.magicthegathering.io/v1"
TIMEOUT_SECONDS = 10
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

chunk_size = 800
chunk_overlap = 150
path_to_rules = "./data/MagicCompRules 20260417.pdf"

'''

Definición de los prompts

'''

RAG_AGENT_PROMPT = (
    "Eres un juez experto de Magic: The Gathering especializado en reglas. "
    "Respondes dudas de reglas básicas (fases del turno, maná, prioridad, la "
    "pila...) e interacciones entre habilidades de cartas.\n\n"
    "SIEMPRE debes usar la tool `search_mtg_rules` antes de responder, aunque "
    "creas conocer la respuesta, para fundamentar tu explicación en el "
    "reglamento. Si la pregunta menciona cartas concretas, razona paso a paso "
    "sobre qué reglas generales aplican a las habilidades mencionadas.\n\n"
    "Formato de respuesta:\n"
    "1. Respuesta directa y clara a la pregunta.\n"
    "2. Explicación breve del razonamiento.\n"
    "3. Cita la regla o fragmento del reglamento en que te basas.\n\n"
    "Si el reglamento indexado no cubre el caso con suficiente detalle, dilo "
    "explícitamente en vez de inventar la regla, y sugiere escalar a un "
    "agente humano del call center."
)

API_AGENT_PROMPT = (
    "Eres un asistente especializado en la base de datos de cartas de Magic: "
    "The Gathering. Ayudas a los usuarios a encontrar cartas según su "
    "descripción (color, coste de maná, tipo, texto de habilidad, nombre) y "
    "a consultar novedades y sets recientes.\n\n"
    "Usa la tool `search_cards` para búsquedas por criterios, `get_card_by_name` "
    "cuando el usuario mencione una carta concreta por nombre, y "
    "`get_recent_sets` para preguntas sobre las últimas expansiones/novedades.\n\n"
    "Presenta los resultados de forma clara y concisa (nombre, coste, colores, "
    "tipo, texto de reglas resumido, e imagen si está disponible). Si no "
    "encuentras resultados, dilo claramente y sugiere relajar algún criterio "
    "de búsqueda."
)

CARD_CREATOR_PROMPT = (
    "Eres un diseñador de cartas custom de Magic: The Gathering. El usuario te "
    "pedirá crear una carta original (por ejemplo, basada en un personaje o "
    "concepto).\n\n"
    "Entrega la carta en este formato:\n"
    "- Nombre\n"
    "- Coste de maná (usa la notación estándar, p. ej. {1}{R}{W})\n"
    "- Tipo (p. ej. Criatura Legendaria — Humano Pícaro)\n"
    "- Texto de reglas (habilidades con nomenclatura oficial, p. ej. 'Daño "
    "primero', 'Arrollar')\n"
    "- Poder/Resistencia (si aplica)\n"
    "- Una línea de sabor (flavor text)\n"
    "- Una breve justificación de por qué el coste/estadísticas están "
    "balanceados, citando la carta real usada como referencia."
)

SUPERVISOR_SYSTEM_PROMPT = (
    "Eres el supervisor de un chatbot de atención al cliente para el juego "
    "Magic: The Gathering. Tu única tarea es analizar el último mensaje del "
    "usuario (con el resto de la conversación como contexto) y decidir a qué "
    "agente especializado enviarlo:\n\n"
    "- rag_agent: preguntas de reglas básicas del juego (fases del turno, "
    "maná, prioridad, la pila...) o preguntas sobre cómo interactúan las "
    "habilidades de cartas EXISTENTES entre sí (p. ej. daño primero al "
    "cambiar de controlador).\n"
    "- api_agent: el usuario busca cartas reales según una descripción "
    "(color, coste, tipo, texto...), pide el detalle/imagen de una carta "
    "concreta, o pregunta por novedades/últimos sets.\n"
    "- card_creator_agent: el usuario pide crear, diseñar o inventar una "
    "carta custom nueva (que no existe en el juego).\n\n"
    "No respondas la pregunta tú mismo, solo decide el enrutado."
)

"""

FUNCIONES GESTIÓN DOCUMENTOS (Reglas de Magic: The Gathering)

Funciones para indexar el reglamento de Magic: The Gathering en una base de datos vectorial local
(FAISS) para poder hacer búsqueda semántica (RAG) sobre el documento por parte del usuario. 
Acepta ficheros PDF y Markdown, que deden depositarse en el durectorio 'data/fichero'. Una de las variables
a definir en la configiración es la ruta del fichero que se desea emplear. 

Para realizar los embeddings de los distintos chunks del documento de reglas se usa un modelo local 
de HuggingFace (sentence-transformers), de esta forma podemos realizar la demo sin recurrir a modelos de pago.
El cambio por un modelo para los embeddings de pago (OpenAI, Cohere, etc.) es trivial, basta con cambiar 
la clase HuggingFaceEmbeddings por la clase correspondiente.

"""

def load_documents(source_path: str):
    """Carga un documento de reglas desde disco y lo devuelve como lista de documentos.

    La función valida que la ruta exista, detecta el tipo de fichero por su extensión
    y usa el loader adecuado:

    - .pdf: usa PyPDFLoader para extraer texto desde archivos PDF.
    - .md: intenta cargar el contenido con UnstructuredMarkdownLoader y, si falla,
      recurre a TextLoader como fallback.
    - cualquier otro formato: usa TextLoader como texto plano.

    Args:
        source_path: Ruta del archivo que contiene el reglamento o texto base.

    Returns:
        List[Document]: Documentos cargados por LangChain listos para ser troceados
        y embebidos en el índice FAISS.

    Raises:
        FileNotFoundError: Si la ruta indicada no existe.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el fichero de reglas: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".md":
        # Fallback simple: cargar como texto plano si Unstructured no está disponible
        try:
            loader = UnstructuredMarkdownLoader(str(path))
        except Exception:
            loader = TextLoader(str(path), encoding="utf-8")
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    return loader.load()


def build_index(source_path: str, index_dir: Path = INDEX_DIR, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
    """Construye y guarda un índice vectorial FAISS a partir de un documento de reglas.

    La función carga el archivo fuente, lo divide en fragmentos con un splitter
    semántico, genera embeddings con el modelo configurado y guarda el resultado
    en el directorio indicado para su uso posterior en búsquedas RAG.

    Args:
        source_path: Ruta del documento fuente (PDF o Markdown) con el reglamento.
        index_dir: Directorio donde se almacenará el índice FAISS generado.
        chunk_size: Tamaño máximo de cada fragmento en tokens/caracteres.
        chunk_overlap: Superposición entre fragmentos consecutivos para preservar contexto.

    Returns:
        None: Guarda el índice localmente en disco.
    """
    print(f"[ingest] Cargando documento: {source_path}")
    docs = load_documents(source_path)
    print(f"[ingest] {len(docs)} documento(s) cargado(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Documento troceado en {len(chunks)} fragmentos")

    print(f"[ingest] Generando embeddings con {EMBEDDING_MODEL} (descarga el modelo la primera vez)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    print(f"[ingest] Índice FAISS guardado en: {index_dir}")


def load_retriever(index_dir: Path = INDEX_DIR, k: int = 4):
    """Carga un índice FAISS guardado y devuelve un retriever para búsquedas semánticas.

    La función reconstruye el vector store a partir del directorio del índice
    almacenado en disco usando el modelo de embeddings configurado. Luego
    devuelve un retriever que recupera los `k` fragmentos más relevantes para
    una consulta del usuario.

    Args:
        index_dir: Ruta del directorio donde está guardado el índice FAISS.
        k: Número de fragmentos que devolverá el retriever por consulta.

    Returns:
        Retriever: Objeto de LangChain listo para buscar documentos relevantes.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": k})

"""

FUNCIONES SISTEMA RAG (Reglas de Magic: The Gathering)

Expone el reglamento indexado a través de las funciones de gestión de documentos como una tool de
LangChain mediante lista para pasarse a un agente de LangGraph. Esta es la tool que usa el 
"Agente especialista RAG" para responder dudas de reglas e interacciones entre cartas.

"""

retriever = None

'''

Definición de la tool de búsqueda semántica sobre el reglamento de Magic: The Gathering.
En este caso el docstring nos sirve para documentar la tool y explicar su uso, lo que permitirá
al agente RAG decidir cuándo usarla (en función de la consulta del usuario) y cómo.

'''

def get_rules_search_tool():
    """
    Devuelve la tool de búsqueda semántica sobre el reglamento de Magic: The Gathering.
    
    La herramienta usa el índice FAISS ya cargado para recuperar los fragmentos
    más relevantes a partir de una pregunta o descripción del estado del juego.
    Se usa principalmente para responder dudas sobre reglas, fases del turno,
    prioridad, pila y interacciones entre cartas.

    Returns:
        Tool: herramienta de LangChain preparada para ser añadida a un agente.
            
    """
    global retriever
    if retriever is None:
        retriever = load_retriever()

    return create_retriever_tool(
        retriever,
        name="search_mtg_rules",
        description=(
            "Busca en el reglamento oficial de Magic: The Gathering fragmentos "
            "relevantes para responder preguntas sobre reglas básicas (fases del "
            "turno, maná, prioridad, la pila, acciones basadas en estado, etc.) "
            "o sobre interacciones entre habilidades de cartas (p. ej. daño "
            "primero, daño doble, cambio de controlador en combate). "
            "La entrada será una pregunta o descripción de la situación de juego en "
            "lenguaje natural. Úsala siempre antes de responder cualquier "
            "pregunta de reglas, y cita siempre el fragmento del reglamento "
            "en el que se basa la respuesta."
        ),
    )


"""

Tools de LangChain que consultan la API pública de Magic: The Gathering
(https://docs.magicthegathering.io/):

  - Buscar cartas por criterios (color, coste de maná, tipo, texto, nombre...)
  - Obtener el detalle/imagen de una carta concreta
  - Listar sets recientes (para dudas sobre "nuevos releases")

Estas tools las usa el "Agente especialista API", especializado en búsqueda de
cartas.

"""

@tool
def search_cards(
    name: Optional[str] = None,
    colors: Optional[str] = None,
    type: Optional[str] = None,
    cmc_lte: Optional[float] = None,
    cmc_gte: Optional[float] = None,
    text: Optional[str] = None,
    page_size: int = 5,
) -> str:
    """
    Busca cartas de Magic: The Gathering usando la API oficial según los
    criterios indicados. Usa esta tool para cualquier petición de búsqueda
    de cartas por descripción (color, coste de maná, tipo de criatura,
    texto de habilidad, nombre parcial, etc.).

    Args:
        name: nombre (parcial) de la carta a buscar.
        colors: colores separados por coma en inglés, p. ej. "White",
            "Red,Blue". Colores válidos: White, Blue, Black, Red, Green.
        type: tipo o subtipo de carta, p. ej. "Creature", "Warrior",
            "Instant". Puede combinarse con espacio, p. ej. "Creature Warrior".
        cmc_lte: coste de maná convertido máximo (menor o igual que).
        cmc_gte: coste de maná convertido mínimo (mayor o igual que).
        text: texto que debe aparecer en la casilla de reglas de la carta.
        page_size: número máximo de resultados a devolver (por defecto 5).

    Returns:
        Un resumen en texto de las cartas encontradas (nombre, coste,
        colores, tipo, texto de reglas e imagen si está disponible).
    """
    params = {"pageSize": min(page_size, 20)}
    if name:
        params["name"] = name
    if colors:
        params["colors"] = colors
    if type:
        params["type"] = type
    if text:
        params["text"] = text

    try:
        resp = requests.get(f"{BASE_URL}/cards", params=params, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Error consultando la API de Magic the Gathering: {exc}"

    cards = resp.json().get("cards", [])

    # La API pública no soporta filtro directo por rango de cmc, se filtra
    # aquí sobre los resultados devueltos.
    if cmc_lte is not None:
        cards = [c for c in cards if c.get("cmc") is not None and c["cmc"] <= cmc_lte]
    if cmc_gte is not None:
        cards = [c for c in cards if c.get("cmc") is not None and c["cmc"] >= cmc_gte]

    if not cards:
        return "No se encontraron cartas que cumplan esos criterios."

    lines = []
    for c in cards[:page_size]:
        lines.append(
            f"- {c.get('name')} | Coste: {c.get('manaCost', 'N/A')} "
            f"(CMC {c.get('cmc', 'N/A')}) | Colores: {', '.join(c.get('colors') or []) or 'Incoloro'} "
            f"| Tipo: {c.get('type', 'N/A')} | Rareza: {c.get('rarity', 'N/A')}\n"
            f"  Texto: {c.get('text', '(sin texto de reglas)')}\n"
            f"  Imagen: {c.get('imageUrl', 'N/A')}"
        )
    return "\n".join(lines)


@tool
def get_card_by_name(name: str) -> str:
    """
    Obtiene el detalle completo (texto de reglas, coste, imagen, set de
    origen, etc.) de una carta concreta buscando por su nombre exacto o
    aproximado. Útil cuando el usuario menciona una carta por su nombre y
    se necesita su información real, por ejemplo para resolver una
    interacción entre cartas o para usarla como referencia al crear una
    carta custom.

    Args:
        name: nombre de la carta a buscar.
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/cards", params={"name": name, "pageSize": 3}, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Error consultando la API de Magic the Gathering: {exc}"

    cards = resp.json().get("cards", [])
    if not cards:
        return f"No se encontró ninguna carta con el nombre '{name}'."

    card = cards[0]
    return (
        f"Nombre: {card.get('name')}\n"
        f"Coste de maná: {card.get('manaCost', 'N/A')} (CMC {card.get('cmc', 'N/A')})\n"
        f"Colores: {', '.join(card.get('colors') or []) or 'Incoloro'}\n"
        f"Tipo: {card.get('type', 'N/A')}\n"
        f"Poder/Resistencia: {card.get('power', '-')}/{card.get('toughness', '-')}\n"
        f"Texto de reglas: {card.get('text', '(sin texto de reglas)')}\n"
        f"Set: {card.get('setName', 'N/A')}\n"
        f"Imagen: {card.get('imageUrl', 'N/A')}"
    )


@tool
def get_recent_sets(page_size: int = 5) -> str:
    """
    Devuelve los sets (expansiones) de Magic: The Gathering más recientes,
    útil para responder preguntas del usuario sobre novedades o últimos
    lanzamientos ("¿Qué carta ha salido nuevo?", "¿Cuál es la última
    expansión?").

    Args:
        page_size: número máximo de sets a devolver.
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/sets", params={"pageSize": min(page_size, 20)}, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Error consultando la API de Magic the Gathering: {exc}"

    sets = resp.json().get("sets", [])
    if not sets:
        return "No se pudo obtener información de sets recientes."

    # Ordenar por fecha de salida descendente cuando esté disponible
    sets_sorted = sorted(sets, key=lambda s: s.get("releaseDate") or "", reverse=True)

    lines = [
        f"- {s.get('name')} ({s.get('code')}) — lanzamiento: {s.get('releaseDate', 'N/A')}, "
        f"tipo: {s.get('type', 'N/A')}"
        for s in sets_sorted[:page_size]
    ]
    return "\n".join(lines)


'''

AGENTES ESPECIALIZADOS (RAG; API; CARD CREATOR)

Construye los agentes especializados usando el framework de IA Agéntica
LangGraph.

Agentes:
  - rag_agent: resuelve dudas de reglas básicas e interacciones entre
    cartas, apoyándose en la tool de búsqueda semántica sobre el reglamento.
  - api_agent: busca cartas reales y responde preguntas sobre novedades,
    usando las tools que consultan la API de magicthegathering.io.
  - card_creator_agent: diseña cartas custom

La API para llamar al modelo LLM es un tier gratuito de Groq  y el modelo empleado 
en esta demo es llama-3.3-70b-versatile.

'''

def llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=temperature)


def build_rag_agent():
    tools = [get_rules_search_tool()]
    return create_agent(llm(), tools, system_prompt=RAG_AGENT_PROMPT, name="rag_agent")


def build_api_agent():
    tools = [search_cards, get_card_by_name, get_recent_sets]
    return create_agent(llm(), tools, system_prompt=API_AGENT_PROMPT, name="api_agent")


def build_card_creator_agent():
    tools = [search_cards, get_card_by_name]
    return create_agent(llm(temperature=0.7), tools, system_prompt=CARD_CREATOR_PROMPT, name="card_creator_agent")


'''

AGENTE SUPERVISOR (ROUTER) Y ERNTRADA DEL USUARIO AL CHATBOT

El nodo supervisor clasifica la intención del usuario y usa el patrón de handoff (paso de control) 
meiante `Command(goto=...)` de LangGraph para enviar la conversación al agente
especializado correspondiente:

  - rag_agent          -> dudas de reglas básicas e interacciones entre cartas
  - api_agent          -> búsqueda de cartas / novedades vía API oficial
  - card_creator_agent -> (bonus) creación de cartas custom

Cada agente especializado es a su vez un grafo ya compilado, que aquí se usa directamente como
nodo del grafo principal.

La estrucutra del grafo resultante sería:

    START -> supervisor -> {rag_agent | api_agent | card_creator_agent} -> END

'''

AgentName = Literal["rag_agent", "api_agent", "card_creator_agent"]
class RouteDecision(BaseModel):
    """Decisión de enrutado tomada por el supervisor."""

    agent: AgentName = Field(
        description=(
            "Agente al que enviar la conversación: "
            "'rag_agent' para dudas de reglas básicas o interacciones entre "
            "cartas existentes (requieren consultar el reglamento); "
            "'api_agent' para búsqueda de cartas por descripción/criterios o "
            "preguntas sobre novedades/sets recientes; "
            "'card_creator_agent' cuando el usuario pide crear/diseñar una "
            "carta custom nueva que no existe."
        )
    )
    reason: str = Field(description="Breve justificación (una frase) de la decisión de enrutado.")


def build_router_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0).with_structured_output(RouteDecision)


'''

Cración y compilación del grafo principal del chatbot, que incluye el nodo supervisor y los agentes especializados.

'''

def build_graph():
    router_llm = build_router_llm()
    rag_agent = build_rag_agent()
    api_agent = build_api_agent()
    card_creator_agent = build_card_creator_agent()

    def supervisor(state: MessagesState) -> Command[AgentName]:
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
        decision: RouteDecision = router_llm.invoke(messages)
        return Command(goto=decision.agent)

    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("api_agent", api_agent)
    graph.add_node("card_creator_agent", card_creator_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("rag_agent", END)
    graph.add_edge("api_agent", END)
    graph.add_edge("card_creator_agent", END)

    return graph.compile()


def main():
    
    if not GROQ_API_KEY:
        print(
            "ERROR: falta la variable de entorno GROQ_API_KEY. "
            "Expórtala antes de ejecutar la demo.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        build_index(path_to_rules)
    except Exception as exc:
        print(f"[ingest] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    
    print("Construyendo el grafo del chatbot (supervisor + agentes)...")
    app = build_graph()

    print("\n=== Chatbot MTG (demo) — escribe 'salir' para terminar ===\n")
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in {"salir", "exit", "quit"}:
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        result = app.invoke({"messages": history})
        history = result["messages"]

        last_message = history[-1]
        print(f"\nAsistente MTG: {last_message.content}\n")

if __name__ == "__main__":
    main()