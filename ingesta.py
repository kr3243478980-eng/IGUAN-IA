# --- PASO 1: IMPORTACIONES ---
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# --- PASO 2: CONFIGURAR EL TRADUCTOR ---
# Usamos Llama 3 para convertir el texto en vectores (números)
embeddings = OllamaEmbeddings(model="llama3")

# --- PASO 3: CARGAR TU ARCHIVO DE TEXTO ---
# El archivo datos.txt debe estar en la misma carpeta
try:
    loader = TextLoader("./datos.txt", encoding="utf-8")
    documentos = loader.load()
    print("Archivo datos.txt cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")

# --- PASO 4: DIVIDIR EL TEXTO EN TROZOS ---
# Esto ayuda a que la IA encuentre respuestas más precisas
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
trozos = text_splitter.split_documents(documentos)

# --- PASO 5: GUARDAR EN LA BASE DE DATOS (DB_IGUAN) ---
print("Convirtiendo datos a vectores... Tu RTX 3050 está trabajando.")
vectorstore = Chroma.from_documents(
    documents=trozos, 
    embedding=embeddings, 
    persist_directory="./db_iguan"
)
print("¡PROCESO COMPLETADO! Ahora IGUAN ya tiene memoria.")