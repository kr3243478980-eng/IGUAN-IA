# --- PASO 1: IMPORTACIONES ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

# --- PASO 2: CONFIGURACIÓN DE MODELOS ---
# El cerebro para razonar y el traductor para buscar
llm = OllamaLLM(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")

# --- PASO 3: CONEXIÓN A LA BASE DE DATOS LOCAL ---
# Accedemos a la carpeta db_iguan que creamos con el otro script
vectorstore = Chroma(
    embedding_function=embeddings, 
    persist_directory="./db_iguan"
)

# --- PASO 4: FUNCIÓN DE BÚSQUEDA ---
def buscar_en_archivo(pregunta_usuario):
    # Buscamos los 3 fragmentos de texto más relevantes
    docs = vectorstore.similarity_search(pregunta_usuario, k=3)
    
    # Unimos los fragmentos en un solo bloque de contexto
    contexto_limpio = "\n\n".join([d.page_content for d in docs])
    
    # Creamos el mensaje final para la IA
    prompt_final = f"""
    Eres IGUAN IA, un asistente experto en Barrancabermeja.
    Usa el siguiente contexto para responder la pregunta de forma precisa.
    Si la información no está en el contexto, dilo amablemente.
    
    CONTEXTO:
    {contexto_limpio}
    
    PREGUNTA: {pregunta_usuario}
    """
    
    # Generamos la respuesta usando el hardware de tu Asus TUF
    respuesta = llm.invoke(prompt_final)
    print(f"\nIGUAN IA dice: {respuesta}")

# --- PASO 5: EL DISPARADOR DEL CHAT ---
if __name__ == "__main__":
    print("\n--- SISTEMA IGUAN IA INICIADO (LOCAL) ---")
    print("Escribe 'salir' para finalizar.\n")
    while True:
        usuario = input("Tu pregunta: ")
        if usuario.lower() in ['salir', 'exit', 'quit']:
            print("Cerrando IGUAN IA. ¡Hasta luego!")
            break
        
        if usuario.strip() == "":
            continue
            
        buscar_en_archivo(usuario)
