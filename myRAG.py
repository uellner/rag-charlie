from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
import os
import uuid

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from ragas import evaluate
from datasets import Dataset as HFDataset
from rag_evaluation import evaluation_examples

class NewRAGPipeline:
    def __init__(self, pdf_path, chroma_dir):
        """
        Inicializa la pipeline cargando variables de entorno, rutas, modelos de embeddings y LLM.
        """
        load_dotenv()  # Carga las variables desde el archivo .env
        set_llm_cache(None) # Desactiva el caché de LLM para evitar problemas de memoria
        self.pdf_path = pdf_path  # Ruta al PDF con los textos
        self.chroma_dir = chroma_dir  # Directorio donde se guardará el índice vectorial
        self.api_key = os.getenv("OPENAI_API_KEY")  # Clave de API de OpenAI desde el .env

        # Modelo de embeddings de OpenAI
        self.embedding_model = OpenAIEmbeddings(
            api_key=self.api_key,
            model="text-embedding-3-small"
        )

        # LLM (modelo de lenguaje) utilizado para generar respuestas
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo",
            temperature=0.3,
            request_timeout=60,
            cache=False  # Impede cache interno da LangChain
        )

        if self.vectorstore_is_empty():
            docs = self.load_documents()  # 1. Carga los PDFs
            chunks = self.split_documents_character(docs)  # 2. Divide en partes (por caracteres)
            # chunks = self.split_documents_semantic(docs)  # Alternativamente: divide semánticamente
            self.vector_store = self.create_vector_store(chunks)  # 3. Crea el índice vectorial
        else:
            self.vector_store = self.load_vector_store()  # 3. Crea el índice vectorial
        
        self.retriever = self.vector_store.as_retriever(  # 4. Crea el recuperador Chroma
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        
        self.chain = self.build_chain()  # 5. Construye el flujo RAG

    
    def load_documents(self):
        """
        Carga el contenido del PDF usando PyMuPDFLoader.
        """
        print("\nCargando los PDFs...")

        pdf_files = [f for f in os.listdir(self.pdf_path) if f.endswith('.pdf')]

        all_docs = []
        for pdf_file in pdf_files:
            print(f"Cargando {pdf_file}...")
            loader = PyMuPDFLoader(os.path.join(self.pdf_path, pdf_file))
            docs = loader.load()
            all_docs.extend(docs)

        # loader = PyMuPDFLoader(self.pdf_path)
        # docs = loader.load()
        print("¡PDFs cargados correctamente!")
        return all_docs
    
    def split_documents_character(self, docs):
        """
        Divide los documentos en fragmentos pequeños (chunks) usando un separador basado en caracteres.
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "\n\n\n"],  # Prioriza saltos de párrafo
            chunk_size=1000,
            chunk_overlap=200,
        )
        return splitter.split_documents(docs)
    
    def split_documents_semantic(self, docs):
        """
        Divide los documentos en fragmentos pequeños usando un separador semántico.
        """
        chunker = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.8,  # sensibilidad entre 0.0 y 1.0
        )
        return chunker.split_documents(docs)
    
    def vectorstore_is_empty(self):
        """
        Verifica si el índice vectorial Chroma está vacío.
        """
        return not os.path.exists(self.chroma_dir) or not os.listdir(self.chroma_dir)
    
    
    def create_vector_store(self, chunks):
        """
        Crea un índice vectorial Chroma a partir de los fragmentos de texto.
        """
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.chroma_dir
        )
        vectorstore.persist()  # Guarda el índice en el directorio especificado
        return vectorstore
    
    def load_vector_store(self):
        """
        Carga el índice vectorial Chroma desde el directorio especificado.
        """
        vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.chroma_dir
        )
        # print("¡Índice vectorial cargado correctamente!")
        return vectorstore

    
    def build_chain(self):
        """
        Construye el pipeline (cadena) de RAG: recibe una pregunta, recupera el contexto y genera la respuesta.
        """
        execution_id = str(uuid.uuid4())  # novo ID para cada chain
        prompt = ChatPromptTemplate.from_template("""
            ID de execução: {execution_id}
            Eres un asistente de IA especializado en responder preguntas sobre los programas de máster de la Universidad Europea.

            Puedes entender preguntas en varios idiomas, pero solo tienes acceso a documentos oficiales en español, que se proporcionan en el contexto a continuación.

            Utiliza **exclusivamente** la información contenida en los materiales y documentos proporcionados en el contexto, extraídos directamente de los PDFs oficiales de la Universidad Europea.

            No inventes datos ni uses conocimientos externos que no estén presentes en el contexto.

            ---
            **Fragmento del texto:**
            {context}

            **Pregunta:**
            {question}
            ---
            **Respuesta clara, precisa y basada únicamente en los documentos oficiales de la Universidad Europea:**
        """)

        # # Função para buscar documentos e concatenar seu conteúdo em uma string
        # def retrieve_and_format(input_dict):
        #     question = input_dict["question"]
        #     docs = self.retriever.invoke(question)  # retorna lista de documentos
        #     # Concatenar o conteúdo dos documentos para enviar ao prompt
        #     return "\n\n".join([doc.page_content for doc in docs])
        
        def retrieve_and_format(question_dict):
            try:
                docs = self.retriever.invoke(question_dict["question"])
                print(f"\n🔍 Documentos recuperados: {len(docs)}")
                # for i, doc in enumerate(docs):
                #     print(f"\n--- Documento {i+1} ---\n{doc.page_content[:500]}\n...")

                context = "\n\n".join([doc.page_content for doc in docs])
                print(f"\n📄 Contexto final com {len(context)} caracteres")
                return context
            except Exception as e:
                print(f"❌ Erro ao recuperar/formatar contexto: {e}")
                return ""


        # Define la cadena de ejecución de LangChain: recuperación → prompt → LLM → parser
        return (
            {
                "context": RunnableLambda(retrieve_and_format),
                # "context": RunnableLambda(lambda x: self.retriever.invoke(x["question"])),  # Recupera contexto en base a la pregunta
                "question": RunnablePassthrough(),  # Pasa directamente la pregunta
                "execution_id": RunnableLambda(lambda _: execution_id)
            }
            | prompt  # Aplica el prompt al contexto y la pregunta
            | self.llm  # Usa el modelo de OpenAI
            | StrOutputParser()  # Convierte la respuesta en una cadena
        )


    # def run(self, question: str):
    #     """
    #     Ejecuta toda la pipeline: carga documentos, indexa, busca y responde a la pregunta.
    #     """
    #     context_docs = self.retriever.invoke(question)  # Recupera los documentos relevantes
    #     print(f"Documentos recuperados: {len(context_docs)}")  # Imprime la cantidad de documentos recuperados

    #     chain = self.build_chain()
        
    #     result = chain.invoke({"question": question})  # 6. Ejecuta con la pregunta
    #     return result, context_docs

    def run(self, question: str):
        
        def perturbar_pergunta(question: str) -> str:
            return f"{question} [{uuid.uuid4().hex[:6]}]"

        try:
            question = perturbar_pergunta(question) # Perturba a pergunta para evitar problemas de cache
            context_docs = self.retriever.invoke(question)

            chain = self.build_chain()
            result = chain.invoke({"question": question})
            return result, context_docs
        except Exception as e:
            print(f"❌ Erro no run: {e}")
            return "Ocorreu um erro ao tentar responder sua pergunta.", []

    

    def run_with_eval(self, question: str, ground_truth: str):
        """
        Ejecuta la pipeline y evalúa la respuesta generada con base en métricas de RAG.
        """
        # Ejecuta la pipeline
        answer, context_docs = self.run(question)

        # Extrae el texto de los documentos recuperados
        context_strings = [doc.page_content for doc in context_docs]

        example = {
            "question": question,
            "contexts": context_strings,
            "answer": answer,
        }

        if ground_truth:
            example["ground_truth"] = ground_truth

        # Crea el dataset para evaluación
        dataset = HFDataset.from_list([example])

        # Evalúa la respuesta con las métricas de RAGAS
        metrics = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        )
        return answer, metrics
    
if __name__ == "__main__":

    FOLDER_PATH = "./data"

    pipeline = NewRAGPipeline(
        pdf_path=FOLDER_PATH,
        chroma_dir="./chroma_index"
    )

    print("\n\n¡Bienvenido al asistente virtual especializado en los programas de máster de la Universidad Europea (UE)!\n")
    print("Aquí puedes hacer todas tus preguntas sobre los contenidos, requisitos y detalles de los másteres que ofrece la UE.\n")
    print("Si tienes alguna duda, no dudes en preguntar. Estoy aquí para ayudarte a obtener la información más precisa y actualizada basada en los documentos oficiales.\n")
    print("\nPuedes hacer todas tus preguntas sobre los másteres. Escribe 'salir' para terminar.\n")

    
    while True:
        # Pregunta al usuario
        question = input("\n\n- ")
        if question.lower() in ['salir', 'exit', 'quit', 'sair']:
            print("¡Gracias por usar el asistente virtual! ¡Hasta luego!")
            break
        
        try:
            answer, context_docs = pipeline.run(question)
            print(f"\n\n-> {answer}\n")
        except Exception as e:
            print(f"\n\n- Error al procesar la pregunta: {e}\n")
        # print(f"- Context_Docs: {context_docs}\n")

    # Evaluación
    # answer, metrics = pipeline.run_with_eval(
    #     question=question,
    #     ground_truth=evaluation_examples[0]["ground_truth"]
    # )
    # print(f"\n\n- Pregunta: {question}")
    # print(f"- Respuesta: {answer}")
    # print(f"- Métricas: {metrics}\n")
