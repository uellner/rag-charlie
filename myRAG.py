from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import os

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
            temperature=0,
        )
    
    def load_documents(self):
        """
        Carga el contenido del PDF usando PyMuPDFLoader.
        """
        print("\nCargando el PDF...")
        loader = PyMuPDFLoader(self.pdf_path)
        docs = loader.load()
        print("¡PDF cargado correctamente!")
        return docs
    
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
    
    def create_vector_store(self, chunks):
        """
        Crea un índice vectorial Chroma a partir de los fragmentos de texto.
        """
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.chroma_dir
        )

    def build_chain(self, retriever):
        """
        Construye el pipeline (cadena) de RAG: recibe una pregunta, recupera el contexto y genera la respuesta.
        """
        prompt = ChatPromptTemplate.from_template("""
            Eres un asistente de IA entrenado para responder preguntas basadas **exclusivamente en la Enciclopedia Felina del autor PetFood Institute**.
            Utiliza únicamente la información proporcionada en el contexto a continuación, que ha sido extraída directamente del libro escrito por PetFood Institute.  
            No inventes datos ni utilices conocimientos externos.
            Si la respuesta no está claramente presente en el contenido, di que **no es posible responder basándose en el texto proporcionado**.

            ---
            **Fragmento del texto:**
            {context}

            **Pregunta:**
            {question}
            ---
            **Respuesta clara y fiel a la guía para comprender a los gatos:**
        """)

        # Define la cadena de ejecución de LangChain: recuperación → prompt → LLM → parser
        return (
            {
                "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),  # Recupera contexto en base a la pregunta
                "question": RunnablePassthrough()  # Pasa directamente la pregunta
            }
            | prompt  # Aplica el prompt al contexto y la pregunta
            | self.llm  # Usa el modelo de OpenAI
            | StrOutputParser()  # Convierte la respuesta en una cadena
        )

    def run(self, question: str):
        """
        Ejecuta toda la pipeline: carga documentos, indexa, busca y responde a la pregunta.
        """
        docs = self.load_documents()  # 1. Carga el PDF
        chunks = self.split_documents_character(docs)  # 2. Divide en partes (por caracteres)
        # chunks = self.split_documents_semantic(docs)  # Alternativamente: divide semánticamente
        vector_store = self.create_vector_store(chunks)  # 3. Crea el índice vectorial
        retriever = vector_store.as_retriever(  # 4. Crea el recuperador Chroma
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        context_docs = retriever.invoke(question)  # Recupera los documentos relevantes
        chain = self.build_chain(retriever)  # 5. Construye el flujo RAG
        result = chain.invoke({"question": question})  # 6. Ejecuta con la pregunta
        return result, context_docs  

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
    pipeline = NewRAGPipeline(
        pdf_path="./data/EnciclopediaFelinaPetFoodInstitute.pdf",
        chroma_dir="./chroma_index"
    )

    question = input("Pregunta sobre gatos: ")

    # Evaluación
    answer, metrics = pipeline.run_with_eval(
        question=question,
        ground_truth=evaluation_examples[0]["ground_truth"]
    )
    print(f"\n\n- Pregunta: {question}")
    print(f"- Respuesta: {answer}")
    print(f"- Métricas: {metrics}\n")
