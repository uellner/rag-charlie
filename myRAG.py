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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import requests

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
        self.api_key_heygen = os.getenv("HEYGEN_API_KEY")  # Clave de API de HeyGen desde el .env

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
    
    def split_documents_chunks(self, docs):
        """
        Divide os documentos em fragmentos maiores (chunks) com base no tamanho do texto.
        """
        chunks = []
        
        for doc in docs:
            doc_content = doc.page_content
            
            # Vamos aumentar o tamanho do chunk para 3000 caracteres
            doc_chunks = [doc_content[i:i+3000] for i in range(0, len(doc_content), 3000)]  # Blocos de 3000 caracteres
            
            # Filtra os chunks para garantir que o tamanho seja adequado
            chunks.extend([chunk for chunk in doc_chunks if len(chunk) > 200])  # Chunks com mais de 200 caracteres
        
        return chunks
    
    # def split_documents_by_sentences(self, docs):
    #     """
    #     Divide os documentos em chunks por sentenças, o que pode gerar fragmentos mais significativos.
    #     """
    #     sentence_splitter = SentenceTextSplitter(
    #         chunk_size=2000,  # Tamanho maior para a sentença
    #         chunk_overlap=400,  # Sobreposição entre os chunks
    #     )
        
    #     all_chunks = []
    #     for doc in docs:
    #         doc_chunks = sentence_splitter.split_text(doc.page_content)
    #         all_chunks.extend(doc_chunks)
        
    #     return all_chunks
    
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

                context = "\n\n".join([doc.page_content for doc in docs])
                # print(f"\n📄 Contexto final com {len(context)} caracteres")
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

    # Função para calcular a similaridade entre a pergunta e os documentos
    def calculate_similarity(self, question, context_docs):
        """
        Calcula a similaridade entre a pergunta e os documentos fornecidos.
        """
        # Obter os embeddings da pergunta (certifique-se de que seja uma matriz 2D)
        question_embedding = self.embedding_model.embed_query(question)  # Obtém o embedding da pergunta
        question_embedding = np.array(question_embedding).reshape(1, -1)  # Reformata para 2D (1, n_features)

        # Obter os embeddings dos documentos (também em 2D)
        doc_embeddings = self.embedding_model.embed_documents([doc.page_content for doc in context_docs])  # Embeddings dos documentos
        doc_embeddings = np.array(doc_embeddings)  # Garante que seja um array numpy
        doc_embeddings = doc_embeddings.reshape(len(doc_embeddings), -1)  # Reformata para 2D

        similarities = []
        for doc_embedding in doc_embeddings:
            # Calcular a similaridade de cosseno entre a pergunta e o documento
            sim_score = cosine_similarity(question_embedding, doc_embedding.reshape(1, -1))  # Reformata o doc_embedding para 2D
            similarities.append(sim_score[0][0])  # Similaridade em si (valor único)

        return similarities
    
    def print_docs_with_scores(self, context_docs, similarities):
        # Adicionar a pontuação nos documentos
        for i, doc in enumerate(context_docs):
            doc.metadata['score'] = similarities[i]  # Adicionando a pontuação de similaridade no metadado

        # Ordenar os documentos pela similaridade (pontuação mais alta primeiro)
        context_docs_sorted = sorted(context_docs, key=lambda x: x.metadata['score'], reverse=True)

        # Exibir os top-n documentos mais relevantes
        n = 3  # Número de documentos a exibir
        for i, doc in enumerate(context_docs_sorted[:n]):
            print(f"\n--- Document {i+1} ---")
            print(f"Similarity Score: {doc.metadata['score']}")
            print(doc.page_content)  # Exibe o conteúdo do documento

    def generar_video_heygen(self, texto, avatar_id):
        url = "https://api.heygen.com/v2/video/generate"
        headers = {
            'X-Api-Key': self.api_key_heygen,
            'Content-Type': 'application/json'
        }
        data = {
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        # "avatar_id": "Daisy-inskirt-20220818",
                        "avatar_id": avatar_id,  # Usa el avatar_id proporcionado
                        "avatar_style": "normal"
                    },
                    "voice": {
                        "type": "text",
                        "input_text": texto,
                        # "voice_id": "2d5b0e6cf36f460aa7fc47e3eee4ba54"
                        "voice_id": "789278f26ffd4f2b833f348e76f3c4bc"  # Usa una voz en español
                    },
                    "background": {
                        "type": "color",
                        "value": "#080808"
                    }
                }
            ],
            "dimension": {
                "width": 1280,
                "height": 720
            }
        }
        r = requests.post(url, json=data, headers=headers)
        # print(f"Status Code: {r.status_code}")
        # print(r.json())
        if r.status_code == 200:
            print("\n✅ Vídeo enviado para generación!")
            video_request_id = r.json()["data"]["video_id"]
            self.wait_for_video(video_request_id=video_request_id)  # Espera hasta que el video esté listo
        else:
            print("\n❌ Error al enviar el vídeo para generación.")
            print(f"Mensaje de error: {r.json().get('message', 'No se proporcionó mensaje de error')}")


    def wait_for_video(self, video_request_id):
        """
        Espera hasta que el video esté listo.
        """
        url = f"https://api.heygen.com/v1/video_status.get?video_id={video_request_id}"
        headers = {
            'X-Api-Key': self.api_key_heygen,
            'Content-Type': 'application/json'
        }

        start_time = time.time()
        
        while True:
            response = requests.get(url, headers=headers)
            data = response.json()
            status = data.get("data", {}).get("status")
            if response.status_code == 200:
                if status == "completed":
                    video_url = data["data"].get("video_url")

                    end_time = time.time()  # Marca o fim do processo
                    total_time = end_time - start_time

                    print("\n✅ El video está listo!")
                    print(f"\n⏱ Tiempo total de generación: {total_time:.2f} segundos")
                    print(f"\n🎥 Link del vídeo: {video_url}")
                    break
                elif status == "failed":
                    print("\n❌ La generación del vídeo ha fallado.")
                    break
                else:
                    print("\n⏳ El video aún se está procesando. Esperando...")
                    time.sleep(30)

    def print_answer(self, answer: str, context_docs: list = None):
        """
        Imprime la respuesta generada por el modelo.
        """
        if context_docs:
            for i, doc in enumerate(context_docs):
                print(f"\n--- Documento {i+1} ---")
                
                score = doc.metadata.get('score', 'Pontuação não disponível')
                print(f"Pontuação de Similaridade: {score}")
    
                # Dividindo o conteúdo em partes menores, como por exemplo linhas
                chunks = doc.page_content.split("\n")  # Separando por linhas
                top_chunks = chunks[:3]  # Exemplo: pegando os 3 primeiros trechos
                
                for chunk in top_chunks:
                    print(chunk)
        print(f"\n\n-> {answer}\n")
    
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
            
            # Divide os documentos em chunks com base na função `split_documents_character`
            # chunks = pipeline.split_documents_by_sentences(context_docs)

            # Calcula a similaridade entre a pergunta e os documentos
            # similarities = pipeline.calculate_similarity(question, context_docs)

            # Adiciona a pontuação de similaridade nos documentos
            # for i, doc in enumerate(context_docs):
            #     doc.metadata['score'] = similarities[i]  # Adiciona a pontuação no metadado

            # Ordena os documentos pela similaridade (pontuação mais alta primeiro)
            # context_docs_sorted = sorted(context_docs, key=lambda x: x.metadata['score'], reverse=True)

            # Exibe os top-n documentos mais relevantes
            # n = 3  # Número de documentos a exibir
            # for i, doc in enumerate(context_docs_sorted[:n]):  # Exibe apenas os top-n documentos
            #     print(f"\n--- Documento {i+1} ---")
            #     print(f"Pontuação de Similaridade: {doc.metadata.get('score', 'Pontuação não disponível')}")
                
            #     # Se a segmentação foi feita corretamente, "doc_chunks" deve ser uma lista com o conteúdo do documento dividido
            #     doc_chunks = chunks[i]  # Associa os chunks ao documento correto
            #     print(f"\nChunks para o Documento {i+1}:")
                
            #     # Exibe os primeiros 3 chunks, ou todos se houver menos de 3
            #     top_chunks = doc_chunks[:3]  # Limita para os 3 primeiros, ou todos se houver menos de 3
            #     for j, chunk in enumerate(top_chunks):
            #         print(f"\nTrecho {j+1}:")
            #         print(chunk)  # Exibe o trecho

            # # Teste de chunks gerados (Exibe os chunks)
            # print("Chunks gerados:")
            # for i, doc in enumerate(chunks[:n]):
            #     print(f"\nDocumento {i+1}:")
            #     for j, chunk in enumerate(doc[:3]):  # Apenas 3 primeiros, ou menos, se houver
            #         print(f"Trecho {j+1}: {chunk}")

            # pipeline.print_answer(answer, context_docs)  # Imprime la respuesta y el contexto recuperado
            print(f"\n\n-> {answer}\n")
            pipeline.generar_video_heygen(texto=answer, avatar_id="d76db879fb5848a4b0729f88499fec48")  # Genera el video con HeyGen
        except Exception as e:
            print(f"\n\n- Error al procesar la pregunta: {e}\n")
        # print(f"- Context_Docs: {context_docs}\n")
