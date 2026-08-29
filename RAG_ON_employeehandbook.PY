from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

#set the ollama model that'll be used to answer questions
Settings.llm = Ollama(model="llama3.2:1b", request_timeout = 120.0) 

#sets the model used for embeddings
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

#points the direction of the documents in question
reader = SimpleDirectoryReader(input_files=[r"C:\Users\Oluwatomisin\Desktop\employee_handbook.txt"]) 

#opens the pdf,  extracts the text and stores in documents 
documents = reader.load_data()

#gives the details on how the chunking would be done
chunk_prescription = SentenceSplitter(chunk_size= 300 , chunk_overlap = 50)


#calls the method(get_nodes...) on the object(chunk_description) for the chunking of the documents 
nodes = chunk_prescription.get_nodes_from_documents(documents)

#converts nodes(chunks) to embeddings and store
index = VectorStoreIndex(nodes)

chat_engine = index.as_chat_engine(similarity_top_k=3)

while True:
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break

    response = chat_engine.chat(user_input) 
    print("Bot: ",response)


