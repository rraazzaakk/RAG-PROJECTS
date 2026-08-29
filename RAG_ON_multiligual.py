from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="llama3.2" , request_timeout = 5001.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

reader = SimpleDirectoryReader(input_files=[r"C:\Users\Oluwatomisin\Desktop\multilingual_health_corpus.pdf"])

documents = reader.load_data()

chunk_prescription = SentenceSplitter(chunk_size= 300 , chunk_overlap= 50)

nodes = chunk_prescription.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)

supported_languages = {
    "1": "English",
    "2": "Spanish",
    "3": "French",
    "4": "Italian"
}
while True:
    print("Choose your language:")
    for key, language in supported_languages.items():
        print(f"{key}. {language}")

    choice = input("Enter a number (1-4): ")

    if choice in supported_languages:
        break
        
    else:
        print("NOT A VALID CHOICE\n\n PICK A VALID CHOICE")

chosen_language = supported_languages.get(choice)

chat_engine = index.as_chat_engine(
        similarity_top_k =2, 
        System_prompt = (f"You are a multilingual health information assistant. "
        f"Answer ONLY in {chosen_language}, regardless of what language the retrieved information is written in. "
        f"Translate the relevant facts into {chosen_language} if needed, but keep the answer accurate to the source content. "
        f"Do not mix languages in your response, and do not quote other-language text directly."
        )
)
while True :
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break
    
    response = chat_engine.chat(user_input)
    
    print("Bot: ",response)
