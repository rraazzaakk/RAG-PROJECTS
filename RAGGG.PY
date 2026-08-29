from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from sentence_transformers import CrossEncoder

Settings.llm = Ollama(model ="llama3.2:1b",request_timeout= 1200)
Settings.embed_model= OllamaEmbedding(model_name= "nomic-embed-text")

dataset = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")

train_data = dataset["train"]

unique_labels = set(train_data["label"]) 

target_labels = [
    "Governing Law",
    "Termination For Convenience",
    "Cap On Liability",
    "Uncapped Liability",
    "Non-Compete",
]

filtered = train_data.filter(lambda row: row["label"] in target_labels)

small_list = []

for label in target_labels:
    matches = filtered.filter(lambda row: row["label"] == label)
    example_files = sorted(set(matches["file_name"]))[:2]
    
    small_list.extend(example_files)
    # print(label, "->", example_files)
    
deduplicated_list = list(set(small_list))
filtered_final = filtered.filter(lambda row: row["file_name"] in deduplicated_list)

documents = []

for row in filtered_final:
    docs=Document(
        text = row["clause"],
        metadata = {"file name" : row["file_name"] , "label" : row["label"] }
        
    )
    documents.append(docs)

index = VectorStoreIndex.from_documents(documents)

chat_engine = index.as_chat_engine(similarity_top_k = 10)

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
while True:
    user_input = input("you: ")
    
    if user_input.lower() == "exit":
        break
    response = chat_engine.chat(user_input)

    q_c = []

    for context in response.source_nodes:
        q_c.append((user_input , context.node.text))

    score = model.predict(q_c)

    paired = list(zip(score , response.source_nodes))
    paired.sort(reverse = True , key=lambda pair :pair[0])

    for s, node in paired[:3]:
        print(s , node.node.metadata["label"], node.node.text[:300])