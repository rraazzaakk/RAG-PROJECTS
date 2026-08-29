from pypdf import PdfReader
import chromadb
import ollama

reader = PdfReader(r"C:\Users\Oluwatomisin\Desktop\employee_handbook.pdf")
text = ""
for page in reader.pages:
    text+= page.extract_text()
    
def chunk_texts(text , chunk_size = 300):
    chunks =[]
    for i in range(0,len(text),chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

chunks = chunk_texts(text ,chunk_size=300)
client = chromadb.PersistentClient(path="./my_chroma_db")

collection = client.get_or_create_collection(name="ex_chunks")

collection.add(
    documents= chunks,
    ids=[str(i) for i in range(len(chunks))]
    )

messages = []
while True:
    user_input = input("you :")
    if user_input.lower() == "exit":
        break

    result = collection.query(
    query_texts=[user_input],
    n_results=1
    )
    context =  result["documents"][0][0]
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided document."  

    {context}
    {user_input}
    """
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model="llama3.2",
        messages=messages
    )

    reply = response["message"]["content"]
    print("Bot:", reply)

    messages.append({"role": "assistant", "content": reply})
    
    print( context =  result["documents"][0][0])
    
        