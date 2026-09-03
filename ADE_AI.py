import requests
from bs4 import BeautifulSoup
import time
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import streamlit as st
import pypdf
from io import BytesIO
import os

gemini_api_key = os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

Settings.llm = Groq(model="openai/gpt-oss-120b", api_key=groq_api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview", api_key=gemini_api_key)


def scrape_page(url):
    web_page = requests.get(url)
    soup = BeautifulSoup(web_page.text, "html.parser")

    for tag in soup.find_all(["nav", "footer"]):
        tag.decompose()

    clean_text = soup.get_text(separator=" ", strip=True)
    cleaner_text = clean_text.split("View Qualified Candidates")[0]
    return cleaner_text

def remove_spam(text):
    words = text.split()
    
    safe_words = []  
    
    spam_keywords = ["bet", "bahis", "casibom", "giriş", "escort", "casino", "iptv", "taraftarium", "kaufen", "follower","tür","ö"]
    for word in words:
        is_spam = False
        for keyword in spam_keywords:
            if keyword in word.lower():
                is_spam = True
        if is_spam == False:
            safe_words.append(word)
    
    result = ' '.join(safe_words)
    return result

def scrape_pdf(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    reader = pypdf.PdfReader(pdf_file)
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    return text

pdf_url = "https://adelekeuniversity.edu.ng/wp-content/uploads/2026/07/AU-2026-2027-SUMMARY-BILL.pdf"
pdf_text = scrape_pdf(pdf_url)
finalpdf_txt =  " ".join(pdf_text.split())

@st.cache_resource
def build_index():
    documents = []
    unique_urls = [
        "https://adelekeuniversity.edu.ng/about-au/",
        "https://adelekeuniversity.edu.ng/history/",
        "https://adelekeuniversity.edu.ng/undergraduate-programs/",
        "https://adelekeuniversity.edu.ng/postgraduate-programs/",
        "https://adelekeuniversity.edu.ng/faculties/",
        "https://adelekeuniversity.edu.ng/faculty-of-art/",
        "https://adelekeuniversity.edu.ng/faculty-of-science/",
        "https://adelekeuniversity.edu.ng/faculty-of-engineering/",
        "https://adelekeuniversity.edu.ng/faculty-of-law/",
        "https://adelekeuniversity.edu.ng/faculty-of-basic-medical-science/",
        "https://adelekeuniversity.edu.ng/faculty-of-business-social-science/",
        "https://adelekeuniversity.edu.ng/tuition-fees/",
        "https://adelekeuniversity.edu.ng/scholarship/",
        "https://adelekeuniversity.edu.ng/campus-and-facilities/",
    ]


    for url in unique_urls:
        scraped = scrape_page(url)
        print(f"Scraping: {url}")
        removedspam = remove_spam(scraped)
        
        docs = Document(
            text=removedspam,
            metadata={"source_url": url}
        )
        documents.append(docs)
        
        time.sleep(1)
        
        
    pdf_doc = Document(
        text=finalpdf_txt,
        metadata={"source_url": pdf_url}
    )
    documents.append(pdf_doc)

    index = VectorStoreIndex.from_documents(documents, show_progress = True)
    return index

index = build_index()
system_prompt = (
    "You are Ade, the official AI assistant for Adeleke University. "
    "You are warm, encouraging, and genuinely helpful, like a knowledgeable senior student showing "
    "a new student around. Your job is to help prospective and current students with questions about "
    "programs, faculties, admissions, tuition fees, scholarships, and campus life. "
    "Always answer using only the information provided to you, but speak naturally, as if you simply "
    "know this information yourself. Never say phrases like 'according to the document', 'based on the "
    "provided text', or 'the documents state' - just answer directly, the way a helpful person would. "
    "If you don't have the specific information needed to answer a question, say so honestly, and where "
    "possible, point the person to the right university contact or department instead of guessing. "
    "Keep your tone friendly and conversational, not overly formal, and feel free to organize longer "
    "answers with headings or bullet points to make them easy to read."
)

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(similarity_top_k=15, system_prompt=system_prompt)

chat_engine = st.session_state.chat_engine
st.title("Adeleke University AI Assistant")
st.write("Ask a question about programs, faculties, tuition, scholarships, and campus life at Adeleke University.")

# Keep conversation history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redisplay the full conversation history so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box, sits at the bottom, submits on Enter
user_input = st.chat_input("Ask ADE a question...")

if user_input:
    # Show and store the user's message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get and show ADE's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat_engine.chat(user_input)
                answer = str(response)
            except Exception as e:
                answer = f"Something went wrong: {e}"
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
