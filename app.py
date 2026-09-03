import os
os.environ["HF_HOME"] = "C:\\hfcache"

from datasets import load_dataset
from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from sentence_transformers import CrossEncoder
import streamlit as st


api_key = os.environ.get("GEMINI_API_KEY")

Settings.llm = GoogleGenAI(model="gemini-3.6-flash", api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview", api_key=api_key)


@st.cache_resource
def build_index():
    dataset = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_clause_classification")
    train_data = dataset["train"]

    target_labels = [
        "Governing Law",
        "Termination For Convenience",
        "Cap On Liability",
        "Uncapped Liability",
        "Non-Compete",
    ]

    filtered = train_data.filter(lambda row: row["label"] in target_labels)
    
    balanced_rows = []
    for label in target_labels:
        matches = filtered.filter(lambda row: row["label"] == label)
        rows_for_label = matches.select(range(min(18, len(matches))))
        balanced_rows.extend(rows_for_label)

    
    documents = []
    for row in balanced_rows:
        docs = Document(
            text=row["clause"],
            metadata={"file_name": row["file_name"], "label": row["label"]}
        )
        documents.append(docs)

    index = VectorStoreIndex.from_documents(documents)
    return index

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')


index = build_index()
chat_engine = index.as_chat_engine(similarity_top_k=20)
reranker_model = load_reranker()


st.title("Smart Contract Search")
st.write("Ask a question about the loaded contract clauses (Governing Law, Termination For Convenience, Cap On Liability, Uncapped Liability, Non-Compete).")

st.write("**Try an example question:**")
col1, col2, col3 = st.columns(3)
example_clicked = None
with col1:
    if st.button("Is there a limit on liability?"):
        example_clicked = "Is there a limit on liability?"
with col2:
    if st.button("Is there a non-compete clause?"):
        example_clicked = "Is there a non-compete clause?"
with col3:
    if st.button("What governing law applies?"):
        example_clicked = "What governing law applies?"

user_input = st.text_input("Or type your own question:", value=example_clicked if example_clicked else "")

ask_clicked = st.button("Ask")

if (ask_clicked or example_clicked) and user_input:
    with st.spinner("Retrieving and reranking..."):
        try:
            response = chat_engine.chat(user_input)
        
        except Exception as e:
                st.error("Something went wrong retrieving an answer. Please try again in a moment.")
                st.stop()

        st.subheader("Answer")
        st.write(str(response))
        st.divider()
        q_c = []
        for context in response.source_nodes:
            q_c.append((user_input, context.node.text))

        scores = reranker_model.predict(q_c)

        paired = list(zip(scores, response.source_nodes))
        paired.sort(reverse=True, key=lambda pair: pair[0])

        with st.expander("Top 3 Reranked Clauses"):
            for s, node in paired[:3]:
                st.write(f"**Score:** {s:.3f} | **Label:** {node.node.metadata['label']}")
                st.write(node.node.text[:300])
                st.write("---")