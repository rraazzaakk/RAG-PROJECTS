RAG Projects

A collection of Retrieval-Augmented Generation (RAG) projects, built progressively from first principles to increasingly complex, real-world applications. Each project was built, tested, and debugged end-to-end. This includes diagnosing real bugs, hitting real API constraints, and documenting genuine limitations rather than just showcasing what worked.

Projects
ADE - Adeleke University AI Assistant (ADE_AI.py)

A chatbot that scrapes a real university website and a linked PDF fee schedule to answer prospective and current student questions about programs, admissions, tuition, scholarships, and campus life.

Highlights:

Custom web scraper (requests + BeautifulSoup) with spam-content filtering. The source site had gambling and escort spam, plus fake-software spam, injected into multiple pages, requiring both structural cleanup (removing duplicated nav/footer) and keyword-based filtering.
PDF extraction (pypdf) to pull structured tuition data out of a downloadable fee schedule, verified for accuracy against the source document
Persistent chat memory across a Streamlit session (st.session_state)
Explicit anti-hallucination guardrails in the system prompt, added after discovering the model would occasionally invent plausible-sounding but fake contact details
Switched from Gemini to Groq mid-project after hitting a 20-requests-per-day free-tier ceiling. Documented real free-tier constraints for both providers, including rate limits versus token limits.
CUAD Legal Clause Reranker Comparison (app.py)

Compares plain embedding retrieval against cross-encoder reranking on real legal contract clauses from the CUAD (Contract Understanding Atticus Dataset) dataset, specifically testing whether reranking improves accuracy on a hard case: distinguishing "Cap On Liability" from "Uncapped Liability" clauses.

Highlights:

Reproducible dataset filtering, including a fix for a real bug where unordered set() selection made results non-deterministic across runs
Cross-encoder reranking (sentence-transformers) layered on top of baseline vector search
Documented a genuine retrieval-miss finding: reranking can only reorder what's already retrieved. It cannot compensate for content that never made it into the candidate pool.
Multilingual Health Advisory Bot (RAG_ON_multiligual.py)

A RAG assistant over a 4-language (English, Spanish, French, Italian) health guidance corpus, designed to answer in whichever language the user selects, regardless of what language the question was asked in.

Highlights:

Tested and found that similarity_top_k directly affects answer reliability in multilingual retrieval. A wider top_k caused the model to occasionally mix languages or hallucinate facts when reconciling near-duplicate content across 4 languages, while a narrower top_k fixed it.
Manual RAG Pipeline (RAG_ON_employeehandbook.PY)

A RAG pipeline built entirely by hand, with no framework, over a company handbook PDF. Covers chunking, embedding, ChromaDB storage and retrieval, and multi-turn conversation, using local Ollama.

Highlights:

Diagnosed and fixed the ChromaDB .add() silent-duplicate-id bug
Compared fixed-size chunking against sentence-boundary-aware chunking
ChromaDB Exercises (RAGEXCERCISE.py)

Smaller, focused exercises on metadata tagging and debugging common chunking and retrieval bugs.

Tech Stack

Python, LlamaIndex, ChromaDB, Ollama, Google Gemini, Groq, Streamlit, BeautifulSoup, pypdf, sentence-transformers, Hugging Face datasets

Real Constraints Encountered

Every project here hit genuine, documented limitations rather than working perfectly on the first try:

Free-tier API rate limits (Gemini: 20 generation requests per day; Groq: 200,000 tokens per day)
Retrieval gaps that persisted even after tuning top_k and chunk overlap
Occasional model hallucination, addressed with explicit prompt guardrails
Windows-specific path-length bugs when caching Hugging Face datasets
Environment differences between local development and Streamlit Cloud deployment, such as NLTK filesystem permissions
Setup

Each project has its own dependencies listed in requirements.txt. Most require an API key (Gemini and/or Groq) set as an environment variable or Streamlit secret. See the comments at the top of each script for specifics.
