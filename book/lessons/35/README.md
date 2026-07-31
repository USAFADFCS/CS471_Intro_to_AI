

Lesson 35 — Retrieval-Augmented Generation (RAG) 

Explain the RAG pipeline (retrieve → augment → generate)  

Describe how external data improves model reliability  

Analyze tradeoffs between base models and RAG systems  


## L35 Retrieval-Augmented Generation
**Objectives:** Explain vector embeddings and database search; design document chunking strategies; build a RAG pipeline to ground LLM outputs in retrieved operational data.

*   Diagram the RAG pipeline, explaining how textual Air Force Instructions (AFIs) are converted into vector embeddings and stored for rapid semantic search.
*   Build a basic RAG system, testing different document chunking strategies on unclassified aircraft maintenance logs to observe how the context window size impacts the quality of generated troubleshooting steps.
*   Analyze the trade-offs of a RAG system when retrieved intelligence reports contain conflicting capabilities of an adversary. Design a system prompt to handle this ambiguity, cite specific sources, and express calculated uncertainty.
