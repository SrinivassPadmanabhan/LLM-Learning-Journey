# LLM Scientist roadmap 

1. LLM Architecture --> Architectural overview, Tokenization, Attention Mechanisms, Sampling Techniques
2. Pre Training Models --> Data Preparation, Distribute Training, Training Optimisation, Monitoring
3. Post Training Models --> Storage & Chat templates, Synthetic Data genration, Data enhancement, Quality filtering
4. Supervised Fine tuning  --> Training technique, Training Parameters, Distributed Training, Monitoring
5. Preference Alignment --> Rejection Sampling, Direct preference optimization, reward model, reinforcement learning
6. Evaluation --> Automated benchmarks, human evaluation, model based evaluation, Feedback signals
7. Quantization --> Base technique, GGUF and llama.cpp, GPTQ & AWQ, SmoothQuant & Zero QUant
8. New Trend --> Model merging, Multimodal models, Test - Time Compute
![alt text](Images/LLM_Scientist_Roadmap.png)


# LLM Engineer Roadmap

1. Running LLMs --> LLM APIs, Open-Source LLMs, Prompt Engineering, Structuring Outputs
2. Building a vector storage --> Ingesting documents, splitting documents, embedding models, vector databases
3. RAG --> Orchestrators, Retrievers, memory, evaluation
4. Advanced RAG --> Query construction, Agents & Tools, Post-processing, Program LLMs
5. Agents --> Agent Fundamentals, Agent Frameworks, Multi Agents
6. Inference optimization --> Flash attention, Key-Value Cache, Speculative decoding
7. Deploying LLMs --> Local deployment, Demo Deployment, server deployment, edge deployment
8. Securing LLMs --> Prompt hacking, backdoors, defensive measures

![alt text](Images/LLM_Engineering_Roadmap.png)

Link which i used to get the information is: https://github.com/mlabonne/llm-course/tree/main



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# What is there under the LLMs
### Each data flows through all this step in an LLM:
1. Embedding
2. Key
3. Query
4. Value
5. Output
6. Up-projection
7. Down-projection
8. Unembedding

# Encoder and Decoder in the LLM models:

1. Encoder = a reader. Reads the entire document first, understands all of it, then answers questions about it. Sees every word — past, present, future in the sentence — simultaneously. Best at understanding. (Bi-directional)
Example: Milk sales dropped after bank holiday weekend.
in this "dropped" is inflenced in the context of "bank holiday" and "weekend." Every word influences every other word. This is bidirectional attention — the key feature of encoders. It produce a dense vector embedding — a numerical representation of meaning. It is that embedding vector which is then used in similarity search (comparing vectors in a database using cosine similarity). The encoder produces the input to the search; the vector database performs the search

2. Decoder = a writer. Writes one word at a time, left to right. At each step it can only see what it has written so far — never the future. Best at generating. (Uni-directional from left to right).
Demand for fresh produce rose sharply this week.  Each new word is chosen based only on what they have written so far

3. Encoder-Decoder = a translator. First reads and fully understands the source (encoder), then writes the output word by word (decoder). Best at transforming one sequence into another. 
Example: Imagine translating a supplier contract from French to English. First, a bilingual analyst reads and fully understands the entire French document (encoder). Then, a writer produces the English version word by word, guided by the encoder's understanding (decoder). Two separate jobs, working together.

For the Retrieval augment generation where we need the understanding and the generation as well  
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Questions:
1. When should I use an LLM vs a traditional ML model? (2–3 sentences)
Ans: LLMs are best suited for tasks involving unstructured text—such as generation, extraction, Q&A, and agent-like reasoning—where context and language understanding are key. Traditional ML models are better for structured, tabular problems with well-defined features, like prediction, classification, and forecasting. Both can be probabilistic, but ML models are typically more efficient and interpretable, while LLMs offer flexibility for complex language tasks.

2. The one-sentence encoder vs decoder definition:
Ans: Encoder-only models (BERT) read the entire input bidirectionally to build a deep contextual representation — best for understanding tasks like classification and search — while decoder-only models (GPT, Claude, LLaMA) generate text autoregressively by predicting one token at a time, seeing only previous tokens — best for generation and conversation

