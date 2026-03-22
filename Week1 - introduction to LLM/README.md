
# Day 1
## LLM Scientist roadmap 

1. LLM Architecture --> Architectural overview, Tokenization, Attention Mechanisms, Sampling Techniques
2. Pre Training Models --> Data Preparation, Distribute Training, Training Optimisation, Monitoring
3. Post Training Models --> Storage & Chat templates, Synthetic Data genration, Data enhancement, Quality filtering
4. Supervised Fine tuning  --> Training technique, Training Parameters, Distributed Training, Monitoring
5. Preference Alignment --> Rejection Sampling, Direct preference optimization, reward model, reinforcement learning
6. Evaluation --> Automated benchmarks, human evaluation, model based evaluation, Feedback signals
7. Quantization --> Base technique, GGUF and llama.cpp, GPTQ & AWQ, SmoothQuant & Zero QUant
8. New Trend --> Model merging, Multimodal models, Test - Time Compute
![alt text](Images/LLM_Scientist_Roadmap.png)


## LLM Engineer Roadmap

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
## What is there under the LLMs
### Each data flows through all this step in an LLM:
1. Embedding
2. Key
3. Query
4. Value
5. Output
6. Up-projection
7. Down-projection
8. Unembedding

## Encoder and Decoder in the LLM models:

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

# Day 2 module

How to transformers evole over time:

![alt text](image.png)

There are 2 main approaches for the Transformers they are
1. Masking Language model: Uses the encoder technique which tries to mask any one word in the sentence and tries to predict that word using the bi directional technique( encoder approach)
2. Causal Language model: used the decoder technique which tries to predict the next token based on the previous token in the sequence. 

 ![alt text](Images/Encoder_Decoder_Architecture.png)


Transformer architecture: Input → [Multi-Head Attention → Add & Norm] → [Feed Forward → Add & Norm] → Output
## Encoder architecture (Left side)

Encoder: Input → Embedding → Encoder layers → Output (understanding)

1. Input embedding + postion vector: COnversion of the words to embedding and the position vector added.
2. Multi head attention: In this there is also a multi head attention is happening which reads the text from the bi-directional.
3. Add & Norm: This happens to restrict it from exploding the values normalisation happens and Residual is added to handle the error
4. Feed forward: Process each token further
5. Repeat N times (Nx): Repeat this process multiple times.
Used for: Classification, Search, Similarity


## Decode architecture (Right Side):
Previous tokens --> Decoder --> Next Token prediction

1. Masked Multi-Head Attention: Based on the past words it tries to predict the future words without seeing it.
2. Cross attention layer (Which usually decoder only model will not have):  attends to the encoder output (K and V come from encoder, Q comes from decoder)
3. Feed Forward: Process each token further
4. Add & Norm: This happens to restrict it from exploding the values normalisation happens and Residual is added to handle the error
5. Linear + Softmax (TOP PART): Converts output → probabilities and Predicts next word
6. Repeat N time (Nx): first 3 steps to generate the desired which is needed.

| Feature   | BERT (Encoder)         | GPT (Decoder)            |
| --------- | ---------------------- | ------------------------ |
| Attention | Full (bidirectional)   | Masked (only past)       |
| Purpose   | Understand             | Generate                 |
| Output    | Embeddings             | Next word                |
| Use case  | Classification, search | Chat, content generation |


# Day 3 module
![alt text](Images/Encoder_Decoder_Architecture.png)
History of the Large language models AI.
![alt text](Images/History_of_LLM.png)
1. Non transformer model --> This was built before 2017 some of the models are Bag of words, Word2Vec, Attention
2. Encoder Only model --> Bert, Distilbert, RoBERTA
3. Decoder only model --> GPT models
4. Encoder + Decoder models --> T5, Switch, Flan-T5

a. Bag of Words: It considers language to be nothing more than an almost literal bag-of-words, and ignores the semantic nature or meaning of text.
b. Word 2 Vec: 
    It could capture meaning of words in vectore embeddings through neural networks. This embedding tries to capture the meaning of the words. This embedding has generate values from -1 to 1.
    The number of the dimensions is generally a fixed size. In practical we will not understand what is the properties it represent in the real world. As they are learned through complex mathematical calculations.
    Inputs --> Token (which splits the words to tokens may be same word can be a token or it can get splitted in the multiple token usually it takes the greedy approach) --> embedding --> average of the token embedding represents the sentence token.
    It creates the static embedding for example it will take the same embedding for the bank when it is used in the river bank and even the same finance bank

Attention allows the model to focus on part pf the input that are relevant to one another

Attend to each signal, relevant to each other and amplify their signal

INitially the contexually embedding which averages the total embedding which reduce becomes difficult for the decoder to understand the sequence but soon aftermath attention mechanism came into picture which reads each words embedding and the reads which embedding will be closer to the other in the decoder step.


initially, in the encoder what is happening is reading the word and trying to predict and the next word  post that embedding has been passed to the decoder where the cross-attention mechanism happen which tries to translate the words. THis made the model very powerful in the translation task but it becomes difficult in the text classification.

Post that the BERT model was created which uses the encoder only in which mask language technique to understand the word better
post that GPT uses the decoder which uses the next word prediction based on the temperature the cretivity increases/ decreases

self attention  which comprises of the  Relevance scoring and the combining information

In feedforward network there is a concept of Mixture of experts (Sparse model) when one expert is activated other expert become deactivated, these experts are good at converting the words to vectors. To go to the correct experts there is a router is needed which reads the input sends it to the correct experts.

Question:
what is the difference between pipeline() and AutoModel?
Ans: pipeline() is a high-level abstraction that simplifies model usage by handling tokenization, inference, and decoding internally. In contrast, AutoModel with AutoTokenizer provides low-level control over each step, which is essential for customization, debugging, and production use cases like streaming or custom decoding.

