#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install protobuf
# !pip install sentencepiece
# !pip install bitsandbytes
# !pip install torch
# !pip install -U bitsandbytes accelerate
get_ipython().system('pip install llama-cpp-python')


# In[1]:


import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import re
from collections import Counter
import json
load_dotenv(r"../../.env")  
hf_token = os.getenv("hugging_face_token")
hf_token_id = os.getenv("hugging_face_token_fine_tune")
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor, as_completed



# ### Basic Setup

# In[2]:


model_id = "Qwen/Qwen2.5-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Qwen model - apply quantization config
tokenizer = AutoTokenizer.from_pretrained(model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Let it handle device placement automatically
    token=hf_token_id
)
qwen_model.eval()

# Embedding model (for retrieval-based few-shot)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# In[3]:


mistral_model = Llama(
    model_path=r"D:\LLM-Learning-Journey\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # 0 = full CPU mode
    verbose=False
)
mistral_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_id, token=hf_token_id)



# ### Data Setup

# In[4]:


few_shot_examples = [
    ("I was charged extra on my bill",
     "billing issue",
     "Customer mentions extra charge → relates to incorrect billing → billing issue"),

    ("My order arrived late and box was broken",
     "delivery issue",
     "Mentions late delivery and damaged package → logistics/shipping problem → delivery issue"),

    ("The item stopped working in 2 days",
     "product quality",
     "Product failed shortly after use → defect or poor quality → product quality issue"),

    ("Refund is not processed yet",
     "return/refund",
     "Customer waiting for refund → issue in return/refund process → return/refund"),

    ("Money deducted but payment failed",
     "payment issue",
     "Payment deducted but transaction failed → payment system problem → payment issue"),
]

labels = [
    "billing issue",
    "delivery issue",
    "product quality",
    "return/refund",
    "payment issue",
    "customer service"
]

complaints = [
    "I was charged twice for my purchase and customer care is not responding",
    "The delivery was delayed by 5 days and the package was damaged",
    "The product quality is very poor and not worth the price",
    "I want to return the item but the app is not allowing me to",
    "Payment failed but money got deducted from my account"
]


# ### EMBEDDINGS

# In[5]:


embedder = SentenceTransformer('all-MiniLM-L6-v2', token = hf_token)
example_texts = [f"Complaint: {ex[0]} Category: {ex[1]}" for ex in few_shot_examples]
example_embeddings = embedder.encode(example_texts, convert_to_tensor=True)
# query_embedding = embedder.encode("I want to return the item but the app is not allowing me", convert_to_tensor=True)
# cosine_scores = util.cos_sim(query_embedding, example_embeddings)[0]
# cosine_scores_numpy = cosine_scores.cpu().numpy()  # Move to CPU and convert to numpy
# for i, (ex, score) in enumerate(zip(few_shot_examples, cosine_scores_numpy)):
#     print(f"Example {i+1}: '{ex[0]}' (Category: {ex[1]}) - Cosine Similarity: {score:.4f}")


# In[6]:


def few_shot_examples_identifier(query, top_k=3, threshold = 0.5):
    query_embedding = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.cos_sim(query_embedding, example_embeddings)[0]
    cosine_scores_numpy = cosine_scores.cpu().numpy()  # Move to CPU and convert to numpy
    # scored_examples = [(ex, score) for ex, score in zip(few_shot_examples, cosine_scores_numpy) if score >= threshold]
    scored_examples = []
    for i, (ex, score) in enumerate(zip(few_shot_examples, cosine_scores_numpy)):
        # print(f"Example {i+1}: '{ex[0]}' (Category: {ex[1]}) - Cosine Similarity: {score:.4f}")
        if score >= threshold:
            scored_examples.append((ex, score))
    scored_examples.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
    #fallback option if no examples meet the threshold
    if len(scored_examples) == 0:
        arr_position = cosine_scores_numpy.argmax()
        scored_examples.append((few_shot_examples[arr_position], cosine_scores_numpy[arr_position]))
    return scored_examples[:top_k]


# ### GENERATION

# In[7]:


def generate_qwen(prompt, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=151643,
            pad_token_id=151643,
            do_sample=False,  # Disable sampling for deterministic output (use greedy decoding)
            return_dict_in_generate=True,
            # stop_strings=["\n\n", "###", "Explanation"]  # Stop generation at double newline, if needed
        )
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    pre_final_output = decoded[len(prompt):].strip().split("\n\n")[0]  # Get text after prompt and before any double newline
    return pre_final_output

def generate_mistral(prompt, max_new_tokens=40, temperature=0.0):
    """
    Generate text using Mistral GGUF model loaded with llama-cpp-python
    """
    # Generate using llama-cpp-python's API
    response = mistral_model(
        prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=["\n\n", "###", "Explanation"],  # Stop sequences
        echo=False  # Don't include the prompt in the output
    )

    # Extract the generated text
    decoded = response['choices'][0]['text']
    # print(f"Decoded output: '{decoded}'")  # Debug print

    # Clean up: strip whitespace and get text before double newline
    final_output = decoded.strip().split("\n\n")[0]
    return final_output

def generate_reasoning(complaint):
    prompt = f"""
You are analyzing a customer complaint.

Step 1: Identify the MAIN problem.
Step 2: Convert it into a short issue phrase (2–3 words).

Rules:
- Output ONLY the final short phrase
- Do NOT include explanations
- Do NOT include "main issue is"
- Do NOT write full sentences
- Only return the phrase

Examples:

Complaint: I was charged twice for my order  
Answer: charged twice  

Complaint: Delivery was delayed and package was damaged  
Answer: delayed delivery  

Complaint: Payment failed but money got deducted  
Answer: payment failed  

Now process:

Complaint: {complaint}

Answer:
"""
    out = generate_mistral(prompt, max_new_tokens=6)
    return out.split("\n")[0].strip().lower()


# In[8]:


def extract_answer(text):
    answer = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    thought = re.search(r'"thinking"\s*:\s*"([^"]+)"', text)
    final_thought = None
    final_answer = None
    if thought:        
        final_thought = thought.group(1).strip()
    if answer:
        final_answer = answer.group(1).strip()
    return final_answer, final_thought

def extract_json(output):
    match = re.search(r'\{.*?\}', output, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"thinking": "", "answer": "error"}


# ### Zero Shot Prompting

# In[9]:


def zero_shot(complaint):
    scores = {}
    for label in labels:
        prompt = f"""
        <Task>To do the text classification, assign the most appropriate category to the given complaint based on its content. Analyze the complaint carefully and determine which category it best fits into.</Task>
        <label>{label}</label>
        <input>{complaint}</input>

"""
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = qwen_model(**inputs, labels=inputs["input_ids"])
        scores[label] = -outputs.loss.item()
    best_label = max(scores, key=scores.get)
    # softmax for a comparable confidence score
    score_vals = torch.tensor(list(scores.values()))
    confidences = torch.softmax(score_vals, dim=0)
    # best_conf = confidences[list(scores.keys()).index(best_label)].item()
    return best_label, ""


# ### Few Shot Prompting

# In[10]:


def few_shot(complaint):
    examples = few_shot_examples_identifier(complaint)
    ex_prompt = ""
    for example in examples:
        ex_prompt += f"""
        <example_input>{example[0][0]}</example_input>
        <example_output>{example[0][1]}</example_output>
        """
    scores = {}
    for label in labels:
        prompt = f"""
        <Task>To do the text classification, assign the most appropriate category to the given complaint based on its content. Analyze the complaint carefully and determine which category it best fits into.</Task>
        <label>{label}</label>
        <input>{complaint}</input>
        {ex_prompt}
"""
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = qwen_model(**inputs, labels=inputs["input_ids"])
        scores[label] = -outputs.loss.item()
    best_label = max(scores, key=scores.get)
    # softmax for a comparable confidence score
    score_vals = torch.tensor(list(scores.values()))
    confidences = torch.softmax(score_vals, dim=0)
    # best_conf = confidences[list(scores.keys()).index(best_label)].item()
    return best_label, ""


# ### ZERO-SHOT CoT

# In[11]:


def zero_shot_cot(complaint):
    prompt = f"""
You are a classification assistant.

Task:
Classify the complaint into one category.

Follow this EXACT thinking process:
1. Identify the key issue in the complaint
2. Map the issue to the closest category
3. Give the final category

Return ONLY ONE valid JSON object and STOP immediately.
DO NOT generate anything after the JSON.

STRICT FORMAT:
{{
  "thinking": "Issue identified: <issue> → mapped to: <category>",
  "answer": "<category>"
}}

Categories:
- billing issue → problems with charges, invoices, extra money
- delivery issue → late delivery, damaged package, shipping problems
- product quality → defective or poor quality product
- return/refund → refund not processed, return issues
- payment issue → payment failed, money deducted
- customer service → support not responding or poor service

Complaint: {complaint}

Final Answer (JSON only):
"""
    return generate_qwen(prompt)


# ### Few-Shot CoT

# In[12]:


def few_shot_cot(complaint):
    examples = few_shot_examples_identifier(complaint)
    ex_prompt = ""
    for example in examples:
        ex_prompt += f"""
            ### Example
            Complaint: {example[0][0]}
            Output: {{"thinking": "{example[0][2]}", "answer": "{example[0][1]}"}}
            """
    prompt = f"""
You are a classification assistant.

Return ONLY ONE valid JSON object and STOP immediately.

DO NOT generate anything after the JSON.

Format:
{{"thinking": "...", "answer": "..."}}

Categories: ["billing issue", "delivery issue", "product quality", "return/refund", "payment issue", "customer service"]

{ex_prompt}

### Now classify:

Complaint: {complaint}

Final Answer (JSON only):
"""

    return generate_qwen(prompt)


# ### Auto COT

# In[13]:


def auto_cot(complaint):
    signal = generate_reasoning(complaint)
    # print(f"Identified signal: '{signal}'")
    prompt = f"""
Classify the issue into ONE category.

Issue: {signal}

Categories:
- billing issue → problems with charges, invoices, extra money
- delivery issue → late delivery, damaged package, shipping problems
- product quality → defective or poor quality product
- return/refund → unable to return or refund issues
- payment issue → payment failed or money deducted
- customer service → support not responding

Rules:
- Use definitions ONLY for reasoning
- Answer MUST be EXACT category name (no extra text)
- Do NOT include explanations in answer
- Output must be in JSON format with "thinking" and "answer" fields

Output format:
{{
  "thinking": "<issue> → <category>",
  "answer": "<category>"
}}

Final answer (JSON only):
"""
    raw = generate_mistral(prompt, max_new_tokens=100)

    return raw


# ### AUTO-CoT + SC

# In[14]:


def auto_cot_sc(complaint, n=5):
    answers = []

    for _ in range(n):
        out = auto_cot(complaint)
        ans = extract_answer(out)
        if ans:
            answers.append(ans)

    return Counter(answers).most_common(1)[0][0] if answers else "unknown"


# ### ToT

# In[ ]:


def generate_thoughts(complaint, k=3):
    thoughts = []

    for _ in range(k):
        prompt = f"""
You are a classification assistant.

Generate ONE reasoning path and return ONLY JSON.

Task:
- Identify the main issue
- Map it to ONE category

Categories:
billing issue, delivery issue, product quality, return/refund, payment issue, customer service

Rules:
- Output MUST be valid JSON
- Do NOT include any text before or after JSON
- thinking must be short: issue → category
- answer must be ONLY category name

Format:
{{"thinking": "...", "answer": "..."}}

Complaint: {complaint}

JSON:
"""
        out = generate_mistral(prompt, max_new_tokens=80, temperature=0.9)
        thought = extract_json(out)
        thoughts.append(thought)
        # print(f"Generated thought: {thought}")

    return thoughts


# In[ ]:


def evaluate_thoughts(thoughts, complaint):
    scored = []

    for t in thoughts:
        prompt = f"""
You are evaluating a classification result.

Complaint:
{complaint}

Prediction:
Thinking: {t['thinking']}
Answer: {t['answer']}

Categories:
- billing issue → problems with charges, invoices, extra money
- delivery issue → late delivery, damaged package
- product quality → defective or poor quality product
- return/refund → unable to return or refund issues
- payment issue → payment failed or money deducted
- customer service → support not responding

Evaluation Rules:
- Score 10 if category perfectly matches complaint
- Score 5 if partially correct
- Score 0 if wrong category

Return ONLY a number: 0, 5, or 10
Do NOT explain.

Score:
"""
        out = generate_mistral(prompt, max_new_tokens=5, temperature=0.7)

        try:
            score = int(out.strip()[0])
        except:
            score = 0

        scored.append((t, score))
        # print(f"Thought: {t} → Score: {score}")
    return max(scored, key=lambda x: x[1])[0]


# In[17]:


def tot(complaint):
    thoughts = generate_thoughts(complaint, k=3)
    best = evaluate_thoughts(thoughts, complaint=complaint)

    return best  # already JSON


# ### TOT + SC

# In[18]:


def tot_sc(complaint, n=5):
    answers = []

    for _ in range(n):
        out = generate_qwen(f"""
Analyze step by step:

Complaint: {complaint}

Output:
{{"thinking": "...", "answer": "..."}}
""")

        ans = extract_answer(out)
        if ans:
            answers.append(ans)

    return Counter(answers).most_common(1)[0][0] if answers else "unknown"


# In[30]:


rows = []
for i, complaint in enumerate(complaints):
   zs = zero_shot(complaint)
   fs = few_shot(complaint)
   fsc = extract_answer(few_shot_cot(complaint))
   zsc = extract_answer(zero_shot_cot(complaint))
   ac = extract_answer(auto_cot(complaint))
   ac_sc = auto_cot_sc(complaint)
   t = tot(complaint)
   tot_res = (t['answer'] if isinstance(t, dict) else "", t["thinking"] if isinstance(t, dict) else "error")
   tot_sc_res = tot_sc(complaint)
   print(f"Complaint {i + 1}: {complaint}")
   print("Zero-Shot:")
   print(zs)
   print("Few-Shot:")
   print(fs)
   print("Few-Shot CoT:")
   print(fsc)
   print("Zero-Shot CoT:")
   print(zsc)
   print("Auto CoT:")
   print(ac)
   print("Auto CoT with Self-Consistency:")
   print(ac_sc)
   print("TOT:")
   print(tot_res)
   print("TOT with Self-Consistency:")
   print(tot_sc_res)
   print("-" * 50)
   row = {
       "complaint": complaint,
       "zero_shot": zs[0],
       "few_shot": fs[0],
       "few_shot_cot": fsc,
       "zero_shot_cot": zsc,
       "auto_cot": ac,
       "auto_cot_sc": ac_sc,
       "tot": tot_res,
       "tot_sc": tot_sc_res,
       "Zero_shot_COT_thinking": zsc[1],
       "few_shot_cot_thinking": fsc[1],
       "auto_cot_thinking": ac[1],
       "auto_cot_sc_thinking": ac_sc[1],
       "tot_thinking": tot_res[1],
       "tot_sc_thinking": tot_sc_res[1]
   }
   rows.append(row)
   print(row)
   print("-" * 50)
df = pd.DataFrame(rows)
print(df)
df.to_excel("Prompting_techniques_classification_results.xlsx", index=False)
   # print(auto_cot(complaint))


# In[29]:


for i, complaint in enumerate(complaints):
    print(complaints[i])
    t = tot(complaint)
    t = (t['answer'] if isinstance(t, dict) else "", t["thinking"] if isinstance(t, dict) else "error")
    print(t)
    print("-" * 50)


# ### Testing purpose  I have added this

# In[ ]:


def process_complaint(complaint, i):
    print(f"Complaint {i + 1}: {complaint}")

    zs = zero_shot(complaint)
    fs = few_shot(complaint)

    fsc = extract_answer(few_shot_cot(complaint))
    zsc = extract_answer(zero_shot_cot(complaint))
    ac = extract_answer(auto_cot(complaint))
    ac_sc = extract_answer(auto_cot_sc(complaint))
    t = tot(complaint)
    t = (t['answer'] if isinstance(t, dict) else "", t[1] if isinstance(t, dict) else "error")
    t_sc = extract_answer(tot_sc(complaint))

    print("Zero-Shot:", zs)
    print("Few-Shot:", fs)
    print("Few-Shot CoT:", fsc)
    print("Zero-Shot CoT:", zsc)
    print("Auto CoT:", ac)
    print("Auto CoT SC:", ac_sc)
    print("TOT:", t)
    print("TOT SC:", t_sc)
    print("-" * 50)

    return {
        "complaint": complaint,
        "zero_shot": zs[0],
        "few_shot": fs[0],
        "few_shot_cot": fsc[0],
        "zero_shot_cot": zsc[0],
        "auto_cot": ac[0],
        "auto_cot_sc": ac_sc[0],
        "tot": t[0],
        "tot_sc": t_sc[0],
        "Zero_shot_COT_thinking": zsc[1],
        "few_shot_cot_thinking": fsc[1],
        "auto_cot_thinking": ac[1],
        "auto_cot_sc_thinking": ac_sc[1],
        "tot_thinking": t[1],
        "tot_sc_thinking": t_sc[1]
    }


# In[ ]:


rows = []

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(process_complaint, complaint, i)
        for i, complaint in enumerate(complaints)
    ]

    for future in as_completed(futures):
        rows.append(future.result())

df = pd.DataFrame(rows)
print(df)
df.to_excel("Prompting_techniques_classification_results.xlsx", index=False)


# In[27]:


for i, complaint in enumerate(complaints):
    print(complaints[i])
    print(tot(complaints[i]))
    print("-" * 50)
# complaint = complaints[0]
# tot(complaint)


# In[26]:


a = {"thinking": "payment failed → payment issue", "answer": "payment issue"}
str(a)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




