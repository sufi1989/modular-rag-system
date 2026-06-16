import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------- IMPORT RETRIEVER ----------------
from retriever import search

# ---------------- LOAD LLM ----------------
model_name = "HuggingFaceH4/zephyr-7b-beta"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model (this may take time)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1
)

# ---------------- PROMPT ----------------
def build_prompt(context, question):
    return f"""
You are a helpful assistant that answers questions using ONLY the provided context.

If the answer is not in the context, say "I don't know based on the given context."

Context:
{context}

Question:
{question}

Answer clearly and step-by-step:
"""

# ---------------- RAG FUNCTION ----------------
def ask_rag(query, k=5):

    # 1. Retrieve relevant chunks
    results = search(query, k=k)

    # 2. Build context
    context = "\n\n".join([r[0] for r in results])

    # 3. Build prompt
    prompt = build_prompt(context, query)

    # 4. Generate answer
    output = text_gen(prompt)[0]["generated_text"]

    return output, results

# ---------------- TEST LOOP ----------------
if __name__ == "__main__":

    print("RAG SYSTEM READY")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        answer, sources = ask_rag(query, k=5)

        print("\n================ ANSWER ================\n")
        print(answer)

        print("\n================ SOURCES ================\n")

        for i, (text, score) in enumerate(sources):
            print(f"\nRank {i+1} | Score: {score:.4f}")
            print(text[:300])
            print("--------------------------------------------------")