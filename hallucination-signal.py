from llama_cpp import Llama
import numpy as np

llm = Llama(model_path="llama.cpp/models/llama-2-7b-chat.ggmlv3.q4_0.bin",logits_all=True,verbose=False)

def get_llama_logits(text):
    output = llm(text, temperature=0, max_tokens=250, logprobs=1)
    choice = output['choices'][0]
    text = choice['text']
    token_logprobs = choice['logprobs']['token_logprobs']
    return text, token_logprobs

def convert_logprob_to_prob(log_probs):
    return [np.exp(log_prob) for log_prob in log_probs]

def average_token_probability(token_probs):
    return np.mean(token_probs)

def normalized_product_token_probability(token_probs):
    product_probs = np.prod(token_probs)
    return product_probs ** (1 / len(token_probs))

def minimum_token_probability(token_probs):
    return np.min(token_probs)
    
def classify_certainty(min_prob, min_threshold=1e-17):
    """
    Classify the model's certainty based on the minimum probability.
    """
    if min_prob < min_threshold:
        return "Model Uncertain"
    else:
        return "Certain"

def return_certainty_scores(q):
    prompt = f"""[INST] <<SYS>>You are a helpful AI assistant that answers questions, reply concisely<</SYS>>
    {q}[/INST]"""

    model_text, log_probs = get_llama_logits(prompt)

    probs = convert_logprob_to_prob(log_probs)

    # Calculate minimum probability
    min_prob = minimum_token_probability(probs)
    certainty_score = classify_certainty(min_prob)

    return min_prob

queries = ["Whats the capital of france?", \
           "Capital of england is..",\
           "Who is the Uk prime minister in 2023?", \
           "when did aliens first make contact?",
           "who is the prime minister of the UK",\
          "who was the First astronaut on mars",\
          "where can I buy Apples new quantum computer?"
          ]

num_fake_prompts = 4

min_probs =[]
for q in queries:
    prob = return_certainty_scores(q)
    min_probs.append(prob)
    
sorted_indices = np.argsort(min_probs)

# Get the indices of the lowest values
lowest_two_indices = sorted_indices[:num_fake_prompts]
    
for index in lowest_two_indices:
    print(f"Query: '{queries[index]}'")
    print("* Uncertainty Detected *")
    print(f"Minimum Probability: {min_probs[index]}")
    print("-------")

# Example usage
return_certainty_scores("What is the capital of France?")