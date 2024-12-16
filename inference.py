from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from datasets import load_dataset

# Initialize the original LLM model (e.g., LLaMA 3B) with deterministic settings
llm_original = ChatOllama(model='llama3.2:3b', temperature=0)

# Initialize the quoter LLM model using a saved GGUF model
# Ensure the GGUF model is saved correctly for use with Ollama (see link for conversion instructions)
llm_quoter = ChatOllama(model="your saved model", temperature=0)

# Define a prompt template for extracting quotes from a given context
prompt = PromptTemplate(
    template="""
        Given the question and the context provide relevant quotes from the context that support the answer. 
        Your answer must be just the quotes, not the entire context.\n 
        Format: ##begin_quote## quote ##end_quote## for each quote. 
        Do not add anything else other than the quotes.
        Question: {question}
        Context: {context}
        Quotes:
    """,
    input_variables=["question", "context"],
)

# Define agents using the prompt template and LLM models
quoter_agent = prompt | llm_quoter  # Agent for the saved model
original_agent = prompt | llm_original  # Agent for the original LLaMA model

def inference_on_test_set(agent=quoter_agent):
    """
    Run inference on the test set of a dataset using a specified agent.

    Args:
        agent (callable): The agent to use for generating responses. Defaults to `quoter_agent`.

    Returns:
        dict: A dictionary where the keys are sample IDs and the values are the generated quotes.
    """
    # Load the test set from the specified dataset path
    d = load_dataset('path-to-your-dataset')['test']

    results = {}
    for i in d:
        # Generate quotes using the agent
        result = agent.invoke(
            {
                "question": i["question"],  # The question for the sample
                "context": i["context"],  # The context for the sample
            }
        )
        # Store the result in a dictionary keyed by sample ID
        results[i["id"]] = result
    
    return results
