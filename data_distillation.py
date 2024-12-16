from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import load_dataset, Dataset
import random
import pandas as pd

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize the model (Google Generative AI - Gemini 1.5 Pro)
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0  # Deterministic output
)

# Define the prompt template for extracting quotes
prompt = PromptTemplate(
    template="""
        Instruction: Given the question, the context, and the expected answer below, 
        provide relevant quotes from the context that support the answer.
        Your answer must be just the quotes, not the entire context.
        Format: ##begin_quote## quote ##end_quote## for each quote.
        Do not add anything else other than the quotes.
        
        Your turn:
        Question: {question}
        Context: {context}
        Answer: {answer}
        Quotes:
    """,
    input_variables=['question', "context", "answer"],
)

# Combine the model and prompt into an agent
agent = prompt | model

def dataset2df():
    """
    Loads and preprocesses the HotpotQA dataset to create a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing selected samples with 
                      'id', 'question', 'context', 'answer', and 'level'.
    """
    # Load the HotpotQA dataset
    hotpotqa = load_dataset("hotpotqa/hotpot_qa", 'distractor', split="train", trust_remote_code=True)

    # Randomly sample 15,000 examples from the dataset
    sample_indices = random.sample(range(len(hotpotqa)), 15000)
    sampled_hotpotqa = hotpotqa.select(sample_indices)

    # Extract context, question, answer, ID, and level fields
    new_context = ['\n'.join([' '.join(i) for i in sample['context']['sentences']]) for sample in sampled_hotpotqa]
    questions = [sample['question'] for sample in sampled_hotpotqa]
    answers = [sample['answer'] for sample in sampled_hotpotqa]
    ids = [sample['id'] for sample in sampled_hotpotqa]
    levels = [sample['level'] for sample in sampled_hotpotqa]

    # Create a DataFrame
    df = pd.DataFrame({'id': ids, 'question': questions, 'context': new_context, 'answer': answers, 'level': levels})
    return df

def generate_dataset(df, agent):
    """
    Generates a dataset with golden quotes using the agent for semantic extraction.

    Args:
        df (pd.DataFrame): Input DataFrame containing the questions, contexts, and answers.
        agent (callable): The agent for extracting quotes based on the provided prompt and model.

    Returns:
        None: The processed dataset is pushed to the Hugging Face Hub.
    """
    # Convert DataFrame to a list of dictionaries
    documents = df.to_dict(orient='records')
    new_documents = []
    batch_size = 10

    # Process the dataset in batches to improve efficiency
    for i in range(0, len(documents), batch_size):
        batch = [{"question": doc['question'], 
                  "context": doc['context'], 
                  "answer": doc['answer']} for doc in documents[i:i + batch_size]]

        # Use the agent to generate quotes for the batch
        results = agent.batch(batch)
        for j, result in enumerate(results):
            documents[i + j]['quotes'] = result.content  # Add the quotes to each document

        new_documents.extend(documents)

    # Assign new IDs to the processed documents
    for idx, doc in enumerate(new_documents):
        doc['id'] = idx + 1

    # Convert the new documents into a Hugging Face Dataset
    dataset = Dataset.from_dict(new_documents)

    # Split the dataset into training and testing sets
    dataset = dataset.train_test_split(test_size=0.04, seed=42)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub("path-to-your-hf-repo", private=False)

def main():
    """
    Main function to create and push the golden quotes dataset.

    Steps:
        1. Preprocess the HotpotQA dataset to create a DataFrame.
        2. Use the agent to generate quotes for the dataset.
        3. Push the processed dataset to the Hugging Face Hub.
    """
    df = dataset2df()  # Preprocess the dataset
    generate_dataset(df, agent)  # Generate and push the dataset
