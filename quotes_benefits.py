from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables from a .env file
load_dotenv()

# Initialize models with specific configurations
gpt3 = ChatOpenAI(
    model="gpt-3.5-turbo",  # OpenAI's GPT-3.5-turbo model
    temperature=0,  # Deterministic output
)
llama1b = ChatOllama(
    model="llama3.2:1b",  # Smaller Llama model with 1B parameters
    temperature=0,  # Deterministic output
)
llama3b = ChatOllama(
    model="llama3.2:3b",  # Larger Llama model with 3B parameters
    temperature=0,  # Deterministic output
)

# List of models to use for generating answers
models = [gpt3, llama1b, llama3b]

# Prompt for generating answers based on the full context
prompt_context = PromptTemplate(
    template="""
    question:{question}
    context: {context}
    Given the question and context, write the right answer to the question based on the context.
    Just give the exact answer to the question.
    """,
    input_variables=['question', "context"],
)

# Prompt for generating answers based on extracted quotes
prompt_quotes = PromptTemplate(
    template="""
    question:{question}
    quotes: {quotes}
    Given the question and quotes extracted from a document, write the right answer to the question based on the quotes.
    Just give the exact answer to the question.
    """,
    input_variables=['question', "quotes"],
)

def get_model_answers(question, context, quotes):
    """
    Generate answers for a given question using both full context and extracted quotes.
    
    This function compares the performance of different models when answering a question 
    using either the complete context or specific quotes extracted by a quoting model.

    Args:
        question (str): The question to be answered.
        context (str): The complete context or document from which quotes are derived.
        quotes (str): Relevant quotes extracted from the context.

    Returns:
        dict: A dictionary containing answers from each model for both the full context 
              and the quotes. The structure is as follows:
              {
                  "model_name": {
                      "context_answer": <Answer from full context>,
                      "quotes_answer": <Answer from quotes>
                  },
                  ...
              }
    """
    answers = {}
    for model in models:
        # Create agents for answering using context and quotes
        agent_context = prompt_context | model
        agent_quotes = prompt_quotes | model

        # Generate answer using the full context
        a1 = agent_context.invoke(
            {'question': question, 'context': context}
        )
        # Generate answer using the extracted quotes
        a2 = agent_quotes.invoke(
            {'question': question, 'quotes': quotes}
        )

        # Store answers in the dictionary with the model's name as the key
        answers[model.model] = {'context_answer': a1, 'quotes_answer': a2}

    return answers
