from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from ai.chains.prompts import INFERENCE_PROMPT


def _get_llm(model_name: str, temperature: float = 0):
    return ChatOllama(model=model_name, temperature=temperature)


def get_chain(model_name: str, temperature: float = 0):
    prompt = PromptTemplate(
        template=INFERENCE_PROMPT,
        input_variables=["question", "context"],
    )
    llm = _get_llm(model_name, temperature)
    return prompt | llm
