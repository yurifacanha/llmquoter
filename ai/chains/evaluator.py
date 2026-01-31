from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from ai.chains.prompts import EVALUATOR_PROMPT
from ai.parsers import RecallPrecisionOutput

load_dotenv()




def _get_llm():
    return ChatOpenAI(
        model="o4-mini",
        max_retries=0,
        reasoning={"effort": "medium"}
    )


def get_chain():
    prompt = PromptTemplate(
        template=EVALUATOR_PROMPT,
        input_variables=["ground_truth", "system_response"],
    )
    
    llm = _get_llm()
    return prompt | llm.with_structured_output(RecallPrecisionOutput)
