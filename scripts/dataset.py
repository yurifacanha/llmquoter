from typing import Optional
from datasets import Dataset, DatasetDict
from pydantic import BaseModel

from mongo import HotPotQAMongo, HotPotQADocument


CONVERSATION_INSTRUCTION = """Instruction: Given the question and the context provide relevant quotes from the context that support the answer. Your answer must be just the quotes, not the entire context.
format: ##begin_quote## quote ##end_quote## for each quote. Do not add anything else other than the quotes.
-------------------------------------------------------
Question: {question}
Context: {context}
Quotes:
"""


class HFDatasetRecord(BaseModel):
    hf_id: str
    question: str
    context: str
    answer: str
    level: str
    split: str
    quotes: Optional[str] = None
    conversations: Optional[list[dict]] = None


def create_conversation(doc: HotPotQADocument) -> list[dict]:
    user_content = CONVERSATION_INSTRUCTION.format(
        question=doc.question,
        context=doc.context
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": doc.quotes or doc.original_quotes or ""}
    ]


def fetch_all_from_mongo() -> list[HotPotQADocument]:
    with HotPotQAMongo() as mongo:
        all_docs = mongo.find_all()
        return [doc for doc in all_docs if doc.quotes is not None]


def split_by_field(docs: list[HotPotQADocument]) -> dict[str, list[dict]]:
    splits = {"train": [], "validation": [], "test": []}
    for doc in docs:
        split_name = doc.split or "train"
        if split_name in splits:
            conversations = create_conversation(doc)
            record = HFDatasetRecord(**doc.model_dump(), conversations=conversations)
            splits[split_name].append(record.model_dump())
    return splits


def build_dataset_dict(splits: dict[str, list[dict]]) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_list(splits["train"]),
        "validation": Dataset.from_list(splits["validation"]),
        "test": Dataset.from_list(splits["test"])
    })


def push_to_hub(dataset_dict: DatasetDict, repo_name: str, private: bool = False):
    dataset_dict.push_to_hub(repo_name, private=private, token=True)
