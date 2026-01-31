import json
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import Any, Optional
from pydantic import BaseModel, field_validator


class HotPotQADocument(BaseModel):
    hf_id: str
    question: str
    context: str
    answer: str
    level: str
    split: str
    original_quotes: Optional[str] = None
    quotes: Optional[str] = None
    
    @field_validator("quotes", "answer", mode="before")
    @classmethod
    def normalize_text_field(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            texts = [item.get("text", "") for item in v if isinstance(item, dict)]
            return "".join(texts) if texts else str(v)
        return str(v)


class HotPotQAMongo:
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database_name: str = "llmquoter"
    ):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection: Collection = self.db["HotPotQAQuotes"]
        self.collection.create_index("hf_id", unique=True)
    
    def insert_one(self, doc: HotPotQADocument) -> str:
        result = self.collection.insert_one(doc.model_dump())
        return str(result.inserted_id)
    
    def update_field_by_hf_id(self, hf_id: str, field: str, value: Any) -> bool:
        result = self.collection.update_one(
            {"hf_id": hf_id},
            {"$set": {field: value}}
        )
        return result.modified_count > 0
    
    def insert_batch(self, docs: list[HotPotQADocument]) -> int:
        if not docs:
            return 0
        result = self.collection.insert_many([doc.model_dump() for doc in docs])
        return len(result.inserted_ids)
    
    def find_by_hf_id(self, hf_id: str) -> Optional[HotPotQADocument]:
        data = self.collection.find_one({"hf_id": hf_id})
        if data:
            data.pop("_id", None)
            return HotPotQADocument(**data)
        return None
    
    def find_all(self, split: str = "") -> list[HotPotQADocument]:
        docs = []
        filter = {"split": split} if split else {}
        for data in self.collection.find(filter):
            data.pop("_id", None)
            docs.append(HotPotQADocument(**data))
        return docs
    
    def populate(self, data_dir: str = "data") -> int:
        data_path = Path(data_dir)
        files = {
            "train": data_path / "train.json",
            "validation": data_path / "validation.json",
            "test": data_path / "test.json"
        }
        
        total_inserted = 0
        for split_name, file_path in files.items():
            if not file_path.exists():
                print(f"Skipping {file_path} - file not found")
                continue
            
            print(f"Loading {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = []
            for item in data:
                doc = HotPotQADocument(
                    hf_id=item["id"],
                    question=item["question"],
                    context=item["context"],
                    answer=item["answer"],
                    level=item["level"],
                    split=split_name,
                    original_quotes=item.get("quotes"),
                    quotes=None
                )
                docs.append(doc)
            
            if docs:
                inserted = self.insert_batch(docs)
                total_inserted += inserted
                print(f"Inserted {inserted} documents from {split_name}")
        
        print(f"Total documents inserted: {total_inserted}")
        return total_inserted
    
    def close(self):
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    mongo = HotPotQAMongo()
    mongo.populate()
    mongo.close()