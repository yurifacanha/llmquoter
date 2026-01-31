from pydantic import BaseModel, Field


class RecallPrecisionOutput(BaseModel):
    recall: float = Field(description="Fraction (0.0 to 1.0) of ground truth quotes present in system response")
    precision: float = Field(description="Fraction (0.0 to 1.0) of system response quotes present in ground truth")
