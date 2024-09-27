from pydantic import BaseModel


class ClassificationResponse(BaseModel):
    label: str
