from pydantic import BaseModel

class InsertDocRequest(BaseModel):
    content: str

class UpdateDocRequest(BaseModel):
    doc_id: str
    content: str

class RemoveDocRequest(BaseModel):
    doc_id: str
