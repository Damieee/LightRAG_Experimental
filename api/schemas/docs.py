from pydantic import BaseModel, Field

class InsertDocRequest(BaseModel):
    content: str = Field(
        example="This is a detailed document about machine learning. Machine learning (ML) is a subset of artificial intelligence that focuses on developing systems that can learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data. Common ML techniques include supervised learning, unsupervised learning, and reinforcement learning. Applications range from image recognition to natural language processing."
    )

class UpdateDocRequest(BaseModel):
    doc_id: str = Field(example="doc_123abc456")
    content: str = Field(
        example="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn like humans. It encompasses various subfields including machine learning, neural networks, and deep learning. AI systems can perform tasks such as visual perception, speech recognition, decision-making, and language translation. The field continues to evolve with applications in healthcare, finance, autonomous vehicles, and more."
    )

class RemoveDocRequest(BaseModel):
    doc_id: str = Field(example="doc_123abc456")
