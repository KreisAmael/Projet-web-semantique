from pydantic import BaseModel, Field
# health
class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="The current status of the API (e.g., healthy, degraded)")

class InitSearcherRequest(BaseModel):
    model_name: str
    threshold: float
    relations_synonyms: dict

class NearestNeighborRequest(BaseModel):
    input_text: str
