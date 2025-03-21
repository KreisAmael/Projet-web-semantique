from fastapi import FastAPI, HTTPException, status 
from core import EmbeddingBaseSearcher
from baseModels import HealthCheckResponse, InitSearcherRequest, NearestNeighborRequest

embedding_searcher = None 

app = FastAPI(
    title="Find the Nearest Neighbor based on relatons name synonyms",
    description="API to consitute the final decison of the loan request based on property and credit evaluations, create also a repayment schedule.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# check if boths clients exists
@app.get("/health", summary="Health Check Endpoint", response_model=HealthCheckResponse, tags=["Health"])
def health_check()->HealthCheckResponse:
    """Checks the health of the API and database connection."""
    return HealthCheckResponse(status="healthy")


@app.post("/init_searcher", summary="Initialize the Global EmbeddingBaseSearcher", tags=["Initialization"])
def init_searcher(config: InitSearcherRequest):
    """
    Initializes the global instance of `EmbeddingBaseSearcher` with the provided parameters.
    """
    global embedding_searcher
    try:
        # Initialize global instance
        embedding_searcher = EmbeddingBaseSearcher(
            model_name=config.model_name,
            threshold=config.threshold,
            relations_synonyms=config.relations_synonyms
        )
        return {"message": "EmbeddingBaseSearcher initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/get_nearest_neighbor", summary="Find the Nearest Neighbor", tags=["Search"])
def get_nearest_neighbor(request: NearestNeighborRequest):
    """
    Calls `embedding_searcher.get_nearest_neighbor(input_text)` and returns the result.
    """
    global embedding_searcher
    if embedding_searcher is None:
        raise HTTPException(status_code=400, detail="EmbeddingBaseSearcher is not initialized. Call /init_searcher first.")

    try:
        result = embedding_searcher.get_nearest_neighbor(request.input_text)
        return {"nearest_neighbor": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving nearest neighbor: {str(e)}")