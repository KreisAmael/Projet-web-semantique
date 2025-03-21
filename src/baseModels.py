from pydantic import BaseModel, Field 
from typing import Optional
from enum import Enum

class SpacyModelEnum(str, Enum):
    en_sm = "en_core_web_sm"
    en_lg = "en_core_web_lg"

class TripletExtractorConfig(BaseModel):
    spacy_model : SpacyModelEnum = Field(..., description="spacy model used for entities and relations extraction")
    device : Optional[int] = Field(default=-1, description="number of GPU, default -1 for CPU")
    compute_coref : Optional[bool] = Field(default=True, description="Tell if you want to map none explicit subjects using coref")
    unknown_entity: Optional[str] = Field(default="Unknown", description="Entity type for entity which we doesn't know its entity type")

class EmbeddingBaseSearcherConfig(BaseModel):
    model_name : str = Field(..., description="used model to get embeddings")
    threshold : float = Field(..., description="minumimum threshold to keep the triplet")

class EvaluationConfig(BaseModel):
    threshold : float = Field(..., description="minumimum threshold to validate a prediction")
    true_triplets_path : str = Field(..., description="path of true annoted triplets")
class KgCreatorConfig(BaseModel):
    ontology_path : str = Field(..., description="path of the ontology (description not KG)")
    rel_synonyms_path : str = Field(..., description="json file path of synonyms for each existing property target in the ontology")
    input_path : str = Field(..., description="path of unstructured text to extract the KG")
    output_path : Optional[str] = Field(default=None, description="Path to save the kg, if None save to same path as input_path")
    batch_size : Optional[int] = Field(default=8, description="number of threads (traget : triplet producer) which can run at the same time")
    call_wiki_data : Optional[bool] = Field(default=False, description="Called wikidata to get an entity from an extracted entity")
    evaluationConfig : Optional[EvaluationConfig] = Field(default=None, description="Compute evaluation")
    tripletExtractorConfig : TripletExtractorConfig = Field(..., description="tripletExtractorConfig")
    embeddingBaseSearcherConfig : EmbeddingBaseSearcherConfig = Field(..., description="embeddingBaseSearcherConfig")
    