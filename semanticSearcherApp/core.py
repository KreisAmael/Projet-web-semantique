from sentence_transformers import SentenceTransformer, util
import numpy as np 

class EmbeddingBaseSearcher():
    def __init__(self, model_name : str, threshold : float, relations_synonyms : dict):
        self.embedding_model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.all_vectors : dict = {
            rel_name : self.embedding_model.encode(synonym, convert_to_tensor=True) for rel_name, synonym in relations_synonyms.items()
        }
    
    def get_nearest_neighbor(self, input_text : str)->str:
        # encode input_text
        best_rel, max_similiraty = None, 0
        input_embedding = self.embedding_model.encode(input_text, convert_to_tensor=True)
    
        for rel_name, vectors in self.all_vectors.items():
            
            similarities = np.array(util.cos_sim(input_embedding, vectors)[0])
            current_max_similiraty = np.max(similarities)
            if current_max_similiraty >= max_similiraty:
                max_similiraty = current_max_similiraty
                best_rel = rel_name
        if max_similiraty >= self.threshold:
            return best_rel
        return None

 