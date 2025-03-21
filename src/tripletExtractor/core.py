import tripletExtractor.spacy_component
import spacy
import crosslingual_coreference

class TripletProducer():
    def __init__(self, spacy_model : str, device : int = -1, compute_coref : bool = True, unknown_entity : str = "Unknown"):
        """
            args:
                - spacy_model : name of spacy model, !python -m spacy download $spacy_model
                - device : number of GPU, default -1 for CPU
                - coref : is not necessary
        """
        # fo NER
        self.nlp = spacy.load(spacy_model, disable=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        # system of coref, update non explicit subject using coreference
        self.compute_coref = compute_coref
        if self.compute_coref:
            self.coref  = spacy.load(spacy_model, disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
            self.coref.add_pipe(
                "xx_coref", 
                config={
                    "chunk_size": 2500, 
                    "chunk_overlap": 2, 
                    "device": device
                }
            )
        
            
        # Define rel extraction model
        self.rel_ext = spacy.load(spacy_model, disable=['ner', 'lemmatizer', 'attribute_rules', 'tagger'])
        self.rel_ext.add_pipe(
            "rebel", 
            after="senter",
            config={
                'device': device, # Number of the GPU, -1 if want to use CPU
                'model_name':'Babelscape/rebel-large'
            } # Model used, will default to 'Babelscape/rebel-large' if not given
        )
        self.unknown_entity = unknown_entity

    def extract(self, input_text : str, res : dict = {})->dict:
        res["text"] = input_text
        # extracts entities
        doc = self.nlp(input_text)
        entities_types = {ent.text : ent.label_.capitalize() for ent in doc.ents}
        # suppose we don't have same Texte et differents labels !
        # map subjects using coref
        if self.compute_coref:
            input_text = self.coref(input_text)._.resolved_text
            res["coref_text"] = input_text
            
             
            
        # extract triplets (entities implies in a relation | relations)
        doc = self.rel_ext(input_text)
        res['relations'] = [rel_dict for _, rel_dict in doc._.rel.items()]

        # add entities
        entities = []
        already_add_entities = {}
        current_id = 0 
        for rel in res['relations']:
            head, tail = rel['head'], rel['tail']
            # head
            if head not in already_add_entities:
                already_add_entities[head] = current_id
                entities.append({
                    'text' : head,
                    'type': entities_types.get(head, self.unknown_entity)
                })
                current_id += 1
            # tail
            if tail not in already_add_entities:
                already_add_entities[tail] = current_id
                entities.append({
                    'text' : tail,
                    'type': entities_types.get(tail, self.unknown_entity)
                })
                current_id += 1
            # update relations
            rel['head'] = already_add_entities[head]
            rel['tail'] = already_add_entities[tail]
        res['entities'] = entities
        
        return res

    

        

 