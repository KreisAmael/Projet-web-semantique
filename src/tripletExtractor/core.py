import tripletExtractor.spacy_component # init don't remove
import spacy
import crosslingual_coreference # init don't remove
from crosslingual_coreference.CorefResolver import CorefResolver as Resolver

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
            self.resolver = Resolver()
        
            
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
            coref_doc = self.coref(input_text)
            input_text = coref_doc._.resolved_text
            res['coref_text'] = input_text
             
            """ # kill spans in clustsers which are NER but replaced with a new Span
            heads = coref_doc._.cluster_heads # spans of references
            clusters = coref_doc._.coref_clusters.copy() # replaced spans
            to_replaced = {}
            print("before")
            for i, (head, span) in enumerate(heads.items()):
                map= ""
                for j, replaced_span in enumerate(clusters[i]):
                    new_cluster = clusters[i].copy()
                    start, end = replaced_span[0], replaced_span[1]
                    span_text = doc.text[start:end]
                    map += span_text + " ,"
                    if span_text != head and span_text in entities_types:
                        new_cluster = clusters[i][:j] # we kill also all replaced after even not NER
                        #break
                    else:
                        to_replaced[start] = (end-start, head)
                    clusters[i] = new_cluster
                print(f"{head} : {map}")
            print("-"*100)"""
             

            """print("after")
            for i, (head, span) in enumerate(heads.items()):
                map= ""
                for j, replaced_span in enumerate(clusters[i]):
                    start, end = replaced_span[0], replaced_span[1]
                    span_text = doc.text[start:end]
                    map += span_text + " ,"
                print(f"{head} : {map}")

            
            print(clusters)"""
            #out =  self.resolver.replace_corefs(coref_doc, [clusters])
             
            """new_text = ""
            i = 0
            while i < len(input_text):
                if i in to_replaced:
                    new_text += to_replaced[i][1]
                    i += to_replaced[i][0]
                else:
                    new_text += input_text[i]
                    i += 1
            print(new_text)
            self.d"""
            
            
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

    

        

 