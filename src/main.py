from tripletExtractor import TripletProducer
from evaluation import compute_evaluation
import requests
from baseModels import KgCreatorConfig
import logging
import json
from threading import Thread
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
from utils import load_rdf_graph, call_wiki_api, extract_ontology_triplets
from tqdm import tqdm
import os
import uuid
import time
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INIT_SEARCHER_URL = "http://127.0.0.1:8000/init_searcher"
GET_NEAREST_NEIGHBOR_URL = "http://127.0.0.1:8000/get_nearest_neighbor" 

class KgCreator():
    def __init__(self, kgCreatorConfig : KgCreatorConfig):
        self.batch_size = kgCreatorConfig.batch_size
        self.input_path = kgCreatorConfig.input_path
        self.output_path = kgCreatorConfig.output_path
        self.ontology_path = kgCreatorConfig.ontology_path
        self.call_wiki_data = kgCreatorConfig.call_wiki_data
        self.compute_coref = kgCreatorConfig.tripletExtractorConfig.compute_coref
        self.evaluationConfig = kgCreatorConfig.evaluationConfig

        # load the ontology
        self.ontology_graph = load_rdf_graph(self.ontology_path)
        self.classes, self.properties, self.ontology_triplets = extract_ontology_triplets(self.ontology_graph)
       
        #self.classes = extract_classes(self.ontology_graph)
        #self.properties = extract_properties(self.ontology_graph)

        self.property_names = [str(p).split("/")[-1] for p in list(self.properties)]
        self.property_names = [p for p in self.property_names if p not in ["hasDescription", "hasUrl"]] # banned property based on wikidata augmentation
        self.ontology_triplets = {triplet[1] : {"head" : triplet[0], "tail" : triplet[-1]} for triplet in self.ontology_triplets if triplet[1] in self.property_names}
        
        # init triplet extractor 
        self.tripletProducer = TripletProducer(**kgCreatorConfig.tripletExtractorConfig.dict())
        logging.info(f'Successfully init triplet producer based on "spacy model" : {kgCreatorConfig.tripletExtractorConfig.spacy_model}')
        
        # loand semantic serahcer based on cosine sim
        self.relations_synonyms = json.load(open(kgCreatorConfig.rel_synonyms_path, encoding='utf-8'))
        # init embeddingBaseSearcher
        data = {
            "model_name" : kgCreatorConfig.embeddingBaseSearcherConfig.model_name,
            "threshold" : kgCreatorConfig.embeddingBaseSearcherConfig.threshold,
            "relations_synonyms" :self.relations_synonyms
        }
        response = requests.post(INIT_SEARCHER_URL, json=data)
        if response.status_code != 200:
            raise "Failed to load embeddingBaseSearcher"
        logging.info(f'Successfully load embeddingBaseSearcherConfig')


    def extract_entities_relations(self):
        # laod the data (text)
        data = json.load(open(self.input_path, encoding='utf-8'))
        self.triplets = [{} for _ in range(len(data))]
        threads = []
        for i in tqdm(range(len(data)), desc="Extract entities and relations from text"):
            thread = Thread(target=self.tripletProducer.extract, args=(data[i]['text'],self.triplets[i],))
            thread.start()
            threads.append(thread)
            if len(threads) == self.batch_size:
                for thread in threads:
                    thread.join()
                threads = []
        for thread in threads:
            thread.join()
       
        # to visualize
        output_file = open(os.path.join(self.output_path, "step1_triplets_extraction.json"), "w")
        json.dump(self.triplets, output_file)
        output_file.close()
        return self.triplets
    
    def entities_merging_and_augmentation(self):
        already_get_nearest_neighbor = {}
        already_merged = {}
        self.entities_potentials_types = {} #uri : {ent_type : counter}

        for triplets_obj in tqdm(self.triplets, desc="Merge entities and relations / compute augmentation (Wikidata)"):
            new_entities, new_relations = [], []
            current_already_merged = {}
           
            # retrieve property name
            for rel in triplets_obj['relations']:
                if rel['relation'] not in already_get_nearest_neighbor:
                    data = {
                        "input_text": rel['relation'] # relation type
                    }
                    response = requests.post(GET_NEAREST_NEIGHBOR_URL, json=data)
                    if response.status_code != 200:
                        raise f"Failed to get bets matching property of {rel['relation']}"
                    already_get_nearest_neighbor[rel['relation']] = response.json()['nearest_neighbor']
                if already_get_nearest_neighbor[rel['relation']] is None: # the relation doesn't match any existing property
                    continue
                
                # merge head & tail
                current_entities = {"head" : None, "tail" : None}
                for rd in list(current_entities.keys()):
                    ent = triplets_obj['entities'][rel[rd]]['text']
                    ent_id = ent.lower().strip()
                    if ent_id not in already_merged:
                        output = None
                        if self.call_wiki_data:
                            output = call_wiki_api(ent) # to sensible 
                        if output:
                            already_merged[ent_id] = output
                        else:
                            already_merged[ent_id] = {
                                "id" : str(uuid.uuid4()),
                                "label" : ent
                            }
                        self.entities_potentials_types[already_merged[ent_id]['id']] = {}
                    
                    if already_merged[ent_id]['id'] not in current_already_merged:
                        current_already_merged[already_merged[ent_id]['id']] = True
                        new_entities.append(already_merged[ent_id])
                       
                    current_entities[rd] = already_merged[ent_id]['id'] 
                    # asisgn potential ent types
                    current_type = self.ontology_triplets[already_get_nearest_neighbor[rel['relation']]][rd]
                    if current_type not in self.entities_potentials_types[already_merged[ent_id]['id']]:
                        self.entities_potentials_types[already_merged[ent_id]['id']][current_type] = 0
                    self.entities_potentials_types[already_merged[ent_id]['id']][current_type] += 1

                
                new_relations.append({
                    'relation' : already_get_nearest_neighbor[rel['relation']],
                    'head' : current_entities['head'],
                    'tail' : current_entities["tail"]
                })
            triplets_obj['entities'] = new_entities
            triplets_obj['relations'] = new_relations
            



        self.entities = list(already_merged.values())
        # to visualize
        output_file = open(os.path.join(self.output_path, "step2_merge_augmentation.json"), "w")
        json.dump(self.triplets, output_file)
        output_file.close()
        
            

    def build_rdf_graph(self):
        logging.info('-'*100)
        start_time = time.time()
        kgCreator.extract_entities_relations()
        kgCreator.entities_merging_and_augmentation()

        # save as ttl graph
        new_graph = Graph()
        ex = Namespace("http://output.org/")

        total_classes = 0
        total_properties = 0
        for ent in tqdm(self.entities, desc="Assign best class to each entity and add Instance"):
            ent["type"] = max(self.entities_potentials_types[ent['id']], key=self.entities_potentials_types[ent['id']].get)
            self.entities_potentials_types[ent['id']] = (ent['label'].replace(' ', '_'), ent['label'])

            # to make easier visualisation 
            #entity_uri = ex[ent['id']]
            entity_uri = ex[self.entities_potentials_types[ent['id']][0]]
            class_uri = ex[ent["type"]]
            new_graph.add((entity_uri, RDF.type, class_uri))
            new_graph.add((entity_uri, RDFS.label, Literal(ent['label'])))
            total_classes += 1
            # normally manage hasDescription and hasUrl but issue with wikidata to sensible 
            if 'description' in ent:
                new_graph.add((entity_uri, ex['hasDescription'], Literal(ent['description'])))
                total_classes += 1
            if 'url' in ent:
                new_graph.add((entity_uri, ex['hasUrl'], Literal(ent['url'])))
                total_classes += 1

        logging.info(f"Add {total_classes} classes")

        already_add = []
        if self.evaluationConfig:
            final_triplets = []
        for triplets_obj in tqdm(self.triplets, desc="Add properties"):
            for rel in triplets_obj['relations']:
                subj_uri = rel['head']
                obj_uri = rel['tail']
                rel_uri = rel['relation']  # 🔹 Use ontology-mapped relation (not raw extracted one)
                rel_id = f"{subj_uri}-{obj_uri}-{rel_uri}"
                if rel_id not in already_add:
                    # to make easier visualisation 
                    subj_uri = self.entities_potentials_types[rel['head']][0]
                    obj_uri = self.entities_potentials_types[rel['tail']][0]
                    new_graph.add((ex[subj_uri], ex[rel_uri], ex[obj_uri]))
                    already_add.append(rel_id)
                    total_properties += 1

                    if self.evaluationConfig:
                        final_triplets.append({
                            "subject" : self.entities_potentials_types[rel['head']][1],
                            "property_type" : rel_uri,
                            "object" : self.entities_potentials_types[rel['tail']][1]
                        })


        logging.info(f"Add {total_properties} properties")
        
        graph_name = "graph"
        if self.compute_coref:
            graph_name += "_coref"
        if self.call_wiki_data:
            graph_name += "_wikidata"
        graph_name += '.ttl'
        new_graph.serialize(os.path.join(self.output_path, graph_name), format="turtle")
        
        
        if self.evaluationConfig:
            """output_file = open(os.path.join("evaluation", graph_name.replace("ttl", "json")), "w")
            json.dump(final_triplets, output_file)
            output_file.close()"""
            true_triplets = json.load(open(self.evaluationConfig.true_triplets_path, encoding='utf-8'))
            output_eval = compute_evaluation(
                name=graph_name.replace(".ttl", ""),
                property_types=self.property_names,
                true_triplets=true_triplets,
                predicted_triplets=final_triplets,
                threshold=self.evaluationConfig.threshold
                )
            logging.info(f"Evaluation:\n{output_eval}")

        logging.info(f"Save the graph at {os.path.join(self.output_path, graph_name)}")
        end_time = time.time()
        logging.info(f"took {end_time-start_time} seconds")
        logging.info('-'*100)
         
        return new_graph
    
if __name__ == "__main__":
    config = KgCreatorConfig(
        input_path="test/test.json",
        ontology_path="test/ontology.ttl",
        output_path="test",
        rel_synonyms_path="test/rel_synonyms.json",
        batch_size=8,
        call_wiki_data=False,
        evaluationConfig = {
            "true_triplets_path" : "evaluation/true_triplets.json",
            "threshold" : 0.8
        },
        tripletExtractorConfig = {
            "spacy_model":"en_core_web_lg",
            "device":-1,
            "compute_coref": False
        },
        embeddingBaseSearcherConfig = {
            "model_name" : "sentence-transformers/all-mpnet-base-v2",
            "threshold" : 0.25
        }
    )
    kgCreator = KgCreator(
        config
    )

   
    kgCreator.build_rdf_graph()