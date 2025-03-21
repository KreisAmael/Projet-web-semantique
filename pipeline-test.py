import spacy
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Load NLP model
nlp = spacy.load("en_core_web_lg")

# Load Sentence Transformer for embeddings
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load the RDF graph
def load_rdf_graph(file_path, format="turtle"):
    graph = Graph()
    graph.parse(file_path, format=format)
    return graph

# Extract ontology classes and properties
def extract_ontology_triplets(graph):
    classes = set()
    properties = set()
    triplets = set()

    # Get ontology classes
    for s, p, o in graph.triples((None, RDF.type, RDFS.Class)):
        classes.add(s)

    # Get ontology properties
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        properties.add(s)

    # Get relationships between classes
    for s, p, o in graph.triples((None, RDFS.domain, None)):
        for _, _, range_class in graph.triples((s, RDFS.range, None)):
            triplets.add((get_last_part(o), get_last_part(s), get_last_part(range_class)))

    return classes, properties, list(triplets)  # Convert triplets to a list

# Extract entities and relations from text using SpaCy
def extract_entities_and_relations(text):
    doc = nlp(text)
    entities = set()
    relations = set()

    entity_map = {}
    for ent in doc.ents:
        entities.add((ent.text, ent.label_))  # Store entity with its label
        for token in ent:
            entity_map[token.text] = (ent.text, ent.label_)  # Save entity-label mapping

    for token in doc:
        if token.pos_ == "VERB":
            subjects = set()
            objects = set()

            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subjects.add(entity_map.get(child.text, (child.text, None)))

                    for conj in child.conjuncts:
                        subjects.add(entity_map.get(conj.text, (conj.text, None)))

                elif child.dep_ == "dobj":
                    objects.add(entity_map.get(child.text, (child.text, None)))

                    for conj in child.conjuncts:
                        objects.add(entity_map.get(conj.text, (conj.text, None)))

                elif child.dep_ == "prep":
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            objects.add(entity_map.get(prep_child.text, (prep_child.text, None)))

            for subj in subjects:
                for obj in objects:
                    relations.add((subj[0], token.lemma_, obj[0]))  # Store only entity names

    return list(entities), list(relations)

# Helper function to extract the last part of a URI
def get_last_part(uri):
    return str(uri).rsplit("/", 1)[-1]

# Generate embeddings for a triplet
def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

# Match extracted triplets to ontology triplets using cosine similarity
def get_best_match(extracted_triplet, ontology_triplets, threshold=0.7):
    extracted_embedding = get_embedding(" -> ".join(extracted_triplet))
    ontology_embeddings = np.array([get_embedding(" -> ".join(t)) for t in ontology_triplets])

    print("Extracted triplet:"," -> ".join(extracted_triplet))
    print("Ontology triplet:",[" -> ".join(t) for t in ontology_triplets])

    similarities = cosine_similarity([extracted_embedding], ontology_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    print("Best score:",best_score)
    return ontology_triplets[best_idx] if best_score >= threshold else None

# Assign classes to entities based on relation frequency and similarity
def assign_classes(entities, matched_triplets, ontology_classes):
    entity_classes = {}
    entity_to_classify = entities.copy()
    triplet_remaining = matched_triplets.copy()

    print("Entities:",entities)
    print("Matched triplet:",matched_triplets)
    print("Ontology classes:",ontology_classes)

    while len(entity_to_classify) != 0 and len(triplet_remaining) != 0:
        entity_votes = Counter()

        # Step 1: Count how many times each extracted entity could match an ontology class
        entity_set = {entity for entity, _ in entity_to_classify}
        for (subj, _, obj), (_, _, _) in triplet_remaining:
            if subj in entity_set:
                entity_votes[subj] += 1
            if obj in entity_set:
                entity_votes[obj] += 1

        if entity_votes == Counter() :
            break
        print("Vote:",entity_votes)
        print("Most common:",entity_votes.most_common(1)[0][0])

        entity = entity_votes.most_common(1)[0][0]

        #Finding the most common entity to classify it
        index = 0
        for i, e in enumerate(entity_to_classify):
            if e[0] == entity:
                index = i
                break
        
        print("Index:", index)
        entity_to_classify.pop(index)
        all_matched_classes = []

        for triplet_pair in triplet_remaining:
            if triplet_pair[0][0] == entity or triplet_pair[0][2] == entity:
                all_matched_classes.append(triplet_pair)
        
        print("Matching classes: ", all_matched_classes)
        class_votes = Counter()

        #finding the class that correspond to the most relations
        for (subj, _, obj), (sclass, _, oclass) in all_matched_classes:
            if subj == entity:
                class_votes[sclass] += 1
            if obj == entity:
                class_votes[oclass] += 1
        
        print("Class votes:",class_votes)
        print("Best class:", class_votes.most_common(1))
        best_class = class_votes.most_common(1)[0][0]
        entity_classes[entity] = best_class

        #removing relations with the wrong class for the entity
        for (subj, rel1, obj), (sclass, rel2, oclass) in all_matched_classes:
            if subj == entity and sclass != best_class:
                triplet_remaining.remove(((subj, rel1, obj), (sclass, rel2, oclass)))
            if obj == entity and oclass != best_class:
                triplet_remaining.remove(((subj, rel1, obj), (sclass, rel2, oclass)))
        
        #removing relations where both entities are clissified
        for (subj, rel1, obj), (sclass, rel2, oclass) in triplet_remaining:
            if subj in entity_classes and obj in entity_classes:
                triplet_remaining.remove(((subj, rel1, obj), (sclass, rel2, oclass)))

        #removing entities that have no relations
        object_set = {subj for (subj, rel1, obj), (sclass, rel2, oclass) in triplet_remaining}
        object_set.update({obj for (subj, rel1, obj), (sclass, rel2, oclass) in triplet_remaining})
        print("Set: ", object_set)
        for e, label in entity_to_classify:
            if not e in object_set:
                entity_to_classify.remove((e, label))

        print('\033[36m',"Remaining entities:", entity_to_classify,'\033[0m')
        print('\033[36m',"Remaining relations:", triplet_remaining,'\033[0m')

    return entity_classes



# Build RDF graph using valid triplets and classes
def build_rdf_graph(valid_entities, matched_triplets):
    new_graph = Graph()
    ex = Namespace("http://output.org/")
    
    print("Valid entities:", valid_entities)
    print("Matched triplets:", matched_triplets)

    # Step 1: Add entities with their assigned classes
    for entity, cls in valid_entities.items():
        entity_uri = ex[entity.replace(" ", "_")]
        class_uri = ex[cls]
        
        new_graph.add((entity_uri, RDF.type, class_uri))
        new_graph.add((entity_uri, RDFS.label, Literal(entity)))  # Preserve entity names

    # Step 2: Add valid relations (ontology-aligned)
    for (subj_text, extracted_rel, obj_text), (subj_cls, ontology_rel, obj_cls) in matched_triplets:
        if subj_text in valid_entities and obj_text in valid_entities:
            subj_uri = ex[subj_text.replace(" ", "_")]
            obj_uri = ex[obj_text.replace(" ", "_")]
            rel_uri = ex[ontology_rel]  # 🔹 Use ontology-mapped relation (not raw extracted one)

            new_graph.add((subj_uri, rel_uri, obj_uri))

    return new_graph



# Main function
def main():
    # Load ontology
    ontology_file = "ontology_complex.ttl"
    ontology_graph = load_rdf_graph(ontology_file)

    # Extract ontology triplets
    ontology_classes, ontology_properties, ontology_triplets = extract_ontology_triplets(ontology_graph)

    # Input text
    text = "John Doe works for Acme Corp as a Project Manager and has been there for two years. He is skilled in Programming and Leadership and is currently working on Project X. He is also participating in the Tech Conference 2023. Alice Smith is employed by Tech Solutions as a Senior Data Scientist. She has skills in Data Analysis and Project Management and is involved in the AI Development project. She will attend the AI Summit next month. Bob Johnson is the Chief Technology Officer at Global Innovations. He has expertise in Machine Learning and Cloud Computing and is leading the Blockchain Initiative. He is a speaker at the Future Tech Expo. Acme Corp is located in New York and organizes the Tech Conference 2023. Tech Solutions has its headquarters in San Francisco and is hosting the AI Summit. Global Innovations operates from London and is planning the Future Tech Expo."

    # Extract entities and relations
    entities, relations = extract_entities_and_relations(text)
    print("Extracted Entities:", entities)
    print("Extracted Relations:", relations)

    # Match extracted triplets to ontology
    matched_triplets = []
    for triplet in relations:
        best_match = get_best_match(triplet, ontology_triplets, threshold=0.25)
        if best_match:
            matched_triplets.append((triplet,best_match))

    print("Matched Triplets:", matched_triplets)

    # Assign best ontology classes to entities
    valid_entities = assign_classes(entities, matched_triplets, ontology_classes)
    print("Assigned Classes:", valid_entities)

    # Build and save RDF graph
    new_graph = build_rdf_graph(valid_entities, matched_triplets)
    new_graph.serialize("output_graph.ttl", format="turtle")
    print("New RDF graph saved to 'output_graph.ttl'.")
    new_graph.print()

if __name__ == "__main__":
    main()
