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
    entity_votes = Counter()

    # Count occurrences of each entity in relations
    for subj, rel, obj in matched_triplets:
        entity_votes[subj] += 1
        entity_votes[obj] += 1

    # Assign most frequent entity classes first
    for entity, _ in entity_votes.most_common():
        for subj, rel, obj in matched_triplets:
            if entity == subj:
                entity_classes[entity] = subj
            elif entity == obj:
                entity_classes[entity] = obj

    # Assign remaining entities using similarity
    for entity, label in entities:
        if entity not in entity_classes:
            best_match = get_best_match((entity, "type", "?"), [(get_last_part(cls), "type", get_last_part(cls)) for cls in ontology_classes], threshold=0.25)
            if best_match:
                entity_classes[entity] = best_match[0]

    return entity_classes

# Build RDF graph using valid triplets and classes
def build_rdf_graph(valid_entities, matched_triplets):
    new_graph = Graph()
    ex = Namespace("http://example.org/")

    for entity, cls in valid_entities.items():
        entity_uri = ex[entity.replace(" ", "_")]
        class_uri = ex[cls]
        new_graph.add((entity_uri, RDF.type, class_uri))
        new_graph.add((entity_uri, RDFS.label, Literal(entity)))

    for subj, rel, obj in matched_triplets:
        if subj in valid_entities and obj in valid_entities:
            subj_uri = ex[subj.replace(" ", "_")]
            obj_uri = ex[obj.replace(" ", "_")]
            rel_uri = ex[rel]
            new_graph.add((subj_uri, rel_uri, obj_uri))

    return new_graph

# Main function
def main():
    # Load ontology
    ontology_file = "ontology.ttl"
    ontology_graph = load_rdf_graph(ontology_file)

    # Extract ontology triplets
    ontology_classes, ontology_properties, ontology_triplets = extract_ontology_triplets(ontology_graph)

    # Input text
    text = "Jeff was working for Amazon during 2 years."

    # Extract entities and relations
    entities, relations = extract_entities_and_relations(text)
    print("Extracted Entities:", entities)
    print("Extracted Relations:", relations)

    # Match extracted triplets to ontology
    matched_triplets = []
    for triplet in relations:
        best_match = get_best_match(triplet, ontology_triplets, threshold=0.25)
        if best_match:
            matched_triplets.append(best_match)

    print("Matched Triplets:", matched_triplets)

    # Assign best ontology classes to entities
    valid_entities = assign_classes(entities, matched_triplets, ontology_classes)
    print("Assigned Classes:", valid_entities)

    # Build and save RDF graph
    new_graph = build_rdf_graph(valid_entities, matched_triplets)
    new_graph.serialize("output_graph.ttl", format="turtle")
    print("New RDF graph saved to 'output_graph.ttl'.")

if __name__ == "__main__":
    main()
