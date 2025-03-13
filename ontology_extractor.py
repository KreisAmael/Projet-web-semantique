import spacy
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL
from sentence_transformers import SentenceTransformer, util
import torch

# Load SpaCy's English model
nlp = spacy.load("en_core_web_lg")

# Load a pre-trained sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the RDF graph
def load_rdf_graph(file_path, format="turtle"):
    graph = Graph()
    graph.parse(file_path, format=format)
    return graph

# Extract classes from the RDF graph
def extract_classes(graph):
    classes = set()
    for s, p, o in graph.triples((None, RDF.type, RDFS.Class)):
        classes.add(s)
    for s, p, o in graph.triples((None, RDF.type, OWL.Class)):
        classes.add(s)
    return classes

# Extract properties from the RDF graph
def extract_properties(graph):
    properties = set()
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        properties.add(s)
    for s, p, o in graph.triples((None, RDF.type, OWL.ObjectProperty)):
        properties.add(s)
    for s, p, o in graph.triples((None, RDF.type, OWL.DatatypeProperty)):
        properties.add(s)
    return properties

# Extract relationships from the RDF graph
def extract_relationships(graph):
    relationships = set()
    for s, p, o in graph.triples((None, RDFS.domain, None)):
        relationships.add((s, "domain", o))
    for s, p, o in graph.triples((None, RDFS.range, None)):
        relationships.add((s, "range", o))
    return relationships

# Extract entities and relations from text using SpaCy
def extract_entities_and_relations(text):
    doc = nlp(text)
    entities = set()
    relations = set()

    # Create a mapping of entity tokens to full entity names
    entity_map = {}
    for ent in doc.ents:
        entities.add((ent.text, ent.label_))
        for token in ent:
            entity_map[token.text] = ent.text

    # Extract relationships using dependency parsing
    for token in doc:
        if token.pos_ == "VERB":  # Focus on verbs only
            subjects = set()
            objects = set()

            # Find subjects
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:  # Subject
                    subjects.add(entity_map.get(child.text, child.text))

                    # Handle conjunctions (e.g., "Alice and Bob founded OpenAI")
                    for conj in child.conjuncts:
                        subjects.add(entity_map.get(conj.text, conj.text))

                # Find direct objects
                elif child.dep_ == "dobj":  # Direct object
                    objects.add(entity_map.get(child.text, child.text))

                    # Handle conjunctions in objects (e.g., "Google and NASA")
                    for conj in child.conjuncts:
                        objects.add(entity_map.get(conj.text, conj.text))

                # Handle prepositional phrases (e.g., "for Amazon", "during 2 years")
                elif child.dep_ == "prep":  # Preposition
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":  # Object of the preposition
                            objects.add(entity_map.get(prep_child.text, prep_child.text))

            # Add relations in the format (subject, relation, object)
            for subj in subjects:
                for obj in objects:
                    relations.add((subj, token.lemma_, obj))

    return list(entities), list(relations)

# Use embeddings to map entities to ontology classes
def map_entity_to_class(entity, classes, threshold=0.5):
    # Convert the classes set to a list
    classes = list(classes)

    # Extract local names of classes
    class_names = [str(c).split("/")[-1] for c in classes]

    # Compute embeddings for the entity and classes
    entity_embedding = embedding_model.encode(entity, convert_to_tensor=True)
    class_embeddings = embedding_model.encode(class_names, convert_to_tensor=True)

    # Compute cosine similarity between the entity and classes
    similarities = util.cos_sim(entity_embedding, class_embeddings)[0]

    # Find the class with the highest similarity
    best_match_index = torch.argmax(similarities).item()
    best_match_similarity = similarities[best_match_index].item()

    # Return the best match if similarity is above the threshold
    if best_match_similarity >= threshold:
        return classes[best_match_index]
    else:
        return None

# Use embeddings to map relations to ontology properties
def map_relation_to_property(relation, properties, threshold=0.5):
    # Convert the properties set to a list
    properties = list(properties)

    # Extract local names of properties
    property_names = [str(p).split("/")[-1] for p in properties]

    # Compute embeddings for the relation and properties
    relation_embedding = embedding_model.encode(relation, convert_to_tensor=True)
    property_embeddings = embedding_model.encode(property_names, convert_to_tensor=True)

    # Compute cosine similarity between the relation and properties
    similarities = util.cos_sim(relation_embedding, property_embeddings)[0]

    # Find the property with the highest similarity
    best_match_index = torch.argmax(similarities).item()
    best_match_similarity = similarities[best_match_index].item()

    # Return the best match if similarity is above the threshold
    if best_match_similarity >= threshold:
        return properties[best_match_index]
    else:
        return None

# Build a corresponding RDF graph
def build_rdf_graph(entities, relations, ontology_graph):
    new_graph = Graph()
    ex = Namespace("http://example.org/")

    # Extract classes and properties from the ontology
    classes = extract_classes(ontology_graph)
    properties = extract_properties(ontology_graph)

    print("Classes:\n", classes)
    print("Property:\n", properties)

    # Add entities to the new graph
    for entity, label in entities:
        # Map the entity to an ontology class
        mapped_class = map_entity_to_class(entity, classes)
        if mapped_class:
            entity_uri = ex[entity.replace(" ", "_")]
            new_graph.add((entity_uri, RDF.type, mapped_class))
            new_graph.add((entity_uri, RDFS.label, Literal(entity)))

    # Add relations to the new graph
    for subj, relation, obj in relations:
        subj_uri = ex[subj.replace(" ", "_")]
        obj_uri = ex[obj.replace(" ", "_")]

        # Map the relation to an ontology property
        mapped_property = map_relation_to_property(relation, properties)
        if mapped_property:
            new_graph.add((subj_uri, mapped_property, obj_uri))

    return new_graph

# Main function
def main():
    # Load the ontology
    ontology_file = "ontology.ttl"
    ontology_graph = load_rdf_graph(ontology_file)

    # Input text
    text = "Jeff was working for Amazon during 2 years."

    # Extract entities and relations from text
    entities, relations = extract_entities_and_relations(text)
    print("Extracted Entities:", entities)
    print("Extracted Relations:", relations)

    # Build a corresponding RDF graph
    new_graph = build_rdf_graph(entities, relations, ontology_graph)

    # Save the new RDF graph
    new_graph.serialize("output_graph.ttl", format="turtle")
    print("New RDF graph saved to 'output_graph.ttl'.")

if __name__ == "__main__":
    main()