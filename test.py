import spacy
from rdflib import Graph, URIRef, Namespace

# Step 1: Load Spacy's English model (use the large model for better accuracy)
nlp = spacy.load("en_core_web_lg")

# Step 2: Define a sample ontology
ex = Namespace("http://example.org/ontology#")

# Define classes and properties
ontology = {
    "Person": ex.Person,
    "Organization": ex.Organization,
    "Location": ex.Location,
    "worksAt": ex.worksAt,
    "locatedIn": ex.locatedIn,
}

# Step 3: Define a function to extract entities and relationships using Spacy
def extract_entities_and_relations(text):
    doc = nlp(text)
    entities = set()
    relations = set()

    # Create a mapping of entity tokens to full entity names
    entity_map = {}
    for ent in doc.ents:
        entities.add((ent.text, ent.label_))
        for token in ent:
            entity_map[token.text] = ent.text  # Map token to full entity name

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

                # Handle prepositional phrases (e.g., "in 2014")
                elif child.dep_ == "prep":  # Preposition
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":  # Object of the preposition
                            objects.add(entity_map.get(prep_child.text, prep_child.text))

                            # Handle compound nouns (e.g., "Mars exploration")
                            compound_noun = " ".join([comp.text for comp in prep_child.lefts if comp.dep_ == "compound"] + [prep_child.text])
                            
                            if compound_noun != prep_child.text:
                                if prep_child.text in objects:
                                    objects.remove(prep_child.text)  # Ensure no KeyError
                                objects.add(compound_noun)

            # Add relations in the format (subject, relation, object)
            for subj in subjects:
                for obj in objects:
                    relations.add((subj, token.lemma_, obj))

    return list(entities), list(relations)

# Step 4: Define a function to build the RDF graph
def build_rdf_graph(entities, relations):
    g = Graph()

    # Add entities to the graph
    for entity, label in entities:
        # Replace spaces with underscores in URIs
        entity_uri = URIRef(f"http://example.org/{entity.replace(' ', '_')}")
        if label == "PERSON":
            g.add((entity_uri, ex.type, ontology["Person"]))
        elif label == "ORG" or label == "FAC":  # Handle FAC as Organization
            g.add((entity_uri, ex.type, ontology["Organization"]))
        elif label == "GPE":  # Geo-political entity (e.g., cities, countries)
            g.add((entity_uri, ex.type, ontology["Location"]))

    # Add relationships to the graph
    for subject, verb, obj in relations:
        # Replace spaces with underscores in URIs
        subject_uri = URIRef(f"http://example.org/{subject.replace(' ', '_')}")
        obj_uri = URIRef(f"http://example.org/{obj.replace(' ', '_')}")
        if verb == "work":  # Handle "works at" relationships
            g.add((subject_uri, ontology["worksAt"], obj_uri))
        elif verb == "locate":  # Handle "located in" relationships
            g.add((subject_uri, ontology["locatedIn"], obj_uri))

    return g

# Step 5: Main workflow
def main():
    # Input English text
    text = "Jeff was working for Amazon during 2 years"

    # Step 1: Extract entities and relationships
    entities, relations = extract_entities_and_relations(text)
    print("Extracted Entities:", entities)
    print("Extracted Relations:", relations)

    # # Step 2: Build the RDF graph
    # rdf_graph = build_rdf_graph(entities, relations)

    # # Step 3: Serialize and print the RDF graph
    # print("\nRDF Graph (Turtle Format):")
    # print(rdf_graph.serialize(format="turtle"))

# Run the workflow
if __name__ == "__main__":
    main()