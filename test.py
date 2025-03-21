from rdflib import Graph, Namespace, RDF

# Load the Turtle RDF Graph
turtle_data = """
@prefix ns1: <http://output.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ns1:Alice_Smith a ns1:Person ;
    rdfs:label "Alice Smith" ;
    ns1:worksFor ns1:Senior_Data_Scientist .

ns1:Global_Innovations a ns1:Organization ;
    rdfs:label "Global Innovations" ;
    ns1:locatedIn ns1:London .

ns1:John_Doe a ns1:Person ;
    rdfs:label "John Doe" ;
    ns1:worksFor ns1:Acme_Corp .

ns1:Machine_Learning_and_Cloud_Computing a ns1:Skill ;
    rdfs:label "Machine Learning and Cloud Computing" .

ns1:Tech_Solutions a ns1:Organization ;
    rdfs:label "Tech Solutions" .

ns1:the_AI_Summit a ns1:Event ;
    rdfs:label "the AI Summit" .

ns1:the_Tech_Conference_2023 a ns1:Event ;
    rdfs:label "the Tech Conference 2023" .

ns1:Acme_Corp a ns1:Organization ;
    rdfs:label "Acme Corp" ;
    ns1:locatedIn ns1:New_York .

ns1:London a ns1:Location ;
    rdfs:label "London" .

ns1:New_York a ns1:Location ;
    rdfs:label "New York" .

ns1:Senior_Data_Scientist a ns1:Organization ;
    rdfs:label "Senior Data Scientist" .
"""

# Initialize RDF Graph
g = Graph()
g.parse(data=turtle_data, format="turtle")

# Define Namespace
NS = Namespace("http://output.org/")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

# Extract all Subjects with their Classes
entity_classes = {}
for s, _, o in g.triples((None, RDF.type, None)):
    entity_classes[s] = str(o).replace(NS, "ns1:")

# Extract Triples in JSON Format
triplets = []
for s, p, o in g:
    if p == RDF.type or p == RDFS.label:
        continue  # Skip type definitions and labels

    subject_label = g.value(s, RDFS.label, default=s.split("/")[-1])  # Get Label
    range_label = g.value(o, RDFS.label, default=o.split("/")[-1])  # Get Label

    triplet = {
        "subject": str(subject_label),
        "property_type": str(p).replace(NS, "ns1:"),
        "object": str(range_label)
    }
    triplets.append(triplet)

# Print Results
import json
json.dump(triplets, open('graph_1.json', "w"))
