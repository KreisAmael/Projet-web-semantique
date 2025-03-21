import requests
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL

def call_wiki_api(item : str):
  item = item.strip()
  item = item.replace(' ', '_')
  try:
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
    data = requests.get(url).json()
    # Return the first id (Could upgrade this in the future), normalize with wikidata value 
    # Porblème avec les homonymes a bon entendeur
    return {
        "id" : data['search'][0]['id'],
        "label" : data['search'][0]['label'],
        "url" : data['search'][0]['url'],
        "description" : data['search'][0]['description'] 
    }
  except:
    return None
  


# Manage graphs
# Load the RDF graph
def load_rdf_graph(file_path, format="turtle"):
    graph = Graph()
    graph.parse(file_path, format=format)
    return graph

def get_last_part(uri):
    return str(uri).rsplit("/", 1)[-1]

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

 