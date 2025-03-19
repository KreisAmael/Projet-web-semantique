import spacy
from spacy import displacy

# Load Spacy's English model
nlp = spacy.load("en_core_web_lg")

# Input text
text = "The Margherita pizza is a classic pizza. Margherita has a tomato sauce base and is topped with mozzarella. The Pepperoni pizza is another popular pizza variety, with tomato sauce, mozzarella, and pepperoni. The Savoyarde is a white base pizza, which uses a creamy garlic sauce instead of tomato sauce, topped with mozzarella, spinach, and ricotta cheese."

# Process the text
doc = nlp(text)

# Print tokens and their attributes
print("Tokens and Attributes:")
for token in doc:
    print(f"Token: {token.text} Lemma: {token.lemma_} POS: {token.pos_} Dep: {token.dep_} Head: {token.head.text}")

# Print entities
print("\nEntities:")
for ent in doc.ents:
    print(f"Entity: {ent.text} Label: {ent.label_}")

# Print dependency tree
print("\nDependency Tree:")
displacy.serve(doc, style="dep")