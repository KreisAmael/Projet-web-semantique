import spacy
from spacy import displacy

# Load Spacy's English model
nlp = spacy.load("en_core_web_lg")

# Input text
text = "Jeff was working for Amazon during 2 years. Jeff loves icecream. Amazon bought Twitch. Twitch hiered Simon."

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