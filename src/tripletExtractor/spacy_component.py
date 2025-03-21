
from spacy import Language 
from spacy.tokens import Doc, Span
from transformers import pipeline
from typing import List
import re
import hashlib


def extract_triplets(text: str) -> List[str]:
    """
    parses the text to triplets
    1. Split the text into tokens
    2. If the token is <triplet>, <subj>, or <obj>, then set the current variable to the appropriate value
    3. If the token is not one of the above, then append it to the appropriate variable
    4. If the current variable is <subj>, then append the triplet to the list of triplets

    :param text: str - the text to be parsed
    :type text: str
    :return: A list of dictionaries.
    """

    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():

        if (token == "<triplet>"):

            current = "t"

            if (relation != ""):

                triplets.append(
                        {
                            "head": subject.strip(),
                            "type": relation.strip(),
                            "tail": object_.strip()
                            }
                        )
                relation = ""

            subject = ""

        elif (token == "<subj>"):

            current = "s"

            if (relation != ""):

                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip()
                        }
                    )

            object_ = ""

        elif (token == "<obj>"):

            current = "o"
            relation = ""

        else:

            if (current == "t"):

                subject += " " + token

            elif (current == "s"):

                object_ += " " + token

            elif (current == "o"):

                relation += " " + token

    if ((subject != "") and (relation != "") and (object_ != "")):

        triplets.append(
                {
                    "head": subject.strip(),
                    "type": relation.strip(),
                    "tail": object_.strip()
                    }
                )

    return triplets


  
@Language.factory(
        "rebel",
        requires = ["doc.sents"],
        assigns = ["doc._.rel"],
        default_config = {
            "model_name": "Babelscape/rebel-large",
            "device": 0,
            },
        )
class RebelComponent:

    def __init__(
            self,
            nlp, name,
            model_name: str,
            device: int,
        ):

        assert model_name is not None, ""

        self.triplet_extractor = pipeline(
                "text2text-generation",
                model = model_name,
                tokenizer = model_name,
                device = device
                )

        # Register custom extension on the Doc
        if (not Doc.has_extension("rel")):

            Doc.set_extension("rel", default = {})
       
       

    def _generate_triplets(self, sent: Span) -> List[List[dict]]:
        """
        1. We pass the text of the sentence to the triplet extractor.
        2. The triplet extractor returns a list of dictionaries.
        3. We extract the token ids from the dictionaries.
        4. We decode the token ids into text.
        5. We extract the triplets from the text.
        6. We return the triplets.

        The triplet extractor is a model that takes a sentence as input and returns a list of dictionaries.
        Each dictionary contains the token ids of the extracted triplets.

        The token ids are the numbers that represent the words in the sentence.
        For example, the token id of the word "the" is 2.

        The token ids are decoded into text using the tokenizer.
        The tokenizer is a model that takes a list of token ids as input and returns a list of words.

        :param sents: List[Span]
        :type sents: List[Span]
        :return: A list of lists of dicts.
        """
        output_ids = self.triplet_extractor(sent.text, return_tensors=True, return_text=False)[0]["generated_token_ids"]["output_ids"]
        extracted_text = self.triplet_extractor.tokenizer.batch_decode(output_ids[0])
        extracted_triplets = extract_triplets(extracted_text[0])
        return extracted_triplets
    
    def set_annotations(self, doc: Doc, triplets: List[dict]):
        for triplet in triplets:

            # Remove self-loops (relationships that start and end at the entity)
            if triplet['head'] == triplet['tail']:
                continue

            # manage rebel hallucinations
            head_span = re.search(triplet["head"], doc.text)
            tail_span = re.search(triplet["tail"], doc.text)
            if not head_span or not tail_span:
                continue               
           
            index = hashlib.sha1("".join([triplet['head'], triplet['tail'], triplet['type']]).encode('utf-8')).hexdigest()
            if index not in doc._.rel:
                doc._.rel[index] = {"relation": triplet["type"], "head": triplet['head'], "tail": triplet['tail']}
    
    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets)
        return doc