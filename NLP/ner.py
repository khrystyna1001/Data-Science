import spacy
from spacy import displacy
from spacy.tokens import Span

def main():
    nlp = spacy.load('en_core_web_sm')
    print(nlp.pipe_names)

    text = nlp("Tesla Inc is going to acquire twitter for $45 billion")

    for ent in text.ents:
        print(ent.text, " | ", ent.label_, " | ")

    # print(nlp.pipe_labels['ner'])

    s = text[3:6]
    s1 = Span(text, 0, 2, label='ORG')
    s2 = Span(text, 8, 11, label='ORG')

    text.set_ents([s1,s2], default='unmodified')

    for ent in text.ents:
        print(ent.text, " | ", ent.label_, " | ")
    
if __name__ == "__main__":
    main()