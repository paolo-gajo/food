import spacy
from spacy.training.example import Example

# Sample dataset: each sentence is annotated with the start and end index for each entity type
TRAIN_DATA = [
    ("Add 2 cups of milk and 1 cup of sugar.", {"entities": [(4, 5, "QUANTITY"), (6, 10, "UNIT"), (14, 18, "INGREDIENT"), (23, 24, "QUANTITY"), (25, 29, "UNIT"), (33, 38, "INGREDIENT")]}),
    ("Saute onions for 5 minutes.", {"entities": [(0, 5, "METHOD"), (6, 12, "INGREDIENT"), (16, 17, "QUANTITY"), (18, 25, "UNIT")]}),
    ("Bake the cake at 350 degrees for 30 minutes.", {"entities": [(0, 4, "METHOD"), (9, 13, "INGREDIENT"), (17, 20, "QUANTITY"), (21, 28, "UNIT"), (32, 34, "QUANTITY"), (35, 42, "UNIT")]}),
    ("Mix 3 tablespoons of olive oil.", {"entities": [(0, 3, "METHOD"), (4, 5, "QUANTITY"), (6, 17, "UNIT"), (21, 30, "INGREDIENT")]}),
    ("Boil 4 liters of water.", {"entities": [(0, 4, "METHOD"), (5, 6, "QUANTITY"), (7, 13, "UNIT"), (17, 22, "INGREDIENT")]}),
    ("Fry the potatoes in 2 tablespoons of oil.", {"entities": [(0, 3, "METHOD"), (8, 16, "INGREDIENT"), (20, 21, "QUANTITY"), (22, 33, "UNIT"), (37, 40, "INGREDIENT")]}),
    ("Steam carrots for 15 minutes.", {"entities": [(0, 5, "METHOD"), (6, 13, "INGREDIENT"), (17, 19, "QUANTITY"), (20, 27, "UNIT")]}),
    ("Roast 5 cloves of garlic.", {"entities": [(0, 5, "METHOD"), (6, 7, "QUANTITY"), (8, 14, "UNIT"), (18, 24, "INGREDIENT")]}),
    ("Pour 1/2 cup of flour.", {"entities": [(0, 4, "METHOD"), (5, 8, "QUANTITY"), (9, 13, "UNIT"), (17, 22, "INGREDIENT")]}),
    ("Chop 3 large onions.", {"entities": [(0, 4, "METHOD"), (5, 6, "QUANTITY"), (7, 12, "UNIT"), (13, 19, "INGREDIENT")]}),
    ("Spread butter on the toast.", {"entities": [(0, 6, "METHOD"), (7, 13, "INGREDIENT"), (17, 22, "INGREDIENT")]}),
    ("Grill for 10 minutes.", {"entities": [(0, 5, "METHOD"), (9, 11, "QUANTITY"), (12, 19, "UNIT")]}),
    ("Stir in the chocolate chips.", {"entities": [(0, 4, "METHOD"), (11, 20, "INGREDIENT")]}),
    ("Whisk 3 eggs and 1 cup of milk.", {"entities": [(0, 5, "METHOD"), (6, 7, "QUANTITY"), (8, 12, "UNIT"), (16, 20, "INGREDIENT"), (24, 25, "QUANTITY"), (26, 30, "UNIT"), (34, 38, "INGREDIENT")]}),
    ("Bake at 180 degrees Celsius.", {"entities": [(0, 4, "METHOD"), (8, 11, "QUANTITY"), (12, 19, "UNIT"), (20, 27, "UNIT")]}),
]

# Initialize an empty model
nlp = spacy.blank("en")

# Add NER pipeline
ner = nlp.add_pipe("ner")

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Train the model
optimizer = nlp.initialize()
for iteration in range(100):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.5, losses={}, sgd=optimizer)

# Test the trained model
test_text = "Mix 3 tablespoons of salted butter and bake for 10 minutes."
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, ent.text)
