from pprint import pprint

from ingest.loaders import MultiSourceLoader

loader = MultiSourceLoader()

inputs = [
    "ai_ml_primer.pdf", "https://www.geeksforgeeks.org/java/object-oriented-programming-oops-concept-in-java/"
]

documents = loader.load(inputs)
print(f"Loaded {len(documents)} documents.")

print("\nMetadata for all loaded content:\n")
for index, document in enumerate(documents, start=1):
    print(f"Document {index}:")
    pprint(document.metadata)
    print()

print(documents[2])
