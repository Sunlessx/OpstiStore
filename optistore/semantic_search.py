import faiss
from sentence_transformers import SentenceTransformer

texts = {
    "A": [
        "We need to reduce our monthly cloud bill.",
        "Cutting storage expenses is a priority this quarter.",
        "How can we lower what we pay for data hosting?",
        "The invoice from our provider went up by 30 percent.",
    ],
    "B": [
        "Move old files to cheaper long-term storage.",
        "Rarely used data should be archived in Glacier.",
        "Infrequently accessed objects can go to a cold tier.",
        "Backups from five years ago are still on expensive disks.",
    ],
    "C": [
        "The same report is saved in twelve different folders.",
        "Many identical copies of the file waste space.",
        "We found duplicate photos taking up gigabytes.",
    ],
    "D": [
        "Customer records must be encrypted at rest.",
        "GDPR requires us to protect personal data.",
        "Only authorized staff may access confidential documents.",
    ],
    "E": [
        "The recipe calls for two cups of flour and three eggs.",
        "Our team won the football match on Saturday.",
        "It will rain tomorrow in the north of the country.",
        "The new smartphone has an excellent camera.",
        "She is learning to play the piano.",
        "The train to Casablanca leaves at nine.",
    ],
}

# Flatten into two aligned lists: the texts and the group of each text
all_texts = []
all_groups = []
for group, group_texts in texts.items():
    for t in group_texts:
        all_texts.append(t)
        all_groups.append(group)

# SentenceTransformer picks GPU or CPU automatically
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(all_texts, normalize_embeddings=True)
print("Embeddings shape:", embeddings.shape)

# Exact cosine search (inner product on normalized vectors)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print("Vectors in index:", index.ntotal)  # expected 20


def search(query, k=5):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q, k)
    print(f'\nQuery: "{query}"')
    for score, i in zip(scores[0], ids[0]):
        print(f"  {score:.3f}  [{all_groups[i]}] #{i:<2} {all_texts[i]}")


queries = [
    "Our hosting costs are out of control",
    "Which data can we put in the cheapest tier?",
    "Why do we keep so many copies of the same thing?",
    "Protecting people's private information",
    "Who is allowed to open sensitive files?",
    "What is the weather going to be like?",
    "How do I bake a cake?",
    "Quantum computing breakthroughs",
    "Old backups are costing us money",
    "The invoice from our provider went up by 30 percent.",
]

for q in queries:
    search(q)