import faiss
from sentence_transformers import SentenceTransformer
from tests.queries_semantic_search import texts

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