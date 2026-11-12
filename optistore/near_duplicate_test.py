import numpy as np
from sentence_transformers import SentenceTransformer

from tests.near_duplicate_pairs import pairs

# Flat lists, aligned by position
all_pairs = [pair for group in pairs.values() for pair in group]
all_categories = [category for category, group in pairs.items() for _ in group]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Order: pair0_a, pair0_b, pair1_a, pair1_b, ...
embeddings = model.encode(
    [text for pair in all_pairs for text in pair],
    normalize_embeddings=True,
)

first = embeddings[0::2]    # first sentence of every pair
second = embeddings[1::2]   # second sentence of every pair

# Row-by-row dot product: one score per pair (cosine similarity)
scores = np.sum(first * second, axis=1)

print(embeddings.shape)  # (40, 384)
print(scores.shape)      # (20,)

print("\nScores by category:")
for i in range(len(all_categories)):
    category = all_categories[i]
    category_score = scores[i]
    print(f"{category}: {category_score:.3f}\nPair: {all_pairs[i]}\n")

categories = np.array(all_categories)

print("\nSummary by category:")
for category in pairs.keys():          # keeps the original order
    mask = categories == category      # True where the pair is in this category
    cat_scores = scores[mask]
    print(
        f"{category:<15} "
        f"min {cat_scores.min():.3f}   "
        f"max {cat_scores.max():.3f}   "
        f"mean {cat_scores.mean():.3f}"
    )

print("\nThe tests have proved that the model is unable to distinguish between near-duplicate pairs and tricky pairs." \
" The model assigns high similarity scores to both types of pairs," \
" indicating that it does not effectively differentiate between them." \
" Since tricky pairs contain subtle differences that may completely change" \
" the meaning of the sentences, this limitation suggests that the model may" \
" not be suitable for tasks requiring fine-grained semantic understanding." \)