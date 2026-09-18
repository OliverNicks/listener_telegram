import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 1. Initialize session and tokenizer 📦
session = ort.InferenceSession("onnx_model/model.onnx")
tokenizer = Tokenizer.from_file("onnx_model/tokenizer.json")


def mean_pooling(last_hidden_state, attention_mask):
  """Averages token embeddings while masking out padding tokens 🎭."""
  input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
  sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
  sum_mask = np.clip(
      np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
  )
  return sum_embeddings / sum_mask


def get_embedding(text: str) -> np.ndarray:
  """Transforms raw text into a normalized sentence vector 📐."""
  encoding = tokenizer.encode(text)

  inputs = {
      "input_ids": np.array([encoding.ids], dtype=np.int64),
      "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
      "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
  }

  outputs = session.run(None, inputs)
  last_hidden_state = outputs[0]

  # Pool into a single vector
  pooled = mean_pooling(last_hidden_state, inputs["attention_mask"])

  # L2 normalization for cosine similarity
  norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
  return pooled / np.clip(norm, a_min=1e-9, a_max=None)


# 2. Define anchor and test sentences 🧪
anchor = "Підкажіть, де купити якісний комбінезон?"
candidate_a = (
    "Привіт, хочу поцікаватися, чи знає хтось де купити комбінезон на дитину?"
)
candidate_b = (
    "Вчора випадково порвали зимовий комбінезон, який щойно купили, така"
    " прикрість!"
)

# 3. Generate embeddings
vec_anchor = get_embedding(anchor)
vec_a = get_embedding(candidate_a)
vec_b = get_embedding(candidate_b)

# 4. Compute cosine similarities (dot product of unit vectors)
sim_a = np.dot(vec_anchor, vec_a.T)[0][0]
sim_b = np.dot(vec_anchor, vec_b.T)[0][0]

print(f"Similarity (Anchor vs Candidate A): {sim_a:.4f}")
print(f"Similarity (Anchor vs Candidate B): {sim_b:.4f}")