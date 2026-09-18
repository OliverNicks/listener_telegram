import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class IntentClassifier:

  def __init__(
      self,
      model_path: str,
      tokenizer_path: str,
      buyer_anchors: list[str],
      seller_anchors: list[str],
      threshold: float = 0.55,
  ):
    self.session = ort.InferenceSession(model_path)
    self.tokenizer = Tokenizer.from_file(tokenizer_path)
    self.threshold = threshold
    self.buyer_anchors = buyer_anchors
    self.seller_anchors = seller_anchors



    # Cache matrices for both buyer and seller anchors
    self.buyer_matrix = np.vstack([self._embed(a) for a in buyer_anchors])
    self.seller_matrix = np.vstack([self._embed(a) for a in seller_anchors])

  def _mean_pooling(self, last_hidden_state, attention_mask):
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
    sum_mask = np.clip(
        np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None
    )
    return sum_embeddings / sum_mask

  def _embed(self, text: str) -> np.ndarray:
    encoding = self.tokenizer.encode(text)
    inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
    }
    outputs = self.session.run(None, inputs)
    pooled = self._mean_pooling(outputs[0], inputs["attention_mask"])
    norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
    return pooled / np.clip(norm, a_min=1e-9, a_max=None)

  def predict(self, text: str) -> tuple[bool, float, float]:
    query_vec = self._embed(text)

    # 📐 Compute max similarity against both sets
    buy_sims = np.dot(query_vec, self.buyer_matrix.T)
    sell_sims = np.dot(query_vec, self.seller_matrix.T)

    buy_score = float(np.max(buy_sims))
    sell_score = float(np.max(sell_sims))

    # Display both the matching text and its corresponding score

    best_buy_idx = np.argmax(buy_sims)
    best_sell_idx = np.argmax(sell_sims)
    best_buy_anchor = self.buyer_anchors[best_buy_idx]
    best_sell_anchor = self.seller_anchors[best_sell_idx]

    # Apply lexical bias & suppression before deciding the verdict
    text_lower = text.lower()

    if any(
        phrase in text_lower
        for phrase in ["шукаю", "де купити", "чи є в наявності", "поміряти"]
    ):
      buy_score += 0.15
      sell_score -= 0.10

    if any(
        phrase in text_lower
        for phrase in ["продам", "віддам", "розвантажую", "проліт"]
    ):
      sell_score += 0.15
      buy_score -= 0.10


    # ⚖️ Both conditions must be met
    is_buyer_inquiry = (buy_score > sell_score) and (
        buy_score >= self.threshold
    )
    is_seller_inquiry = (sell_score > buy_score) and (
            sell_score >= self.threshold
        )
    return is_buyer_inquiry, is_seller_inquiry, buy_score, sell_score, best_buy_anchor, best_sell_anchor


"""
Test

if __name__ == "__main__":
    # 1. Define anchors ⚓
    shopping_anchors = [
        "Де можна купити зимовий комбінезон на дитину?",
        "Хочу замовити теплий комбінезон",
        "Підкажіть, де знайти куртку для хлопчика?",
        "Чи є в наявності теплі дитячі рукавички?",
        "Яка ціна на дитячий одяг?",
    ]

    # 2. Instantiate the classifier 🚀
    print("⏳ Initializing classifier...")
    classifier = IntentClassifier(
        model_path="onnx_model/model.onnx",
        tokenizer_path="onnx_model/tokenizer.json",
        anchors=shopping_anchors,
        threshold=0.55,
    )
    print("✅ Ready!")

    # 3. Test cases 🧪
    test_messages = [
        "Привіт! Підкажіть ціну на дитячий комбінезон",
        "Шукаю теплі рукавички на дівчинку 5 років",
        "Погода сьогодні зовсім не для прогулянок",
        "Вчора розбили екран на телефоні, де полагодити?",
    ]

    print("\n--- Running Test Predictions ---")
    for msg in test_messages:
        is_inquiry, score = classifier.predict(msg)
        verdict = "MATCH ✅" if is_inquiry else "REJECT ❌"
        print(f"[{verdict}] (Score: {score:.4f}) -> {msg}")"""