from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.classifier import IntentClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "tokenizer.json"


def test_mean_pooling_and_l2_normalization_math():
    """Verify mean pooling masks padding tokens and L2 normalization produces a unit vector."""
    # Synthetic batch: 1 sentence, 3 tokens, hidden_dim = 3
    # Token 1 and 2 are active, token 3 is padding
    hidden_states = np.array([
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [100.0, 100.0, 100.0]]
    ])
    attention_mask = np.array([[1, 1, 0]])

    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    pooled = sum_embeddings / sum_mask

    # Mean of token 1 ([1, 2, 3]) and token 2 ([3, 4, 5]) should be [2, 3, 4]
    np.testing.assert_allclose(pooled[0], [2.0, 3.0, 4.0], rtol=1e-5)

    # L2 normalize
    norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
    unit_vec = pooled / np.clip(norm, a_min=1e-9, a_max=None)

    # Check L2 norm equals 1.0
    l2_norm = float(np.linalg.norm(unit_vec[0]))
    assert pytest.approx(1.0, abs=1e-6) == l2_norm


def test_embedding_dimensions_and_l2_norm():
    """Verify embeddings have 384 dimensions and unit length (L2 norm == 1.0)."""
    if MODEL_PATH.exists() and TOKENIZER_PATH.exists():
        # Live ONNX inference test
        classifier = IntentClassifier(
            model_path=str(MODEL_PATH),
            tokenizer_path=str(TOKENIZER_PATH),
            buyer_anchors=["Шукаю зимовий комбінезон"],
            seller_anchors=["Продам зимовий комбінезон"],
        )
        vec = classifier._embed("Де купити дитячий комбінезон?")

        # Check dimensionality for paraphrase-multilingual-MiniLM-L12-v2
        assert vec.shape == (1, 384)

        # Check L2 normalization
        norm = float(np.linalg.norm(vec[0]))
        assert pytest.approx(1.0, abs=1e-5) == norm
    else:
        # Mock session & tokenizer for environments without model binaries (e.g. CI)
        mock_session = MagicMock()
        mock_hidden = np.random.randn(1, 5, 384).astype(np.float32)
        mock_session.run.return_value = [mock_hidden]

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3, 4, 5]
        mock_encoding.attention_mask = [1, 1, 1, 1, 1]
        mock_encoding.type_ids = [0, 0, 0, 0, 0]
        mock_tokenizer.encode.return_value = mock_encoding

        classifier = object.__new__(IntentClassifier)
        classifier.session = mock_session
        classifier.tokenizer = mock_tokenizer

        vec = classifier._embed("Де купити дитячий комбінезон?")
        assert vec.shape == (1, 384)
        norm = float(np.linalg.norm(vec[0]))
        assert pytest.approx(1.0, abs=1e-5) == norm
