import torch
import pytest

from src.models.mlp import ChurnMLP


class TestChurnMLP:
    """Testes para a arquitetura MLP de churn."""

    def test_model_initializes_with_default_params(self):
        """
        Given: número de features de entrada
        When: ChurnMLP é instanciado
        Then: deve criar o modelo com parâmetros padrão
        """
        model = ChurnMLP(n_features=3)

        assert model.n_features == 3
        assert model.hidden_dims == (128, 64, 32)
        assert model.dropout_rates == (0.3, 0.3, 0.2)

    def test_model_raises_error_when_hidden_dims_and_dropout_mismatch(self):
        """
        Given: hidden_dims e dropout_rates com tamanhos diferentes
        When: ChurnMLP é instanciado
        Then: deve levantar ValueError
        """
        with pytest.raises(ValueError):
            ChurnMLP(
                n_features=3,
                hidden_dims=(128, 64),
                dropout_rates=(0.3,),
            )

    def test_forward_returns_expected_shape(self, sample_features):
        """
        Given: tensor de entrada com 3 features
        When: forward é chamado
        Then: deve retornar logits com shape [batch_size, 1]
        """
        x = torch.tensor(sample_features, dtype=torch.float32)
        model = ChurnMLP(n_features=x.shape[1])

        result = model(x)

        assert result.shape == (x.shape[0], 1)

    def test_predict_proba_returns_values_between_zero_and_one(self, sample_features):
        """
        Given: tensor de entrada com 3 features
        When: predict_proba é chamado
        Then: deve retornar probabilidades entre 0 e 1
        """
        x = torch.tensor(sample_features, dtype=torch.float32)
        model = ChurnMLP(n_features=x.shape[1])

        result = model.predict_proba(x)

        assert result.shape == (x.shape[0], 1)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_count_parameters_returns_positive_integer(self):
        """
        Given: modelo MLP instanciado
        When: count_parameters é chamado
        Then: deve retornar um inteiro positivo
        """
        model = ChurnMLP(n_features=3)

        result = model.count_parameters()

        assert isinstance(result, int)
        assert result > 0
