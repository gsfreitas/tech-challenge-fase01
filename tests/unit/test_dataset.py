import numpy as np
import torch
import pytest

from src.training.dataset import (
    ChurnDataset,
    build_dataloaders,
    compute_pos_weight,
)


class TestChurnDataset:
    """Testes para o Dataset PyTorch de churn."""

    def test_dataset_returns_correct_length(self, sample_features):
        """
        Given: features e target com três amostras
        When: ChurnDataset é criado
        Then: deve retornar tamanho correto
        """
        y = np.array([0, 1, 0])

        dataset = ChurnDataset(sample_features, y)

        assert len(dataset) == 3

    def test_dataset_returns_tensors(self, sample_features):
        """
        Given: Dataset criado
        When: acessa um item
        Then: retorna tensores X e y
        """
        y = np.array([0, 1, 0])

        dataset = ChurnDataset(sample_features, y)

        x_item, y_item = dataset[0]

        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)

    def test_dataset_converts_features_to_float32(self, sample_features):
        """
        Given: features numéricas
        When: Dataset é criado
        Then: X deve ser float32
        """
        y = np.array([0, 1, 0])

        dataset = ChurnDataset(sample_features, y)

        assert dataset.X.dtype == torch.float32

    def test_dataset_converts_target_to_column_vector_float32(self, sample_features):
        """
        Given: target 1D
        When: Dataset é criado
        Then: y deve ser float32 e shape [n,1]
        """
        y = np.array([0, 1, 0])

        dataset = ChurnDataset(sample_features, y)

        assert dataset.y.dtype == torch.float32
        assert dataset.y.shape == (3, 1)


# ======================
# DATALOADERS
# ======================


class TestBuildDataLoaders:
    """Testes para criação dos DataLoaders."""

    def test_build_dataloaders_returns_three_loaders(self, sample_features):
        """
        Given: dados de treino/val/test
        When: build_dataloaders é chamado
        Then: retorna três loaders
        """
        y = np.array([0, 1, 0])

        train_loader, val_loader, test_loader = build_dataloaders(
            sample_features,
            y,
            sample_features,
            y,
            sample_features,
            y,
            batch_size=2,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_build_dataloaders_dataset_sizes(self, sample_features):
        """
        Given: dados com 3 amostras
        When: loaders são criados
        Then: datasets internos têm tamanho correto
        """
        y = np.array([0, 1, 0])

        train_loader, val_loader, test_loader = build_dataloaders(
            sample_features,
            y,
            sample_features,
            y,
            sample_features,
            y,
            batch_size=2,
        )

        assert len(train_loader.dataset) == 3
        assert len(val_loader.dataset) == 3
        assert len(test_loader.dataset) == 3

    def test_train_loader_batch_shape(self, sample_features):
        """
        Given: batch_size=2
        When: pega uma batch
        Then: shapes devem estar corretos
        """
        y = np.array([0, 1, 0])

        train_loader, _, _ = build_dataloaders(
            sample_features,
            y,
            sample_features,
            y,
            sample_features,
            y,
            batch_size=2,
        )

        X_batch, y_batch = next(iter(train_loader))

        assert X_batch.shape[1] == 3
        assert y_batch.shape[1] == 1


# ======================
# POS WEIGHT
# ======================


class TestComputePosWeight:
    """Testes para cálculo do peso da classe positiva."""

    def test_compute_pos_weight_returns_tensor(self):
        """
        Given: target binário válido
        When: compute_pos_weight é chamado
        Then: retorna tensor
        """
        y_train = np.array([0, 0, 1])

        result = compute_pos_weight(y_train)

        assert isinstance(result, torch.Tensor)

    def test_compute_pos_weight_calculates_correct_ratio(self):
        """
        Given: 2 negativos e 1 positivo
        When: compute_pos_weight é chamado
        Then: retorna 2.0
        """
        y_train = np.array([0, 0, 1])

        result = compute_pos_weight(y_train)

        assert result.item() == 2.0

    def test_compute_pos_weight_has_shape_one(self):
        """
        Given: target válido
        When: compute_pos_weight é chamado
        Then: shape deve ser [1]
        """
        y_train = np.array([0, 0, 1])

        result = compute_pos_weight(y_train)

        assert result.shape == torch.Size([1])

    def test_compute_pos_weight_raises_when_no_positive(self):
        """
        Given: dataset sem classe positiva
        When: compute_pos_weight é chamado
        Then: deve levantar erro
        """
        y_train = np.array([0, 0, 0])

        with pytest.raises(ValueError):
            compute_pos_weight(y_train)
