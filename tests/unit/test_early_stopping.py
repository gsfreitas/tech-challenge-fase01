import torch
from torch import nn
import pytest

from src.models.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Testes para EarlyStopping."""

    def test_initializes_with_default_values(self):
        """
        Given: EarlyStopping instanciado com valores padrão
        When: objeto é criado
        Then: atributos iniciais devem estar corretos
        """
        early_stopping = EarlyStopping()

        assert early_stopping.patience == 10
        assert early_stopping.min_delta == 0.0
        assert early_stopping.mode == "min"
        assert early_stopping.counter == 0
        assert early_stopping.best_score is None
        assert early_stopping.should_stop is False

    def test_raises_error_for_invalid_mode(self):
        """
        Given: mode inválido
        When: EarlyStopping é instanciado
        Then: deve levantar ValueError
        """
        with pytest.raises(ValueError):
            EarlyStopping(mode="invalid")

    def test_first_score_is_improvement(self):
        """
        Given: EarlyStopping sem best_score
        When: _is_improvement é chamado
        Then: deve retornar True
        """
        early_stopping = EarlyStopping()

        result = early_stopping._is_improvement(0.5)

        assert result is True

    def test_min_mode_detects_improvement(self):
        """
        Given: EarlyStopping em modo min com best_score definido
        When: score menor que best_score é avaliado
        Then: deve retornar True
        """
        early_stopping = EarlyStopping(mode="min", min_delta=0.0)
        early_stopping.best_score = 0.5

        result = early_stopping._is_improvement(0.4)

        assert result is True

    def test_max_mode_detects_improvement(self):
        """
        Given: EarlyStopping em modo max com best_score definido
        When: score maior que best_score é avaliado
        Then: deve retornar True
        """
        early_stopping = EarlyStopping(mode="max", min_delta=0.0)
        early_stopping.best_score = 0.5

        result = early_stopping._is_improvement(0.6)

        assert result is True

    def test_call_updates_best_score_when_improves(self):
        """
        Given: modelo e score inicial
        When: EarlyStopping é chamado
        Then: best_score, best_epoch e best_state_dict devem ser atualizados
        """
        model = nn.Linear(3, 1)
        early_stopping = EarlyStopping()

        early_stopping(0.5, model)

        assert early_stopping.best_score == 0.5
        assert early_stopping.best_epoch == 1
        assert early_stopping.best_state_dict is not None
        assert early_stopping.counter == 0

    def test_call_increments_counter_when_no_improvement(self):
        """
        Given: EarlyStopping com best_score já definido
        When: novo score não melhora
        Then: counter deve aumentar
        """
        model = nn.Linear(3, 1)
        early_stopping = EarlyStopping(patience=3)

        early_stopping(0.5, model)
        early_stopping(0.6, model)

        assert early_stopping.counter == 1
        assert early_stopping.should_stop is False

    def test_should_stop_when_patience_is_reached(self):
        """
        Given: EarlyStopping com patience igual a 2
        When: score não melhora por 2 chamadas consecutivas
        Then: should_stop deve ser True
        """
        model = nn.Linear(3, 1)
        early_stopping = EarlyStopping(patience=2)

        early_stopping(0.5, model)
        early_stopping(0.6, model)
        early_stopping(0.7, model)

        assert early_stopping.should_stop is True

    def test_restore_best_without_state_dict_does_not_fail(self):
        """
        Given: EarlyStopping sem best_state_dict salvo
        When: restore_best é chamado
        Then: não deve levantar erro
        """
        model = nn.Linear(3, 1)
        early_stopping = EarlyStopping()

        early_stopping.restore_best(model)

        assert early_stopping.best_state_dict is None

    def test_restore_best_loads_best_state_dict(self):
        """
        Given: EarlyStopping com best_state_dict salvo
        When: restore_best é chamado após alteração do modelo
        Then: pesos do melhor estado devem ser restaurados
        """
        model = nn.Linear(3, 1)
        early_stopping = EarlyStopping()

        early_stopping(0.5, model)
        best_weight = model.weight.clone()

        with torch.no_grad():
            model.weight.add_(10)

        early_stopping.restore_best(model)

        assert torch.equal(model.weight, best_weight)
