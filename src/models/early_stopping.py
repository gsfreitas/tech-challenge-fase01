"""
Early Stopping para loops de treino PyTorch.

Monitora uma métrica de validação e para o treino
quando ela não melhora por `patience` épocas consecutivas.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping com restore do melhor modelo.

    Args:
        patience: quantas epochs sem melhora antes de parar
        min_delta: melhora mínima em val_loss para contar como "melhorou"
                   (evita parar por melhorias insignificantes)
        mode: 'min' se a métrica deve diminuir (loss), 'max' se deve
              aumentar (auc, f1, etc.)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode deve ser 'min' ou 'max', recebi: {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.best_state_dict: dict[str, Any] | None = None
        self.should_stop: bool = False
        self._current_epoch = 0

    def __call__(self, score: float, model) -> None:
        """
        Avalia se o score melhorou. Atualiza contador e flag de stop.

        Args:
            score: valor da métrica monitorada na epoch atual
            model: o modelo PyTorch, para snapshot do state_dict
        """
        self._current_epoch += 1

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = self._current_epoch

            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0
            logger.debug(
                "Epoch %d: score melhorou para %.4f (best)",
                self._current_epoch,
                score,
            )
        else:
            self.counter += 1
            logger.debug(
                "Epoch %d: sem melhora (%d/%d)",
                self._current_epoch,
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping ativado na epoch %d (best=%.4f na epoch %d).",
                    self._current_epoch,
                    self.best_score,
                    self.best_epoch,
                )

    def _is_improvement(self, score: float) -> bool:
        """
        Retorna True se `score` é melhor que `self.best_score`
        """
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def restore_best(self, model) -> None:
        """
        Restaura o state_dict do melhor epoch no modelo
        """
        if self.best_state_dict is None:
            logger.warning("Nenhum best_state_dict salvo.")
            return
        model.load_state_dict(self.best_state_dict)
        logger.info(
            "Modelo restaurado para best epoch %d (score=%.4f).",
            self.best_epoch,
            self.best_score,
        )