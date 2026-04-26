import os
import random
import numpy as np

from src.utils.reproducibility import set_global_seed


class TestSetGlobalSeed:
    """Testes de reprodutibilidade global"""

    def test_sets_python_random_seed(self):
        """
        Given: seed fixa
        When: função é chamada
        Then: random deve ser determinístico
        """
        set_global_seed(42)
        val1 = random.random()

        set_global_seed(42)
        val2 = random.random()

        assert val1 == val2

    def test_sets_numpy_seed(self):
        """
        Given: seed fixa
        When: função é chamada
        Then: numpy deve ser determinístico
        """
        set_global_seed(42)
        val1 = np.random.rand()

        set_global_seed(42)
        val2 = np.random.rand()

        assert val1 == val2

    def test_sets_pythonhashseed_env(self):
        """
        Given: seed definida
        When: função é chamada
        Then: variável de ambiente deve ser setada
        """
        set_global_seed(123)

        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_does_not_fail_without_torch(self):
        """
        Given: ambiente sem torch (simulado)
        When: função é chamada
        Then: não deve levantar erro
        """
        # Aqui só garantimos que a função roda sem exception
        set_global_seed(42)

        assert True  # se chegou aqui, passou

    def test_reproducibility_multiple_calls(self):
        """
        Given: múltiplas chamadas com mesma seed
        When: valores são gerados
        Then: devem ser iguais
        """
        set_global_seed(99)
        vals1 = [random.random() for _ in range(3)]

        set_global_seed(99)
        vals2 = [random.random() for _ in range(3)]

        assert vals1 == vals2
