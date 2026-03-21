import unittest
import numpy as np

from sem2 import (
    jaccard_similarity,
    build_user_user_matrix,
    predict_items_for_user,
    predict_rating,
)
from utils import build_user_item_matrix


class TestSeminar2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.score = 0
        # Загружаем данные один раз
        self.user_item_matrix = build_user_item_matrix()  
        self.user_user_matrix = build_user_user_matrix(self.user_item_matrix)

    def test1_jaccard_similarity(self):
        """Тест коэффициента Жаккара между двумя векторами"""
        a = np.array([1, 2, 0, 0, 5])
        b = np.array([0, 1, 3, 0, 4])

        # Общие ненулевые позиции: индексы 1 и 4 → пересечение = 2
        # Все ненулевые: a=[0,1,4], b=[1,2,4] → объединение = {0,1,2,4} → 4
        # J = 2/4 = 0.5
        sim = jaccard_similarity(a, b)
        self.assertAlmostEqual(sim, 0.5, places=6)

        # Полное совпадение
        a2 = np.array([1, 0, 3])
        b2 = np.array([2, 0, 4])
        sim2 = jaccard_similarity(a2, b2)
        self.assertAlmostEqual(sim2, 1.0, places=6)

        # Нет общих
        a3 = np.array([1, 0, 0])
        b3 = np.array([0, 2, 0])
        sim3 = jaccard_similarity(a3, b3)
        self.assertAlmostEqual(sim3, 0.0, places=6)
        print("\n" + "=" * 80)
        print("Jaccard similarity tests passed. Score +2")

    def test2_build_user_user_matrix_shape_and_values(self):
        """Тест матрицы схожести: размерность, диагональ, значения в [0,1]"""
        matrix = self.user_user_matrix

        n_users = self.user_item_matrix.shape[0]
        self.assertEqual(matrix.shape, (n_users, n_users))

        # Диагональ должна быть ~1.0 (сам с собой)
        diag = np.diag(matrix)
        np.testing.assert_allclose(diag, 1.0, atol=1e-5)

        # Все значения должны быть в [0, 1]
        self.assertTrue(np.all((matrix >= 0) & (matrix <= 1)))

        # Симметрична?
        self.assertTrue(np.allclose(matrix, matrix.T))

        # Схожесть вкусов пользователей 1 и 2: 0.01
        self.assertAlmostEqual(matrix[1, 2], 0.01, places=2)
        self.assertAlmostEqual(matrix[1, 22], 0.02, places=2)

        print("\n" + "=" * 80)
        print("User-user matrix tests passed. Score +3")

    def test3_predict_rating_basic(self):
        """Тест предсказания рейтинга"""
        user_id, item_id = 1, 1

        # Проверяем, что функция возвращает число
        rating = predict_rating(
            user_id, item_id, self.user_user_matrix, self.user_item_matrix, topk=10
        )
        self.assertIsInstance(rating, float)
        self.assertGreaterEqual(rating, 0.0)
        self.assertLessEqual(rating, 5.0)

        # Если никто не оценил фильм — должно вернуться 0.0
        fake_ratings = np.zeros_like(self.user_item_matrix)
        rating_zero = predict_rating(0, 0, self.user_user_matrix, fake_ratings, topk=10)
        self.assertEqual(rating_zero, 0.0)

        # rated videos and expected rating
        item_id = 47
        rating = predict_rating(
            user_id, item_id, self.user_user_matrix, self.user_item_matrix
        )
        self.assertAlmostEqual(rating, 4.55, places=2)

        print("\n" + "=" * 80)
        print("Predict rating tests passed. Score +3")

    def test4_predict_items_for_user(self):
        """Тест рекомендации фильмов"""
        user_id = 1
        k = 5
        recommendations = predict_items_for_user(
            user_id, self.user_user_matrix, self.user_item_matrix, k
        )

        # Проверяем длину
        self.assertEqual(len(recommendations), k)
        self.assertTrue(all(isinstance(idx, int) for idx in recommendations))

        # Индексы в допустимом диапазоне
        n_items = self.user_item_matrix.shape[1]
        self.assertTrue(all(0 <= idx < n_items for idx in recommendations))

        # Не должно быть дубликатов
        self.assertEqual(len(set(recommendations)), k)

        # Убедимся, что пользователь ещё не оценил эти фильмы
        user_rated = self.user_item_matrix[user_id] > 0
        for item_idx in recommendations:
            self.assertFalse(
                user_rated[item_idx], f"User already rated item {item_idx}"
            )

        self.assertTrue(
            all(r in [1215, 1248, 2118, 2342, 2391] for r in recommendations)
        )
        print("\n" + "=" * 80)
        print("test4_predict_items_for_user passed. Score +2")


if __name__ == "__main__":
    unittest.main()
