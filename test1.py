import unittest

from sem1 import evaluate_rec_systems, random_recommend, top_n_recommend
from utils import load_data, accuracy


class TestSeminar1(unittest.TestCase):
    def setUp(self):
        self.score = 0
        return super().setUp()

    def test1_random_recs(self):
        # Тест случайных рекомендаций
        recs = random_recommend(n_recommendations=10)
        self.assertEqual(len(recs), 10)
        self.assertTrue(all(isinstance(r, int) for r in recs))
        ratings_df, _ = load_data()
        all_movie_ids = ratings_df["movieId"].unique()
        self.assertTrue(all(r in all_movie_ids for r in recs))
        print("\n" + "=" * 80)
        print("test1_random_recs passed. Score +3")

    def test2_top_n_recs(self):
        # Тест популярных фильмов
        recs_10_50 = top_n_recommend(n_recommendations=10, min_ratings=50)
        self.assertEqual(len(recs_10_50), 10)
        # Проверяем, что хотя бы 5 из 10 рекомендаций совпадают с топ-10 по 50+ рейтингам
        ground_true_top_10_ids_50 = (318, 858, 2959, 1276, 750, 904, 1221, 48516, 1213, 912)
        self.assertTrue(
            accuracy([rec[0] for rec in recs_10_50], ground_true_top_10_ids_50) >= 0.5
        )
        # Проверяем топ-10 по 10+ рейтингам
        recs_10_10 = top_n_recommend(n_recommendations=10)
        self.assertEqual(len(recs_10_10), 10)
        ground_true_top_10_ids_10 = (1041, 3451, 1178, 1104, 2360, 1217, 318, 951, 1927, 922)
        self.assertTrue(
            accuracy([rec[0] for rec in recs_10_10], ground_true_top_10_ids_10) >= 0.5
        )
        print("\n" + "=" * 80)
        print("test2_top_n_recs passed. Score +3")

    def test3_evaluate_rec_systems(self):
        # Тест оценки системы
        metrics = evaluate_rec_systems()
        self.assertAlmostEqual(metrics["random_accuracy"], 0.1, places=2)
        self.assertAlmostEqual(metrics["popular_accuracy"], 0.1, places=2)
        metrics = evaluate_rec_systems(user_id=609)
        self.assertAlmostEqual(metrics["random_accuracy"], 0.0, places=2)
        self.assertAlmostEqual(metrics["popular_accuracy"], 0.1, places=2)
        metrics = evaluate_rec_systems(user_id=608)
        self.assertAlmostEqual(metrics["random_accuracy"], 0.2, places=2)
        self.assertAlmostEqual(metrics["popular_accuracy"], 0.1, places=2)

        print("\n" + "=" * 80)
        print("test3_evaluate_rec_systems passed. Score +4")


if __name__ == "__main__":
    unittest.main()
