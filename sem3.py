"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)
        ###########################################################################
        # TODO: Строим матрицу эмбеддингов для фильмов и сохраняем в self.embeddings                       

        genre_matrix = vectorizer.fit_transform(self.movies_df["genres"]) 
        max_movie_id = self.movies_df["movieId"].max()
        n_genres = genre_matrix.shape[1]
        self.embeddings = np.zeros((max_movie_id + 1, n_genres))
        
        for idx, row in self.movies_df.iterrows():
            movie_id = row["movieId"]
            self.embeddings[movie_id] = genre_matrix[idx].toarray().flatten()

        ###########################################################################
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """

        # вектор целевого фильма
        target_vec = self.embeddings[item_id]
        
        if np.all(target_vec == 0):
            return 0.0
        
        # находим все фильмы, оцененные пользователем
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # считаем косинусное сходство с векторами оцененных фильмов
        similarities = []
        ratings = []
        
        for rated_item in rated_items:
            rated_vec = self.embeddings[rated_item]
            if np.all(rated_vec == 0):
                continue

            dot_product = np.dot(target_vec, rated_vec)
            norm_target = np.linalg.norm(target_vec)
            norm_rated = np.linalg.norm(rated_vec)
            
            if norm_target > 0 and norm_rated > 0:
                similarity = dot_product / (norm_target * norm_rated)
                similarities.append(similarity)
                ratings.append(user_ratings[rated_item])
        
        if len(similarities) == 0:
            return 0.0
        
        # отбираем топ-k похожих
        if len(similarities) > k:
            top_k_indices = np.argsort(similarities)[-k:]
            similarities = [similarities[i] for i in top_k_indices]
            ratings = [ratings[i] for i in top_k_indices]
        
        # взвешенное среднее
        weighted_sum = np.sum(np.array(similarities) * np.array(ratings))
        sum_similarities = np.sum(similarities)
        
        if sum_similarities == 0:
            return 0.0
        
        predicted_rating = weighted_sum / sum_similarities
        
        # клипируем результат
        return np.clip(predicted_rating, 0.0, 5.0)

        # raise NotImplementedError("Реализуйте функцию predict_rating")

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """

        # все фильмы, которые оценил пользователь
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return []
        
        # строим профиль пользователя как взвешенное среднее
        user_profile = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        for rated_item in rated_items:
            rating = user_ratings[rated_item]
            item_vec = self.embeddings[rated_item]
            user_profile += rating * item_vec
            total_weight += rating
        
        if total_weight > 0:
            user_profile /= total_weight
        
        if np.all(user_profile == 0):
            return []
        
        # Для всех фильмов, которые пользователь не оценил, считаем сходство
        all_items = set(range(self.embeddings.shape[0]))
        unrated_items = list(all_items - set(rated_items))
        
        similarities = []
        for item_id in unrated_items:
            item_vec = self.embeddings[item_id]
            if np.all(item_vec == 0):
                similarities.append(-1)
                continue

            dot_product = np.dot(user_profile, item_vec)
            norm_user = np.linalg.norm(user_profile)
            norm_item = np.linalg.norm(item_vec)
            
            if norm_user > 0 and norm_item > 0:
                similarity = dot_product / (norm_user * norm_item)
                similarities.append(similarity)
            else:
                similarities.append(-1)
        
        # сортируем по убыванию сходства
        unrated_items_list = list(unrated_items)
        scored_items = [(item_id, sim) for item_id, sim in zip(unrated_items_list, similarities) if sim >= 0]
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # возвращаем top-n
        return [item_id for item_id, _ in scored_items[:n_recommendations]]

        # raise NotImplementedError("Реализуйте функцию predict_items_for_user")


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
