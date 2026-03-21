"""Утилиты для загрузки данных и других вспомогательных функций."""

from pathlib import Path

import numpy as np
import pandas as pd

data_dir = "data/ml-latest-small"
ratings_df = pd.read_csv(Path(data_dir) / "ratings.csv")
movies_df = pd.read_csv(Path(data_dir) / "movies.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные из файлов ratings.csv и movies.csv.

    Returns:
        Кортеж из DataFrame с рейтингами и DataFrame с фильмами
    """
    return ratings_df, movies_df


def build_user_item_matrix() -> pd.DataFrame:
    """
    Строит матрицу пользователь-фильм.
    Размер матрицы (n_users + 1, n_items + 1),
    Индексы пользователей и фильмов начинаются с 1,
    Нулевые строка и столбец — для удобства доступа по ID.
    Returns:
        Матрица пользователь-фильм (в виде DataFrame)
    """
    ratings_df, movies_df = load_data()

    # Создаём отображение оригинальных ID в индексы
    user_ids = ratings_df["userId"].max() + 1
    movie_ids = movies_df["movieId"].max() + 1

    user_item_matrix = np.zeros((user_ids, movie_ids))

    for _, row in ratings_df.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        user_item_matrix[int(user_id), int(movie_id)] = row["rating"]

    return user_item_matrix


def print_user_rated_items(user_id: int, user_item_matrix: np.array) -> None:
    """
    Выводит список фильмов, которые оценил пользователь.

    Args:
        user_id: ID пользователя (начинается с 1)
        user_item_matrix: матрица пользователь-фильм
    """
    if user_id <= 0 or user_id >= user_item_matrix.shape[0]:
        raise IndexError("user_id out of bounds")
    rated_items = np.where(user_item_matrix[user_id] > 0)[0]
    for item_id in rated_items:
        print(
            f"User {user_id} rated {id_to_movie(item_id)} "
            f"with {user_item_matrix[user_id, item_id]}"
        )


def id_to_movie(movie_id: int) -> str:
    """
    Получает название фильма по ID.

    Args:
        movie_id: ID фильма

    Returns:
        Название фильма
    """
    _, movies_df = load_data()
    name = movies_df[movies_df["movieId"] == movie_id]["title"]
    if len(name) > 0:
        return name.to_string(index=False)
    else:
        return f"Unknown id {movie_id}"


def accuracy(predict: list, ground_true: list) -> float:
    """
    Проверяет, пересекаются ли два списка на заданный процент.

    Args:
        predict: список предсказанных ID
        ground_true: список истинных ID
    """
    if not predict or not ground_true:
        return False
    set_pred = set(predict)
    set_gt = set(ground_true)
    intersection = set_pred.intersection(set_gt)
    return len(intersection) / len(set_pred)  


if __name__ == "__main__":
    # Пример использования
    user_item_matrix = build_user_item_matrix()
    user_id = 2
    user_idxs = np.nonzero(user_item_matrix[user_id, :])[0]

    for idx in user_idxs:
        movie_id = idx
        rating = user_item_matrix[user_id, idx]
        print(f"User {user_id} rated {id_to_movie(movie_id)} with {rating}")
