# 무작위 추천 시 성능 확인

import numpy as np
import sys
[sys.path.append(i) for i in ['.', '..']]
from src import base_recommender
from src.base_recommender import BaseRecommender
from collections import defaultdict
# from util.models import RecommendResult, Dataset
from util.models import RecommendResult, Dataset


np.random.seed(0)

class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        unique_user_ids = sorted(dataset.train.user_id.unique())
        uniuqe_movie_ids = sorted(dataset.train.movie_id.unique())

        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(uniuqe_movie_ids, range(len(uniuqe_movie_ids))))


        # user * item matrix
        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(uniuqe_movie_ids)))

        movie_rating_predict = dataset.test.copy()
        pred_results = []

        for i, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # item id가 학습용으로 나오지 않는 경우도 난수를 저장
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue

            user_index  = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        # 각 사용자에 대한 추천 영화
        # 해당 사용자가 아직 평가하지 않은 영화 중 무작위 10개
        # key : user_id value : recommended item id list
        pred_user2items = defaultdict(list)
        
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id" : list})["movie_id"].to_dict()

        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = uniuqe_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

if __name__ == "__main__":
    RandomRecommender().run_sample()