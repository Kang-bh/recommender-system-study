
import sys
[sys.path.append(i) for i in ['.', '..']]
print(sys.path)
from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict
import numpy as np

import os
print("OS : ", os.getcwd())


np.random.seed(0)

# 임계값 200 result
# rmse=1.081672, Precision@K= 0.133696, recall@K=0.042
# 임계값 10 reuslt
# 
# 임계값 100 result
# rmse=1.081672, Precision@K= 0.080435, recall@K=0.027


class PopularityRecommender (BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        minimum_num_rating = kwargs.get("minimum_num_rating", 10) # 임계값 변경 지점

        # 각 아이템별 평균 평갓값 계산 후 예측값으로 사용
        movie_rating_average = dataset.train.groupby("movie_id").agg({"rating" : np.mean})

        movie_rating_predict = dataset.test.merge(movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")).fillna(0)

        # 평균값 높은 10개
        # minimum_mun_rating건 이상의 평가 존재하는 영화 사용
        pred_user2items = defaultdict(list)
        user_watched_moviews = dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        movie_stats = dataset.train.groupby("movie_id").agg({"rating" : [np.size, np.mean]})
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating

        movies_sorted_by_rating = (
            movie_stats[atleast_flg].sort_values(by=("rating", "mean"), ascending=False).index.tolist()
        )

        for user_id in dataset.train.user_id.unique():
            for movie_id in movies_sorted_by_rating:
                if movie_id not in user_watched_moviews[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        
        return RecommendResult(movie_rating_predict.rating_pred, pred_user2items)

if __name__ == "__main__":
    PopularityRecommender().run_sample()