
import sys
[sys.path.append(i) for i in ['.', '..']]

from util.models import RecommendResult, Dataset
from src.base_recommender import BaseRecommender
from collections import defaultdict, Counter
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
np.random.seed(0)

# Precision@K= 0.034783, recall@K=0.011

class AssociationRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 평가값의 임계값
        min_support = kwargs.get("min_support", 0.1)
        min_threshold = kwargs.get("min_threshold", 1)

        user_movie_matrix = dataset.train.pivot(index="user_id", columns="movie_id", values="rating")

        # 4 이상의 평갓값 : 1
        # 4 미만 평갓값 : 0
        user_movie_matrix[user_movie_matrix < 4] = 0
        user_movie_matrix[user_movie_matrix.isnull()] = 0
        user_movie_matrix[user_movie_matrix >= 4] = 1

        # 지지도 높은 영화
        freq_movies = apriori(user_movie_matrix, min_support=min_support, use_colnames=True)

        # 연관 규칙 계산 (리프트 값)
        rules = association_rules(freq_movies, metric="lift", min_threshold=min_threshold)

        # 리프트 값 통해 각 유저가 평가하지 않은 영화 10개 추천
        pred_user2items = defaultdict(list)
        user_evaluated_movies = dataset.train.groupby("user_id").agg({"movie_id" : list})["movie_id"].to_dict()

        movielens_train_high_rating = dataset.train[dataset.train.rating >= 4]

        for user_id, data in movielens_train_high_rating.groupby("user_id"):

            input_data = data.sort_values("timestamp")["movie_id"].tolist()[-5:]

            matched_flags = rules.antecedents.apply(lambda x : len(set(input_data) & x)) >= 1


            consequent_movies = [] # 귀결부 영화 리스트

            for i, row in rules[matched_flags].sort_values("lift", ascending=False).iterrows():
                consequent_movies.extend(row["consequents"])

                counter = Counter(consequent_movies)

                for movie_id, movie_cnt in counter.most_common():
                    if movie_id not in user_evaluated_movies[user_id]:
                        pred_user2items[user_id].append(movie_id)
                    
                    if len(pred_user2items[user_id]) == 10 :
                        break
            
        
        return RecommendResult(dataset.test.rating, pred_user2items)


if __name__ == "__main__":
    AssociationRecommender().run_sample()
            

# todo :creat
