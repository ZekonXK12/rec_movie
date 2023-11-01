import numpy as np
import pandas as pd

import paddle


class UserModel(paddle.nn.Layer):
    def __init__(self):
        super(UserModel, self).__init__()
        # user ID embedding:
        self.user_id_embedding = paddle.nn.Embedding(num_embeddings=6041, embedding_dim=32, sparse=False)
        self.user_id_linear = paddle.nn.Linear(in_features=32, out_features=32)

        # user gender embedding:
        self.user_gender_embedding = paddle.nn.Embedding(num_embeddings=2, embedding_dim=16)
        self.user_gender_linear = paddle.nn.Linear(in_features=16, out_features=16)

        # user age embedding
        self.user_age_embedding = paddle.nn.Embedding(57, 16)
        self.user_age_linear = paddle.nn.Linear(in_features=16, out_features=16)

        # user occupation embedding
        self.user_occupation_embedding = paddle.nn.Embedding(num_embeddings=21, embedding_dim=16)
        self.user_occupation_linear = paddle.nn.Linear(in_features=16, out_features=16)

        self.user_fc = paddle.nn.Linear(in_features=80, out_features=200)

    def forward(self, user_id, user_gender, user_age, user_occupation):
        user_id = self.user_id_embedding(user_id)
        user_id = self.user_id_linear(user_id)
        user_id = paddle.nn.functional.relu(user_id)

        user_gender = self.user_gender_embedding(user_gender)
        user_gender = self.user_gender_linear(user_gender)
        user_gender = paddle.nn.functional.relu(user_gender)

        user_age = self.user_age_embedding(user_age)
        user_age = self.user_age_linear(user_age)
        user_age = paddle.nn.functional.relu(user_age)

        user_occupation = self.user_occupation_embedding(user_occupation)
        user_occupation = self.user_occupation_linear(user_occupation)
        user_occupation = paddle.nn.functional.relu(user_occupation)

        user_feat = paddle.concat([user_id, user_gender, user_age, user_occupation], axis=1)
        user_feat = self.user_fc(user_feat)
        return user_feat


class MovieModel(paddle.nn.Layer):
    def __init__(self):
        super(MovieModel, self).__init__()
        # movie embedding components

        # movie ID
        self.movie_id_embedding = paddle.nn.Embedding(num_embeddings=3953, embedding_dim=32)
        self.movie_id_linear = paddle.nn.Linear(in_features=32, out_features=32)

        # movie title
        self.movie_title_embedding = paddle.nn.Embedding(5217, embedding_dim=32)
        self.movie_title_conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(2, 1),
                                                  padding=0)
        self.movie_title_conv2 = paddle.nn.Conv2D(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1,
                                                  padding=0)

        # movie year
        self.movie_year_embedding = paddle.nn.Embedding(82, embedding_dim=32)
        self.movie_year_linear = paddle.nn.Linear(in_features=32, out_features=32)

        # movie categories
        self.movie_categories_embedding = paddle.nn.Embedding(num_embeddings=19, embedding_dim=32)
        self.movie_categories_linear = paddle.nn.Linear(in_features=32, out_features=32)

        self.movie_combine_linear = paddle.nn.Linear(in_features=128, out_features=200)

    def forward(self, movie_id_tensor, movie_title_tensor, movie_year_tensor, movie_categories_tensor):
        movie_id_feat = self.movie_id_embedding(movie_id_tensor)
        movie_id_feat = self.movie_id_linear(movie_id_feat)
        movie_id_feat = paddle.nn.functional.relu(movie_id_feat)

        movie_title_feat = self.movie_title_embedding(movie_title_tensor)
        movie_title_feat = self.movie_title_conv1(movie_title_feat)
        movie_title_feat = self.movie_title_conv2(movie_title_feat)
        movie_title_feat = paddle.sum(movie_title_feat, axis=2, keepdim=False).reshape((movie_title_feat.shape[0], 32))

        movie_year_feat = self.movie_year_embedding(movie_year_tensor)
        movie_year_feat = self.movie_year_linear(movie_year_feat)
        movie_year_feat = paddle.nn.functional.relu(movie_year_feat)

        movie_categories_feat = self.movie_categories_embedding(movie_categories_tensor)
        movie_categories_feat = self.movie_categories_linear(movie_categories_feat)
        movie_categories_feat = paddle.nn.functional.relu(movie_categories_feat)
        movie_categories_feat = paddle.sum(movie_categories_feat, axis=1, keepdim=False)

        movie_feat = paddle.concat([movie_id_feat, movie_title_feat, movie_year_feat, movie_categories_feat], axis=1)
        movie_feat = self.movie_combine_linear(movie_feat)

        return movie_feat


class CompleteModel(paddle.nn.Layer):
    def __init__(self):
        super(CompleteModel, self).__init__()
        self.user_model = UserModel()
        self.movie_model = MovieModel()

    def forward(self, user_id, user_gender, user_age, user_occupation, movie_id, movie_title, movie_year,
                movie_categories):
        user_output = self.user_model(user_id, user_gender, user_age, user_occupation)
        movie_output = self.movie_model(movie_id, movie_title, movie_year, movie_categories)

        result = paddle.nn.functional.cosine_similarity(user_output, movie_output)
        result = paddle.scale(result, scale=5.0)
        return result
