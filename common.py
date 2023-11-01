import random

import numpy as np


def load_data(dataset=None, mode='train', batch_size=256, shuffle=True):
    len_dataset = len(dataset)
    index_list = np.arange(len_dataset)

    def data_generator():
        if mode == 'train' and shuffle:
            random.shuffle(index_list)

        user_id_list, user_gender_list, user_age_list, user_occupation_list = [], [], [], []
        movie_id_list, movie_title_list, movie_category_list = [], [], []
        rating_list = []

        for i, index in enumerate(index_list):
            user_id_list.append(dataset[i]['user_inf']['user_id'])
            user_gender_list.append(dataset[i]['user_info']['user_gender'])
            user_age_list.append(dataset[i]['user_info']['user_age'])
            user_occupation_list.append(dataset[i]['user_info']['user_occupation'])

            movie_id_list.append(dataset[i]['movie_info']['movie_id'])
            movie_title_list.append(dataset[i]['movie_info']['movie_title'])
            movie_category_list.append(dataset[i]['movie_info']['movie_category'])

            rating_list.append(dataset[i]['rating'])

            if len(user_id_list) >= batch_size:
                yield [
                    np.array(user_id_list),
                    np.array(user_gender_list),
                    np.array(user_age_list),
                    np.array(user_occupation_list),

                    np.array(movie_id_list),
                    np.reshape(np.array(movie_title_list), [batch_size, 15]).astype(np.int64),
                    np.reshape(np.array(movie_category_list), [batch_size, 6]).astype(np.int64),

                    np.reshape(np.array(rating_list), [-1, 1]).astype(np.float32)
                ]

                user_id_list, user_gender_list, user_age_list, user_occupation_list = [], [], [], []
                movie_id_list, movie_title_list, movie_category_list = [], [], []
                rating_list = []

    return data_generator()
