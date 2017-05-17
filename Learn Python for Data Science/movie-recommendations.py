"""
A movie recommendation system.
Source: https://www.youtube.com/watch?v=9gBC9R-msAk&index=3&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create model
model = LightFM(loss='warp')

# Train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:
        known_posititves = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_ids)
        print("  Known positives:")

        for x in known_posititves[:3]:
            print("         %s" % x)

        print("  Recommended:")

        for x in top_items[:3]:
            print("         %s" % x)

sample_recommendation(model, data, [3, 25, 450])
