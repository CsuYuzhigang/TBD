import numpy as np
import pandas as pd

names = ['enron']

for name in names:
    df = pd.read_csv("../processed_data/{}/ml_{}.csv".format(name, name))

    df2 = np.load("../processed_data/{}/ml_{}.npy".format(name, name))

    df3 = np.load("../processed_data/{}/ml_{}_node.npy".format(name, name))

    # Only retain the (u, v, t) pairs that first occur.
    df_unique = df.drop_duplicates(subset=['u', 'i', 'ts'], keep='first')

    df2 = df2[df_unique.index + 1]
    df2 = np.insert(df2, 0, np.zeros(df2.shape[1]), axis=0)

    # reset index
    df_unique = df_unique.reset_index(drop=True)

    df_unique = df_unique.iloc[:, 1:]

    df_unique['idx'] = df_unique.index + 1

    print('{} has {} unique nodes'.format(name, df3.shape[0] - 1))
    print('{} has {} edges'.format(name, df.shape[0]))
    print('{} has {} unique edges'.format(name, df_unique.shape[0]))

    # save
    df_unique.to_csv("./processed_data/{}/processed_{}.csv".format(name, name), index=False)
    np.save("./processed_data/{}/processed_{}.npy".format(name, name), df2)
