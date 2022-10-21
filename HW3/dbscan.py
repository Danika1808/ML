from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

#создаем датасет
centers = [(0, 4), (5, 8), (8, 2)]
cluster_std = [1.2, 1, 1.4]
X, y = make_blobs(n_samples=400, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=10, label="Cluster1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=10, label="Cluster2")
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=10, label="Cluster3")


# Определение класса точки
def point_type(radius, minNeigh, df, index):
    # Получаем координаты точки
    x, y = df.iloc[index]['X'], df.iloc[index]['Y']
    # Проверяем остальные точки на соседство, из подходящих формируем массив
    neighbours = df[((x - df['X'])**2 + (y - df['Y'])**2 <= radius**2) & (df.index != index)]
    #neighbours = df[((np.abs(x - df['X']) <= radius) & (np.abs(y - df['Y']) <= radius)) & (df.index != index)]
    # Определение класса точки
    if len(neighbours) >= minNeigh:
        # возвращаем соседей и тип точки. Три буля - core, border, noise
        return neighbours.index, True, False, False
    elif (len(neighbours) < minNeigh) and len(neighbours) > 0:
        return neighbours.index, False, True, False
    elif len(neighbours) == 0:
        return neighbours.index, False, False, True


def dbscan(radius, minNeigh, df):
    C = 1
    cur_clst = set()
    unvisited = list(df.index)
    total_clst = []
    # Работаем пока не останется непосещённых точек
    while len(unvisited) != 0:
        # Идентификация первой точки кластера
        first_point = True
        # Берем точку наугад, закидываем в стек
        cur_clst.add(random.choice(unvisited))
        # Пока стек не опустеет
        while len(cur_clst) != 0:
            # Берем точку
            curr_idx = cur_clst.pop()
            # Получаем данные
            neigh_indexes, iscore, isborder, isnoise = point_type(radius, minNeigh, df, curr_idx)
            if (isborder & first_point):
                total_clst.append((curr_idx, 0))
                total_clst.extend(list(zip(neigh_indexes, [0 for _ in range(len(neigh_indexes))])))
                unvisited.remove(curr_idx)
                unvisited = [e for e in unvisited if e not in neigh_indexes]
                continue
            unvisited.remove(curr_idx)
            neigh_indexes = set(neigh_indexes) & set(unvisited)
            if iscore:
                first_point = False
                total_clst.append((curr_idx, C))
                cur_clst.update(neigh_indexes)
            elif isborder:
                total_clst.append((curr_idx, C))
                continue
            elif isnoise:
                total_clst.append((curr_idx, 0))
                continue
        if not first_point:
            C += 1
    return total_clst

# Радиус
radius = 0.6
# Минимум соседей
minNeigh = 3
data = pd.DataFrame(X, columns=["X", "Y"])
clustered = dbscan(radius, minNeigh, data)
idx, cluster = list(zip(*clustered))
cluster_df = pd.DataFrame(clustered, columns=["idx", "cluster"])
plt.figure(figsize=(10, 7))
for clust in np.unique(cluster):
    plt.scatter(X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 0],
                X[cluster_df["idx"][cluster_df["cluster"] == clust].values, 1],
                s=10,
                label=f"Cluster{clust}")
plt.show()
