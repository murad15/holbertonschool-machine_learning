#!/usr/bin/env python3
""" qweqe qwe qweqwe qwe """


import numpy as np
import matplotlib.pyplot as plt


def bars():
    """qqweqwe qwe qe qewqwq"""

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # Compute bottoms for stacking
    bottoms = np.zeros(fruit.shape[1])

    for i in range(fruit.shape[0]):
        plt.bar(people, fruit[i], bottom=bottoms, color=colors[i], width=bar_width, label=fruits[i])
        bottoms += fruit[i]  # update bottom for next fruit

    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
