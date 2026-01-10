#!/usr/bin/env python3
""" qe qwe qwe qwe wqeqweqweqe """


import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """qweqeqweqw wqe qwe aqwe qwewqe qwe wqe"""

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins = range(0, 101, 10)
    plt.hist(x=student_grades, bins=bins, color='blue', edgecolor='black')
    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of students")
    plt.show()
