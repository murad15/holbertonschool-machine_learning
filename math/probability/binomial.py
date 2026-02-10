#!/usr/bin/env python3
"""Something that function does"""


class Binomial:
    """qweq qwe qwe qwe qw eqw e qweqweq"""

    def __init__(self, data=None, n=1, p=0.5):

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - variance / mean
            n = round(mean / p)
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the Probability Mass Function for k successes
        """

        # Convert k to integer if needed
        k = int(k)

        # Return 0 if k is out of range
        if k < 0 or k > self.n:
            return 0

        # Calculate combination nCk manually
        numerator = 1
        denominator = 1

        for i in range(1, k + 1):
            numerator *= self.n - (k - i)
            denominator *= i

        combination = numerator / denominator

        # Binomial PMF
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

