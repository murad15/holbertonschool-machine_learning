#!/usr/bin/env python3
"""Something that function does"""


class Poisson:
    """Something that function does"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculetes pmf and else"""

        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        # factorial(k)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # e approximation (Euler's number)
        e = 2.7182818285

        return (e ** (-self.lambtha) *
                (self.lambtha ** k) /
                factorial)

    def cdf(self, k):
        """Calculetes cdf and else"""

        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
