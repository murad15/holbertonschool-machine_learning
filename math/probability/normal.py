#!/usr/bin/env python3
"""Something that function does"""


class Normal:
    """Represents a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            self.mean = float(sum(data) / n)
            variance = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculates the z-score for x"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value for a z-score"""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculates the PDF for x"""
        e = 2.7182818285
        pi = 3.1415926536
        coeff = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coeff * (e ** exponent)

    def cdf(self, x):
        """Calculates the CDF for x"""
        z = (x - self.mean) / self.stddev
        # series approximation for the integral part
        approx = (z - z**3 / 3 + z**5 / 10 - z**7 / 42 + z**9 / 216)
        return 0.5 + (approx / (2 * 3.1415926536 ** 0.5))
