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
        # Abramowitz & Stegun approximation of erf
        def erf(z):
            t = 1.0 / (1.0 + 0.3275911 * abs(z))
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            poly = a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
            approx = 1 - poly * (2.718281828459045 ** (-z*z))
            return approx if z >= 0 else -approx

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return 0.5 * (1 + erf(z))
