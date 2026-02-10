#!/usr/bin/env python3
"""Something that function does"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize a Binomial distribution.

        Parameters:
        data (list): Optional list of observed data
        n (int): Number of Bernoulli trials
        p (float): Probability of success
        """

        # Case 1: data is NOT provided
        if data is None:
            # Validate n
            if n <= 0:
                raise ValueError("n must be a positive value")

            # Validate p
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        # Case 2: data IS provided
        else:
            # Validate data type
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            # Validate data length
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate p first (mean of data divided by max possible successes)
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p first
            p_estimate = 1 - (variance / mean)
            p_estimate = max(min(p_estimate, 0.999999), 0.000001)

            # Estimate n using rounded value
            n_estimate = round(mean / p_estimate)

            if n_estimate <= 0:
                raise ValueError("n must be a positive value")

            # Recalculate p using rounded n
            p_estimate = mean / n_estimate

            if p_estimate <= 0 or p_estimate >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n_estimate)
            self.p = float(p_estimate)
