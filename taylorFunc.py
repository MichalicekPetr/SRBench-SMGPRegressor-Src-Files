# Taylor series helper functions for smooth function interpolation.
# Currently used to convert continuous function values into a smooth triangular transition.

from __future__ import annotations

import numpy as np

class TaylorFunc:

    @staticmethod
    def x_triangle(t, N, f0=1.0):
        """
        Evaluate the triangle-wave Taylor approximation on the fractional part
        of the input and then add back the integer part.

        Example: for t = 2.64, the fractional part is 0.64 and the result is
        approximately 2.64 again when the fractional mapping returns 0.64.
        """
        x = float(t)
        whole = int(np.floor(x))
        frac = x - whole

        total_sum = 0.0
        for n in range(1, N + 1):
            numerator = np.cos((2 * n - 1) * np.pi * frac)
            denominator = (2 * n - 1) ** 2
            total_sum += numerator / denominator

        constant_factor = 4 / (np.pi ** 2)
        frac_result = 0.5 - (constant_factor * total_sum)
        return float(whole + frac_result)
