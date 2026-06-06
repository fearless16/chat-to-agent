"""Kalman filter for smoothing response length trajectory."""

from __future__ import annotations


class ResponseKalmanFilter:
    """Kalman filter for response length trajectory smoothing.

    State vector: [length, velocity, acceleration]^T
    Observation: length (noisy measurement)

    Predict:  x_{t|t-1} = F * x_{t-1|t-1}
    Update:   x_{t|t} = x_{t|t-1} + K * (z_t - H * x_{t|t-1})
    """

    def __init__(self):
        self.F = [
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]

        self.H = [[1.0, 0.0, 0.0]]

        self.P = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]

        self.Q = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 5.0],
        ]

        self.R = [[25.0]]

        self.x = [0.0, 0.0, 0.0]

    def predict(self) -> list[float]:
        x = self.x
        x_new = [
            self.F[0][0] * x[0] + self.F[0][1] * x[1] + self.F[0][2] * x[2],
            self.F[1][0] * x[0] + self.F[1][1] * x[1] + self.F[1][2] * x[2],
            self.F[2][0] * x[0] + self.F[2][1] * x[1] + self.F[2][2] * x[2],
        ]
        self.x = x_new

        P = self.P
        F = self.F
        FT = self._transpose(F)
        FP = self._mat_mul(F, P)
        P_new = self._mat_add(self._mat_mul(FP, FT), self.Q)
        self.P = P_new

        return self.x

    def update(self, measurement: float) -> list[float]:
        H = self.H
        P = self.P
        x = self.x

        z_pred = H[0][0] * x[0] + H[0][1] * x[1] + H[0][2] * x[2]
        y = measurement - z_pred

        HP = self._mat_mul(H, P)
        HT = self._transpose(H)
        S = self._mat_mul(HP, HT)
        S[0][0] += self.R[0][0]

        S_inv = [[1.0 / max(S[0][0], 1e-10)]]
        PHT = self._mat_mul(P, HT)
        K = [[p[0] * S_inv[0][0]] for p in PHT]

        x_new = [
            x[0] + K[0][0] * y,
            x[1] + K[1][0] * y,
            x[2] + K[2][0] * y,
        ]
        self.x = x_new

        KH = self._mat_mul(K, H)
        I = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        I_KH = self._mat_sub(I, KH)
        self.P = self._mat_mul(I_KH, P)

        return self.x

    def smooth(self, measurements: list[float], times: list[float] | None = None) -> list[float]:
        """Forward-filter only smoothing (causal, real-time safe)."""
        smoothed = []
        for m in measurements:
            self.predict()
            state = self.update(m)
            smoothed.append(state[0])
        return smoothed

    def velocity(self) -> float:
        return self.x[1]

    def acceleration(self) -> float:
        return self.x[2]

    def reset(self) -> None:
        self.x = [0.0, 0.0, 0.0]
        self.P = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]

    @staticmethod
    def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        rows_a, cols_a = len(a), len(a[0])
        cols_b = len(b[0])
        result = [[0.0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                s = 0.0
                for k in range(cols_a):
                    s += a[i][k] * b[k][j]
                result[i][j] = s
        return result

    @staticmethod
    def _mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

    @staticmethod
    def _mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

    @staticmethod
    def _transpose(m: list[list[float]]) -> list[list[float]]:
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
