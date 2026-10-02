"""DNA-methylation clocks of cellular replication from the authors' releases."""

import torch

from ._base_models import pyagingModel


class CellDRIFT(pyagingModel):
    """CellDRIFT with its two PCA projections folded into the linear weights.

    The original authors set supplied NaNs to zero. Entirely absent CpGs use
    the released module means, carried in ``reference_values`` for alignment.
    """

    def preprocess(self, x):
        return torch.where(torch.isnan(x), torch.zeros_like(x), x)

    def postprocess(self, x):
        return x


class MiAge(pyagingModel):
    """The authors' five-start bounded nonlinear MiAge least-squares fit.

    L-BFGS-B is evaluated on CPU in float64 and its result returns to the input
    tensor's dtype/device. Missing beta values are omitted from the objective,
    exactly as ``na.rm=TRUE`` in the original R functions. An all-NaN sample
    therefore returns the first tied start (2008), which is not a valid
    biological estimate; callers should ensure measured CpGs are available.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("b", torch.empty(0, dtype=torch.float64))
        self.register_buffer("c", torch.empty(0, dtype=torch.float64))
        self.register_buffer("d", torch.empty(0, dtype=torch.float64))

    def set_parameters(self, b, c, d):
        self.b = torch.tensor(b, dtype=torch.float64)
        self.c = torch.tensor(c, dtype=torch.float64)
        self.d = torch.tensor(d, dtype=torch.float64)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

    def forward(self, x):
        import numpy as np
        from scipy.optimize import fmin_l_bfgs_b

        values = x.detach().to(device="cpu", dtype=torch.float64).numpy()
        if np.isinf(values).any():
            raise ValueError("MiAge beta values must be finite or NaN.")
        b, c, d = (value.detach().to(device="cpu", dtype=torch.float64).numpy() for value in (self.b, self.c, self.d))
        log_b = np.log(b)
        result = np.empty((len(values), 1), dtype=np.float64)
        for index, row in enumerate(values):
            supplied = ~np.isnan(row)
            observed, local_c, local_d = row[supplied], c[supplied], d[supplied]
            local_b, local_log = b[supplied], log_b[supplied]

            def objective(n, local_b=local_b, local_c=local_c, local_d=local_d, observed=observed, local_log=local_log):
                power = np.power(local_b, n[0] - 1)
                residual = local_c + power * local_d - observed
                value = np.sum(residual * residual)
                gradient = 2 * np.sum(residual * power * local_log * local_d)
                return value, np.array([gradient])

            best_value, best_n = np.inf, 2008.0
            # R optim defaults: lmm=5, pgtol=0, maxit=100; author factr=1.
            for start in (2008.0, 4006.0, 6004.0, 8002.0, 500.0):
                n, value, _ = fmin_l_bfgs_b(
                    objective,
                    np.array([start]),
                    bounds=[(10.0, 10000.0)],
                    m=5,
                    factr=1.0,
                    pgtol=0.0,
                    maxiter=100,
                )
                if value < best_value:
                    best_value, best_n = value, n[0]
            if not np.isfinite(best_value):
                raise ValueError("MiAge could not obtain a finite least-squares objective for this sample.")
            result[index, 0] = best_n
        return torch.as_tensor(result, dtype=x.dtype, device=x.device)
