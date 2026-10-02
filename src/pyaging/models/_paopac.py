"""PAOPAC Conventional: public author trees and cohort-dependent inference."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from ._base_models import pyagingModel


class PAOPACTrees(torch.nn.Module):
    """Evaluate the author's numerical LightGBM trees without a LightGBM runtime.

    The builder accepts only the numerical, default-left splits in this pinned
    release. Leaf nodes point to themselves, so all trees can traverse together.
    """

    def __init__(self, features, thresholds, left, right, values, roots, depth):
        super().__init__()
        for name, value, dtype in (
            ("split_features", features, torch.int64),
            ("thresholds", thresholds, torch.float64),
            ("left", left, torch.int64),
            ("right", right, torch.int64),
            ("values", values, torch.float64),
            ("roots", roots, torch.int64),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        self.depth = depth

    def forward(self, x):
        nodes = self.roots.expand(x.shape[0], -1)
        rows = torch.arange(x.shape[0], device=x.device)[:, None]
        for _ in range(self.depth):
            value = x[rows, self.split_features[nodes]]
            # decision_type=2 has missing_type=None: LightGBM treats NaN as 0.
            value = torch.where(torch.isnan(value), 0.0, value)
            go_left = value <= self.thresholds[nodes]
            nodes = torch.where(go_left, self.left[nodes], self.right[nodes])
        return self.values[nodes].sum(dim=1, keepdim=True)


class PAOPAC(pyagingModel):
    """The author's Conventional output, including cohort LOWESS correction.

    Supply Olink NPX proteins and chronological ``age`` in years. Proteins are
    averaged across exact duplicate names, then uppercased, as in the release.
    Distinct names that collide after uppercasing are rejected.
    ``TDI`` is an optional column in the protein matrix; the author interface
    exponentiates and standardizes it like a protein. Metadata covariates are
    not joined into the model matrix. Absent TDI therefore becomes zero after
    normalization. Ages are required for the final cohort correction.

    Cohort context is local to each prediction call, never stored on the model.
    Calling ``forward`` directly treats its complete tensor as one cohort.
    """

    def validate_inputs(self, adata):
        if not any(str(name).upper() == "AGE" for name in adata.var_names):
            raise ValueError("PAOPAC requires chronological age in an 'age' input column (years).")

    def prepare_input_frame(self, adata):
        frame = adata.to_df()
        if frame.columns.has_duplicates:
            frame = frame.T.groupby(level=0, sort=False).mean().T
        frame.columns = frame.columns.str.upper()
        if frame.columns.has_duplicates:
            raise ValueError("PAOPAC feature names collide after uppercasing; cannot reindex duplicate labels.")
        return frame.rename(columns={"AGE": "age"})

    @staticmethod
    def _linear_abundance(matrix):
        # np.power, then fillna(0): missing NPX must not become NPX=0 (2**0=1).
        values = np.power(2.0, matrix[:, :-1].detach().cpu().numpy())
        return np.where(np.isnan(values), 0.0, values)

    def prepare_cohort_context(self, matrix):
        scaler = StandardScaler().fit(self._linear_abundance(matrix))
        return torch.from_numpy(np.stack([scaler.mean_, scaler.scale_]))

    def predict_with_cohort_context(self, batch, context):
        # Keep the original NumPy/StandardScaler arithmetic on the host and the
        # tree traversal on the requested device. Only a batch is transferred.
        values = self._linear_abundance(batch)
        parameters = context.detach().cpu().numpy()
        values = (values - parameters[0]) / parameters[1]
        return self.base_model(torch.as_tensor(values, dtype=torch.float64, device=batch.device))

    def postprocess_cohort(self, predictions, matrix):
        raw = predictions.detach().cpu().numpy().reshape(-1)
        ages = matrix[:, -1].detach().cpu().numpy()
        # Pin the author's statsmodels 0.14.4 defaults explicitly.
        correction = lowess(
            raw - ages, ages, frac=0.75, it=3, delta=0.0, is_sorted=False, missing="drop", return_sorted=False
        )
        return torch.as_tensor(raw - correction, dtype=predictions.dtype, device=predictions.device).reshape_as(
            predictions
        )

    def forward(self, x):
        context = self.prepare_cohort_context(x).to(x.device)
        return self.postprocess_cohort(self.predict_with_cohort_context(x, context), x)

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x
