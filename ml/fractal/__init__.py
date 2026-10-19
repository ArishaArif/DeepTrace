"""Fractal explainability branch for the deepfake detection project.

Re-exports the public inference interface so both of these work:

    from ml.fractal import predict_fractal            # the function (Deep branch)
    from fractal.predict_fractal import predict_fractal, predict_fractal_json

The model itself is loaded lazily on the first call, so importing this
package stays cheap.
"""
from .predict_fractal import predict_fractal, predict_fractal_json

__all__ = ["predict_fractal", "predict_fractal_json"]
