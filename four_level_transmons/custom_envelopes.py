from typing import Callable
import numpy as np
import tensorflow as tf
from c3.c3objs import Quantity
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes


def createNoDriveEnvelope(t_final: float) -> pulse.Envelope:
    return pulse.Envelope(
        name="no_drive",
        params={
            "t_final": Quantity(
                value=t_final, min_val=0.9 * t_final, max_val=1.1 * t_final, unit="s"
            )
        },
        shape=envelopes.no_drive,
    )


def createGaussianPulse(
        t_final: float,
        sigma: float,
        amp: float = 0.5,
        delta: float = -1,
        xy_angle: float = 0.0,
        freq_off: float = 0.5e6,
) -> pulse.Envelope:
    """
    Creates a Gaussian pulse.
    """
    return pulse.Envelope(
        name="gauss",
        desc="Gaussian envelope",
        params={
            "amp": Quantity(value=amp, min_val=0.5 * amp, max_val=1.5 * amp, unit="V"),
            "t_final": Quantity(
                value=t_final, min_val=0.8 * t_final, max_val=1.2 * t_final, unit="s"
            ),
            "sigma": Quantity(
                value=sigma, min_val=0.5 * sigma, max_val=2 * sigma, unit="s"
            ),
            "xy_angle": Quantity(
                value=xy_angle, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
            ),
            "freq_offset": Quantity(
                value=-freq_off,
                min_val=-1.2 * freq_off,
                max_val=-0.9 * freq_off,
                unit="Hz 2pi",
            ),
            "delta": Quantity(value=delta, min_val=-5, max_val=5, unit=""),
        },
        shape=envelopes.gaussian_nonorm,
    )


'''
def createDoubleGaussianPulse(t_final: float, sigma: float, sigma2: float, relative_amp: float) -> pulse.Envelope:
    """
    Creates a superposition of two Gaussian pulses.
    """
    return pulse.Envelope(
        name="gauss",
        desc="Gaussian envelope",
        params={
            "amp": Quantity(value=3, min_val=0.2, max_val=3, unit="V"),
            "t_final": Quantity(value=t_final, min_val=0.9 * t_final, max_val=1.1 * t_final, unit="s"),
            "sigma": Quantity(value=sigma, min_val=0.5 * sigma, max_val=2 * sigma, unit="s"),
            "sigma2": Quantity(value=sigma2, min_val=0.5 * sigma2, max_val=2 * sigma2, unit="s"),
            "relative_amp": Quantity(value=relative_amp, min_val=0.2, max_val=5, unit=""),
            "xy_angle": Quantity(value=0.0, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
            "freq_offset": Quantity(value=-3e6, min_val=-6e6, max_val=2e6, unit="Hz 2pi"),
            "delta": Quantity(value=0, min_val=-5, max_val=3, unit=""),
        },
        shape=envelopes.gaussian_nonorm_double,
    )
'''


def createPWCPulse(
    t_final: float,
    num_pieces: int,
    shape_fctn: Callable,
    values: tf.Tensor = None,
    amp: float = 0.5,
) -> pulse.Envelope:
    """
    Creates a piece-wise constant envelope using the given shape function.
    """
    if values is None:
        t = tf.linspace(0.0, t_final, num_pieces)
        values = shape_fctn(t)

    return pulse.Envelope(
        name="pwc",
        desc="PWC envelope",
        params={
            "amp": Quantity(value=amp, min_val=0.2, max_val=0.6, unit="V"),
            "t_final": Quantity(
                value=t_final, min_val=0.9 * t_final, max_val=1.1 * t_final, unit="s"
            ),
            "xy_angle": Quantity(
                value=0.0, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
            ),
            "freq_offset": Quantity(
                value=-53e6, min_val=-56e6, max_val=-52e6, unit="Hz 2pi"
            ),
            "delta": Quantity(value=-1, min_val=-5, max_val=5, unit=""),
            "t_bin_start": Quantity(0),
            "t_bin_end": Quantity(t_final),
            "inphase": Quantity(values),
        },
        shape=envelopes.pwc_shape,
    )


def createPWCGaussianPulse(
    t_final: float, sigma: float, num_pieces: int, values: tf.Tensor = None
) -> pulse.Envelope:
    """
    Creates a piece-wise constant Gaussian pulse.
    """
    return createPWCPulse(
        t_final,
        num_pieces,
        lambda t: tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2)),
        values,
    )


def createPWCDoubleGaussianPulse(
    t_final: float,
    sigma: float,
    sigma2: float,
    relative_amp: float,
    num_pieces: int,
) -> pulse.Envelope:
    """
    Creates a piece-wise constant superposition of two Gaussian pulses.
    """
    return createPWCPulse(
        t_final,
        num_pieces,
        lambda t: tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))
        - tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma2 ** 2)) * relative_amp,
    )


def createPWCConstantPulse(
    t_final: float, num_pieces: int, value: float = 0.5
) -> pulse.Envelope:
    """
    Creates a piece-wise constant envelope initialised to constant values of 0.5.
    """
    return createPWCPulse(t_final, num_pieces, lambda t: value * np.ones(len(t)))
