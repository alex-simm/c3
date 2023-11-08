import copy
from typing import Callable
import numpy as np
import tensorflow as tf
from c3.c3objs import Quantity
import c3.signal.pulse as pulse
import c3.libraries.envelopes as envelopes
from scipy.signal import resample


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
        useDrag=False,
) -> pulse.Envelope:
    """
    Creates a Gaussian pulse.
    """
    params = {
        "amp": Quantity(value=amp, min_val=0.1 * amp, max_val=10.0 * amp, unit="V"),
        "t_final": Quantity(
            value=t_final, min_val=0.8 * t_final, max_val=t_final, unit="s"
        ),
        "sigma": Quantity(value=sigma, min_val=np.minimum(0.5 * sigma, 1.2 * sigma),
                          max_val=np.maximum(0.5 * sigma, 1.2 * sigma), unit="s"),
        "xy_angle": Quantity(
            value=xy_angle, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=freq_off,
            min_val=min(0.8 * freq_off, 1.2 * freq_off),
            max_val=max(0.8 * freq_off, 1.2 * freq_off),
            unit="Hz 2pi",
        ),
    }

    if useDrag:
        params["delta"] = Quantity(value=delta, min_val=-5, max_val=5, unit="")
        return pulse.EnvelopeDrag(name="gauss", desc="Gaussian envelope", params=params,
                                  shape=envelopes.gaussian_nonorm)
    else:
        return pulse.Envelope(name="gauss", desc="Gaussian envelope", params=params, shape=envelopes.gaussian_nonorm)


def createComplexGaussianPulse(
        t_final: float,
        sigmaR: float,
        sigmaI: float,
        amp: float = 1.0,
        ampR: float = None,
        ampI: float = None,
        delta: float = -1,
        xy_angle: float = 0.0,
        freq_off: float = 0.5e6,
        useDrag=False,
) -> pulse.Envelope:
    """
    Creates a Gaussian pulse.
    """
    params = {
        "amp": Quantity(value=amp, min_val=0.5 * amp, max_val=1.5 * amp, unit="V"),
        "t_final": Quantity(
            value=t_final, min_val=0.8 * t_final, max_val=t_final, unit="s"
        ),
        "sigmaR": Quantity(value=sigmaR, min_val=np.minimum(0.5 * sigmaR, 1.2 * sigmaR),
                           max_val=np.maximum(0.5 * sigmaR, 1.2 * sigmaR), unit="s"),
        "sigmaI": Quantity(value=sigmaI, min_val=np.minimum(0.5 * sigmaI, 1.2 * sigmaI),
                           max_val=np.maximum(0.5 * sigmaI, 1.2 * sigmaI), unit="s"),
        "xy_angle": Quantity(
            value=xy_angle, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=freq_off,
            min_val=min(0.8 * freq_off, 1.2 * freq_off),
            max_val=max(0.8 * freq_off, 1.2 * freq_off),
            unit="Hz 2pi",
        ),
    }

    if ampR is not None and ampI is not None:
        params["ampR"] = Quantity(value=ampR, min_val=0.5 * ampR, max_val=1.5 * ampR, unit="s")
        params["ampI"] = Quantity(value=ampI, min_val=0.5 * ampI, max_val=1.5 * ampI, unit="s")

    if useDrag:
        params["delta"] = Quantity(value=delta, min_val=-5, max_val=5, unit="")
        return pulse.EnvelopeDrag(name="gauss", desc="Gaussian envelope", params=params,
                                  shape=envelopes.gaussian_nonorm_complex)
    else:
        return pulse.Envelope(name="gauss", desc="Gaussian envelope", params=params,
                              shape=envelopes.gaussian_nonorm_complex)


def createSinePulse(
        frequencies: np.array,
        amplitudes: np.array,
        phases: np.array,
        t_final: float,
        amp: float = 0.5,
        xy_angle: float = 0.0,
        freq_off: float = 0.5e6,
        delta: float = -1,
        useDrag=False
) -> pulse.Envelope:
    params = {
        "amp": Quantity(value=amp, min_val=0.5 * amp, max_val=1.5 * amp, unit="V"),
        "t_final": Quantity(
            value=t_final, min_val=0.9 * t_final, max_val=1.1 * t_final, unit="s"
        ),
        "xy_angle": Quantity(
            value=xy_angle, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
        ),
        "freq_offset": Quantity(
            value=freq_off,
            min_val=min(0.8 * freq_off, 1.2 * freq_off) if freq_off != 0 else -1,
            max_val=max(0.8 * freq_off, 1.2 * freq_off) if freq_off != 0 else 1,
            unit="Hz 2pi",
        ),
        "amps": Quantity(value=amplitudes, min_val=1e-2 * amplitudes, max_val=1e2 * amplitudes, unit="V"),
        "freqs": Quantity(value=frequencies, min_val=0.95 * frequencies, max_val=1.05 * frequencies, unit="Hz 2pi"),
        "phases": Quantity(value=phases, min_val=-np.pi * np.ones_like(phases), max_val=np.pi * np.ones_like(phases),
                           unit="rad"),
    }
    if useDrag:
        params["delta"] = Quantity(value=delta, min_val=-5, max_val=5, unit="")
        return pulse.EnvelopeDrag(
            name="sinus",
            desc="Sinus envelope",
            params=params,
            shape=envelopes.fourier_cos,
        )
    else:
        return pulse.Envelope(
            name="sinus",
            desc="Sinus envelope",
            params=params,
            shape=envelopes.fourier_cos,
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


def convertToDRAG(envelope: pulse.Envelope) -> pulse.Envelope:
    params = copy.deepcopy(envelope.params)
    #if "delta" not in params:
    params["delta"] = Quantity(value=1, min_val=-5, max_val=5, unit="")

    return pulse.EnvelopeDrag(
        name=envelope.name,
        desc=envelope.desc,
        params=copy.deepcopy(params),
        shape=envelope.shape,
    )


def convertFromDRAG(envelope: pulse.Envelope) -> pulse.Envelope:
    params = copy.deepcopy(envelope.params)
    if "delta" in params:
        del params["delta"]

    return pulse.Envelope(
        name=envelope.name,
        desc=envelope.desc,
        params=copy.deepcopy(params),
        shape=envelope.shape,
    )


def createPWCPulse(
        num_pieces: int,
        shape_fctn: Callable,
        t_final: float,
        amp: float = 0.5,
        xy_angle: float = 0.0,
        freq_off: float = 0.5e6,
        values: tf.Tensor = None,
) -> pulse.Envelope:
    """
    Creates a piece-wise constant envelope using the given shape function.
    """
    if values is None:
        t = tf.linspace(
            tf.convert_to_tensor(0.0, dtype=tf.float64),
            tf.convert_to_tensor(t_final, dtype=tf.float64),
            num_pieces,
        )
        values = shape_fctn(t)

    return pulse.Envelope(
        name="pwc",
        desc="PWC envelope",
        params={
            "amp": Quantity(value=amp, min_val=0.5 * amp, max_val=1.5 * amp, unit="V"),
            "t_final": Quantity(
                value=t_final, min_val=0.9 * t_final, max_val=1.1 * t_final, unit="s"
            ),
            "xy_angle": Quantity(
                value=xy_angle, min_val=-1.5 * np.pi, max_val=2.5 * np.pi, unit="rad"
            ),
            "freq_offset": Quantity(
                value=freq_off,
                min_val=0.9 * freq_off,
                max_val=1.2 * freq_off,
                unit="Hz 2pi",
            ),
            "t_bin_start": Quantity(0),
            "t_bin_end": Quantity(t_final),
            "inphase": Quantity(tf.math.real(values)),
            "quadrature": Quantity(tf.math.imag(values)) #Quantity(1e-5 * tf.ones_like(values))
        },
        shape=envelopes.pwc,
    )


def createPWCGaussianPulse(
    num_pieces: int,
    t_final: float,
    sigma: float,
    amp: float = 0.5,
    xy_angle: float = 0.0,
    freq_off: float = 0.5e6,
) -> pulse.Envelope:
    """
    Creates a piece-wise constant Gaussian pulse.
    """
    return createPWCPulse(
        num_pieces,
        lambda t: tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2)),
        t_final,
        amp,
        xy_angle,
        freq_off,
    )


'''
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
        num_pieces,
        lambda t: tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma ** 2))
        - tf.exp(-((t - t_final / 2) ** 2) / (2 * sigma2 ** 2)) * relative_amp,
        t_final
    )
'''


def createPWCConstantPulse(
    t_final: float, num_pieces: int, value: float = 0.5
) -> pulse.Envelope:
    """
    Creates a piece-wise constant envelope initialised to constant values of 0.5.
    """
    return createPWCPulse(
        num_pieces, lambda t: value * np.ones(len(t)), t_final=t_final
    )


def convertToPWC(envelope: pulse.Envelope, numPieces: int) -> pulse.Envelope:
    params = envelope.params
    # ts = tf.convert_to_tensor(np.linspace(0, params["t_final"], numPieces))
    # values = envelope.get_shape_values(ts)

    return createPWCPulse(
        numPieces,
        shape_fctn=envelope.get_shape_values,
        t_final=params["t_final"].get_value(),
        amp=params["amp"].get_value(),
        xy_angle=params["xy_angle"].get_value(),
        freq_off=params["freq_offset"].get_value(),
    )


def resamplePWC(envelope: pulse.Envelope, numPieces: int) -> pulse.Envelope:
    inPhaseOld = envelope.params["inphase"].get_value()
    inPhaseNew = resample(envelope.params["inphase"].get_value(), numPieces)
    print("inphase: ", f"{inPhaseOld.shape} ({type(inPhaseOld)})", " -> ", f"{inPhaseNew.shape} ({type(inPhaseNew)})")
    envelope.params["inphase"] = Quantity(value=tf.convert_to_tensor(inPhaseNew))

    quadratureOld = envelope.params["quadrature"].get_value()
    quadratureNew = resample(quadratureOld, numPieces)
    print("inphase: ", f"{quadratureOld.shape} ({type(quadratureOld)})", " -> ",
          f"{quadratureNew.shape} ({type(quadratureNew)})")
    envelope.params["quadrature"] = Quantity(value=tf.convert_to_tensor(quadratureOld))
    return envelope
