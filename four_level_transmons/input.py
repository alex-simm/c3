from typing import Tuple, Dict

import numpy as np
from c3.parametermap import ParameterMap as PMap
from c3.c3objs import Quantity as Qty


class SpectralData:
    def __init__(self, frequencies: np.array, amplitudes: np.array, phases: np.array):
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.phases = phases

    frequencies = None
    amplitudes = None
    phases = None

    def restrictToRange(self, lowerBound: float, upperBound: float):
        indices = np.all([self.frequencies > lowerBound, self.frequencies < upperBound], 0)
        self.frequencies = self.frequencies[indices]
        self.amplitudes = self.amplitudes[indices]
        self.phases = self.phases[indices]

    def selectLargestPeaks(self, N: int):
        if N > 0:
            indices = self.amplitudes.argsort()[-N:][::-1]
            self.frequencies = self.frequencies[indices]
            self.amplitudes = self.amplitudes[indices]
            self.phases = self.phases[indices]
        elif N == 0:
            self.frequencies = np.array([])
            self.amplitudes = np.array([])
            self.phases = np.array([])

    def selectFrequencies(self, indices: np.array):
        self.frequencies = self.frequencies[indices].flatten()
        self.amplitudes = self.amplitudes[indices].flatten()
        self.phases = self.phases[indices].flatten()

    def __str__(self):
        return f"{len(self.frequencies)} frequencies"


def getSpectralDataFromFile(filename: str, spectralRange: Tuple[float, float] = None,
                            numPeaks: Tuple[int, int] = (-1, -1),
                            includedIndices=(None, None)) -> Dict[str, SpectralData]:
    stored_pmap = PMap()
    stored_pmap.read_config(filename)
    stored_params = stored_pmap.asdict()[list(stored_pmap.asdict().keys())[0]]
    driveChannels = stored_params["drive_channels"]
    spectralData = {}
    for driveIdx, driveName in enumerate(driveChannels.keys()):
        envelope = driveChannels[driveName][f"envelope_{driveName}"]
        data = SpectralData(
            envelope.params["freqs"].get_value().numpy() / (2 * np.pi),
            envelope.params["amps"].get_value().numpy() * envelope.params["amp"].get_value().numpy() * 29,
            envelope.params["phases"].get_value().numpy()
        )

        # restrict to spectral range
        if spectralRange is not None and len(spectralRange) > 1 and spectralRange[0] >= 0 and spectralRange[1] >= 0:
            data.restrictToRange(spectralRange[0], spectralRange[1])
            print(f"Drive {driveName}: after restriction to range: {len(data.frequencies)}")

        # find peaks
        # if numPeaks is not None and numPeaks > 0:
        #    peaks = find_peaks(amps)[0]
        #    freqs = freqs[peaks]
        #    amps = amps[peaks]
        #    phases = phases[peaks]
        #    print(f"Drive {driveName}: peaks found: {len(peaks)}")

        if numPeaks[driveIdx] is not None:
            data.selectLargestPeaks(numPeaks[driveIdx])
            print(f"Drive {driveName}: num peaks {len(data.frequencies)}")

        if includedIndices[driveIdx] is not None:
            data.selectFrequencies(includedIndices[driveIdx])
            print(f"Drive {driveName}: included indices {includedIndices[driveIdx]}")

        if len(data.frequencies) > 0:
            print(f"frequencies {driveName}: {data.frequencies[0]:e} {data.frequencies[-1]:e} {len(data.frequencies)}")
        spectralData[driveName] = data
    return spectralData
