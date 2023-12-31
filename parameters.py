import numpy as np

# Description: This file contains all the parameters used in the program that are defined by the MPEG-1 Layer 1 format.

# Input buffer size
FRAME_SIZE = 512

# FFT number of points
FFT_SIZE = 512

# Number of subbands
N_SUBBANDS = 32

# Input buffer shift size
SHIFT_SIZE = 32

# MPEG-1 Layer 1 slot size (minimum unit in a bitstream)
SLOT_SIZE = 32

# Number of frames processed in one block
FRAMES_PER_BLOCK = 12

# Subband size
SUB_SIZE = FFT_SIZE / 2 / N_SUBBANDS

# Large number representing infinity
INF = 123456

# Small number to avoid zero-division in log calculation etc.
EPS = 1e-6

# Minimum decibel value
DBMIN = -200

# Flags used to denote tonal and noise components
UNSET = 0
TONE = 1
NOISE = 2
IGNORE = 3


class Tables:

    def __init__(self, fs, bitrate):
        """Select table depending on the sampling frequency. Bitrate is needed for adjustment of minimum hearing
        threshold."""

        if fs == 44100:
            thrtable = 'D1b'
            crbtable = 'D2b'
        elif fs == 32000:
            thrtable = 'D1a'
            crbtable = 'D2a'
        elif fs == 48000:
            thrtable = 'D1c'
            crbtable = 'D2c'
        else:
            raise ValueError('Sampling frequency must be 44100, 32000 or 48000 Hz.')

        # Read ISO psychoacoustic model 1 tables containing critical band rates,
        # absolute thresholds and critical band boundaries
        freqband = np.loadtxt('tables/' + thrtable, dtype='float32')
        critband = np.loadtxt('tables/' + crbtable, dtype='uint16')

        self.cbnum = critband[-1, 0] + 1
        self.cbound = critband[:, 1]

        self.subsize = freqband.shape[0]
        self.line = freqband[:, 1].astype('uint16')
        self.bark = freqband[:, 2]
        self.hear = freqband[:, 3]
        if bitrate >= 96:
            self.hear -= 12

        self.map = np.zeros(int(FFT_SIZE / 2 + 1), dtype='uint16')
        for i in range(self.subsize - 1):
            for j in range(self.line[i], self.line[i + 1]):
                self.map[j] = i
        for j in range(self.line[self.subsize - 1], int(FFT_SIZE / 2 + 1)):
            self.map[j] = self.subsize - 1

        # Signal-to-noise ratio table, needed for bit allocation in the ISO psychoacoustic model 1.
        self.snr = np.array((0.00, 7.00, 16.00, 25.28, 31.59, 37.75, 43.84, 49.89,
                             55.93, 61.96, 67.98, 74.01, 80.03, 86.05, 92.01), dtype='float32')

        # Hann window.
        self.hann = np.hanning(FFT_SIZE) * np.sqrt(8 / 3.0)

        # MPEG-1 Layer 1 scalefactor table.
        self.scalefactor = np.loadtxt('tables/layer1scalefactors', dtype='float32')

        # MPEG-1 Layer 1 quantization coefficients.
        self.qca = np.array((0.750000000, 0.875000000, 0.937500000,
                             0.968750000, 0.984375000, 0.992187500, 0.996093750, 0.998046875,
                             0.999023438, 0.999511719, 0.999755859, 0.999877930, 0.999938965,
                             0.999969482, 0.999984741), dtype='float32')
        self.qcb = np.array((-0.250000000, -0.125000000, -0.062500000,
                             -0.031250000, -0.015625000, -0.007812500, -0.003906250, -0.001953125,
                             -0.000976563, -0.000488281, -0.000244141, -0.000122070, -0.000061035,
                             -0.000030518, -0.000015259), dtype='float32')


class EncoderParameters:
    """Parameters, tables and header of the MPEG-1 Layer 1 codec."""

    def __init__(self, fs, nch, bitrate):

        # Raises an exception if bitrate of 32 kbits/s is insufficient for encoding of stereo audio.
        if bitrate == 32 and nch == 2:
            raise Exception('Bitrate of 32 kbits/s is insufficient for encoding of stereo audio.')

        # Sets the bitrate, number of channels and sampling frequency.
        self.bitrate = bitrate
        self.nch = nch
        self.fs = fs

        # Raises an exception if the sampling frequency is not supported.
        if self.fs not in (32000, 44100, 48000):
            raise Exception('Unsupported sampling frequency.')

        # Sets the fscode based on the sampling frequency.
        self.fscode = {44100: 0b00, 48000: 0b01, 32000: 0b10}.get(fs)

        # Calculates the number of slots.
        self.nslots = 12 * bitrate * 1000 // fs

        # Sets the copyright and original flags to 0.
        self.copyright = 0
        self.original = 0

        # Sets the channel mode to 0b11 if the number of channels is 1, else 0b10.
        self.chmode = 0b11 if self.nch == 1 else 0b10

        # Sets the modext to 0b10.
        self.modext = 0b10

        # Sets the syncword, mpegversion, layer, crc, and emphasis.
        self.syncword = 0b11111111111
        self.mpegversion = 0b11
        self.layer = 0b11
        self.crc = 0b1
        self.emphasis = 0b00

        # Sets the padbit and rest to 0.
        self.padbit = 0
        self.rest = 0

        # Sets the header based on the above parameters.
        self.header = (self.syncword << 21 | self.mpegversion << 19 |
                       self.layer << 17 | self.crc << 16 |
                       self.bitrate << 7 | self.fscode << 10 |
                       self.padbit << 9 | self.chmode << 6 |
                       self.modext << 4 | self.copyright << 3 |
                       self.original << 2 | self.emphasis)

        # Initializes the Tables class with the sampling frequency and bitrate.
        self.table = Tables(self.fs, bitrate)

    def updateheader(self):
        """Update padbit in header for current frame."""

        self.needpadding()
        if self.padbit:
            self.header |= 0x00000200
        else:
            self.header &= 0xFFFFFDFF

    def needpadding(self):
        """To ensure the constant bitrate, for fs=44100 padding is sometimes needed."""

        dif = (self.bitrate * 1000 * 12) % self.fs
        self.rest -= dif
        if self.rest < 0:
            self.rest += self.fs
            self.padbit = 1
        else:
            self.padbit = 0


def filter_coeffs():
    """Baseband subband filter prototype coefficients."""

    return np.loadtxt('tables/LPfilterprototype', dtype='float32')
