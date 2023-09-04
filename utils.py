import struct

from parameters import *


class WavRead:
    def __init__(self, filename):

        self.filename = filename
        self.fp = open(self.filename, 'rb')

        if filename[-3:] == 'wav':
            self.read_header()

        if self.nbits == 8:
            self.datatype = 'int8'
        elif self.nbits == 16:
            self.datatype = 'int16'
        else:
            self.datatype = 'int32'

        self.nprocessed_samples = 0
        self.audio = [CircBuffer(FRAME_SIZE) for _ in range(self.nch)]

    def read_header(self):
        """Read header information and determine if it is a valid MP3 file with PCM audio samples."""

        buffer = self.fp.read(128)
        ind = buffer.find(b'RIFF')
        if ind == -1:
            raise Exception('Bad WAVE file.')
        ind += 4
        self.chunksize = struct.unpack('<i', buffer[ind:ind + 4])[0]
        ind = buffer.find(b'WAVE')
        if ind == -1:
            raise Exception('Bad WAVE file.')
        ind = buffer.find(b'fmt ')
        if ind == -1:
            raise Exception('Bad WAVE file.')

        ind += 4
        sbchk1sz = struct.unpack('<i', buffer[ind:ind + 4])[0]
        if sbchk1sz != 16:
            raise Exception('Unsupported WAVE file, compression used instead of PCM.')
        ind += 4
        audioformat = struct.unpack('<H', buffer[ind:ind + 2])[0]
        if audioformat != 1:
            raise Exception('Unsupported WAVE file, compression used instead of PCM.')
        ind += 2
        self.nch = struct.unpack('<H', buffer[ind:ind + 2])[0]
        ind += 2
        self.fs = struct.unpack('<I', buffer[ind:ind + 4])[0]
        ind += 4
        self.byterate = struct.unpack('<I', buffer[ind:ind + 4])[0]
        ind += 4
        self.blockalign = struct.unpack('<H', buffer[ind:ind + 2])[0]
        ind += 2
        self.nbits = struct.unpack('<H', buffer[ind:ind + 2])[0]
        if not (self.nbits in (8, 16, 32)):
            raise Exception('Unsupported WAVE file, unsupported number of bits per sample.')
        ind = buffer.find(b'data')
        if ind == -1:
            raise Exception('Unsupported WAVE file, "data" keyword not found in file.')

        ind += 4
        sbchk2sz = struct.unpack('<I', buffer[ind:ind + 4])[0]
        self.nsamples = sbchk2sz * 8 / self.nbits / self.nch
        self.fp.seek(ind + 4)

    def read_samples(self, nsamples):
        """Read desired number of samples from WAVE file and insert it in circular buffer."""

        readsize = self.nch * nsamples
        frame = np.fromfile(self.fp, self.datatype, readsize).reshape((-1, self.nch)).astype('float32') / (1 << self.nbits - 1)
        for ch in range(self.nch):
            self.audio[ch].insert(frame[:, ch])
        self.nprocessed_samples += frame.shape[0]
        return frame.shape[0]


class CircBuffer:
    def __init__(self, size):
        self.size = size
        self.pos = 0
        self.samples = np.zeros(size, dtype='float32')

    def insert(self, frame):
        length = len(frame)
        if self.pos + length <= self.size:
            self.samples[self.pos:self.pos + length] = frame
        else:
            overhead = length - (self.size - self.pos)
            self.samples = np.roll(self.samples, -overhead)
            self.samples[-overhead:] = frame[-overhead:]
            self.samples[:self.pos + length - self.size] = frame[:-overhead]
        self.pos = (self.pos + length) % self.size

    def ordered(self):
        return np.roll(self.samples, -self.pos)

    def reversed(self):
        return np.roll(self.samples, -self.pos)[::-1]


class BitStream:

    def __init__(self, size):
        self.size = size
        self.pos = 0
        self.data = np.zeros(size, dtype='uint8')

    def insert(self, data, nbits, invmsb=False):
        if invmsb:
            data = self.invertmsb(data, nbits)
        datainbytes = self.splitinbytes(data, nbits, self.pos & 0x7)
        ind = self.pos // 8
        for byte in datainbytes:
            if ind >= self.size:
                break
            self.data[ind] |= byte
            ind += 1
        self.pos += nbits

    @staticmethod
    def maskupperbits(data, nbits):
        mask = ~((0xFFFFFFFF << nbits) & 0xFFFFFFFF)
        return data & mask

    @staticmethod
    def invertmsb(data, nbits):
        mask = 1 << (nbits - 1)
        return data ^ mask

    def splitinbytes(self, data, nbits, offset):
        data = self.maskupperbits(data, nbits)
        shift = (8 - (nbits & 0x7) + 8 - offset) & 0x7
        data <<= shift
        nbits += shift
        datainbytes = ()
        loopcount = 1 + (nbits - 1) // 8
        for i in range(loopcount):
            datainbytes = (data & 0xFF,) + datainbytes
            data >>= 8
        return datainbytes


def bitstream_formatting(filename, params, allocation, scalefactor, sample):
    """Form a MPEG-1 Layer 1 bitstream and append it to output file."""

    buffer = BitStream((params.nslots + params.padbit) * 4)

    buffer.insert(params.header, 32)
    params.updateheader()

    for sb in range(N_SUBBANDS):
        for ch in range(params.nch):
            buffer.insert(np.max((allocation[ch][sb] - 1, 0)), 4)

    for sb in range(N_SUBBANDS):
        for ch in range(params.nch):
            if allocation[ch][sb] != 0:
                buffer.insert(scalefactor[ch][sb], 6)

    for s in range(FRAMES_PER_BLOCK):
        for sb in range(N_SUBBANDS):
            for ch in range(params.nch):
                if allocation[ch][sb] != 0:
                    buffer.insert(sample[ch][sb][s], allocation[ch][sb], True)

    fp = open(filename, 'a+')
    buffer.data.tofile(fp)
    fp.close()


def get_scalefactors(sbsamples, sftable):
    """
    Calculates the scale factors for each subband.

    The scale factor is the index of the first element in the scale factor table that is greater than the maximum absolute
    value of the subband samples.

    Args:
        sbsamples (numpy.ndarray): The subband samples.
        sftable (numpy.ndarray): The scale factor table.

    Returns:
        numpy.ndarray: The scale factor indices for each subband.
    """
    sfactorindices = np.zeros(sbsamples.shape[0:-1], dtype='uint8')
    sbmaxvalues = np.max(np.absolute(sbsamples), axis=1)
    for sb in range(N_SUBBANDS):
        i = 0
        while sftable[i + 1] > sbmaxvalues[sb]:
            i += 1
        sfactorindices[sb] = i
    return sfactorindices


def add_db(values):
    # Add power magnitude values
    powers = np.power(10.0, np.array(values) / 10.0)
    return 10 * np.log10(np.sum(powers) + EPS)
