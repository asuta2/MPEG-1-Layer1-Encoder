from parameters import *
from utils import add_db


class InsufficientBitsError(Exception):
    pass


def smr_bit_allocation(params, smr):
    """
    Calculates the bit allocation using the SMR values.

    Args:
        params (object): Object containing parameters for the psychoacoustic model.
        smr (numpy.ndarray): Array of SMR (Signal to Mask Ratio) values for each subband.

    Returns:
        numpy.ndarray: Array of bit allocations for each subband.

    Raises:
        InsufficientBitsError: If there are insufficient bits for encoding.

    """
    bit_allocation = np.zeros(N_SUBBANDS, dtype='uint8')
    bits_header = 32
    bits_alloc = 4 * N_SUBBANDS * params.nch
    bits_available = (params.nslots + params.padbit) * SLOT_SIZE - (bits_header + bits_alloc)
    bits_available //= params.nch

    if bits_available <= 2 * FRAMES_PER_BLOCK + 6:
        raise InsufficientBitsError('Insufficient bits for encoding.')

    snr = params.table.snr
    mnr = snr[bit_allocation] - smr

    while bits_available >= FRAMES_PER_BLOCK:
        subband = np.argpartition(mnr, 1)[0]

        if bit_allocation[subband] == 15:
            mnr[subband] = INF
            continue

        if bit_allocation[subband] == 0:
            bits_needed = 2 * FRAMES_PER_BLOCK + 6
        else:
            bits_needed = FRAMES_PER_BLOCK

        if bits_needed > bits_available:
            mnr[subband] = INF
            continue

        if bit_allocation[subband] == 0:
            bit_allocation[subband] = 2
        else:
            bit_allocation[subband] += 1

        bits_available -= bits_needed
        mnr[subband] = snr[bit_allocation[subband] - 1] - smr[subband]

    return bit_allocation


class TonalComponents:
    # Class to store tonal and non-tonal components.

    def __init__(self, h):
        self.spl = np.copy(h)
        self.flag = np.zeros(h.size, dtype='uint8')
        self.tonal_components = []
        self.noise_components = []


def model1(samples, params, sfindices):
    # Psychoacoustic model 1.

    table = params.table

    x = np.abs(np.fft.rfft(samples * table.hann) / FFT_SIZE)
    np.log10(x + EPS, out=x)
    x *= 20

    # Since the dynamic range of the FFT is 96 dB, we scale the mangitudes so that the maximum is 96 dB.
    x -= np.max(x)
    x += 96

    # Get the scale factors for each subband
    scf = table.scalefactor[sfindices]

    # Initialize an array to store the maximum SPL for each subband
    subband_spl = np.zeros(N_SUBBANDS)

    # Loop through each subband and calculate the maximum SPL
    for sb in range(N_SUBBANDS):
        # Get the FFT magnitudes for the current subband
        fft_magnitudes = x[int(1 + sb * SUB_SIZE): int(1 + sb * SUB_SIZE + SUB_SIZE)]
        # Calculate the maximum SPL for the current subband
        subband_spl[sb] = np.max(fft_magnitudes)
        # Calculate the minimum SPL for the current subband based on the scale factor
        min_spl = 20 * np.log10(scf[0, sb] * 32768) - 10
        # Ensure that the maximum SPL is not lower than the minimum SPL
        subband_spl[sb] = np.maximum(subband_spl[sb], min_spl)

    peaks = np.where((x[3:-6] >= x[2:-7]) & (x[3:-6] > x[4:-5]))[0] + 3

    # Identfies tonal components and mark their neighbors as non-tonal
    tonal = TonalComponents(x)
    tonal.flag[0:3] = IGNORE

    for k in peaks:
        if 2 < k < 63:
            testj = np.array([-2, 2])
        elif 63 <= k < 127:
            testj = np.array([-3, -2, 2, 3])
        else:
            testj = np.array([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
        diff = tonal.spl[k] - tonal.spl[k + testj]
        if np.all(diff >= 7):
            tonal.spl[k] = add_db(tonal.spl[k - 1:k + 2])
            tonal.flag[k + testj] = IGNORE
            tonal.flag[k] = TONE
            tonal.tonal_components.append(k)

    # Identifies non-tonal components
    for i in range(table.cbnum - 1):
        weight = 0.0
        msum = DBMIN
        for j in range(table.cbound[i], table.cbound[i + 1]):
            if tonal.flag[i] == UNSET:
                msum = add_db((tonal.spl[j], msum))
                weight += np.power(10, tonal.spl[j] / 10) * (table.bark[table.map[j]] - i)
        if msum > DBMIN:
            index = weight / np.power(10, msum / 10.0)
            center = table.cbound[i] + int(index * (table.cbound[i + 1] - table.cbound[i]))
            if tonal.flag[center] == TONE:
                center += 1
            tonal.flag[center] = NOISE
            tonal.spl[center] = msum
            tonal.noise_components.append(center)

    # Removes tonal components that are below the hearing threshold
    tonal.tonal_components = [k for k in tonal.tonal_components if tonal.spl[k] >= table.hear[table.map[k]]]

    # Removes non-tonal components that are below the hearing threshold
    tonal.noise_components = [k for k in tonal.noise_components if tonal.spl[k] >= table.hear[table.map[k]]]

    # If the tonal components are closer than 0.5 bark, the one with the lower SPL is removed
    tonal_components = tonal.tonal_components
    map_table = table.map
    bark_table = table.bark

    for i in range(len(tonal_components) - 1):
        this = tonal_components[i]
        nx = tonal_components[i + 1]
        if bark_table[map_table[this]] - bark_table[map_table[nx]] < 0.5:
            if tonal.spl[this] > tonal.spl[nx]:
                tonal.flag[nx] = IGNORE
            else:
                tonal.flag[this] = IGNORE

    tonal.tonal_components = [k for k in tonal_components if tonal.flag[k] != IGNORE]

    # Indiviudal masking thresholds are calculated for each tonal component.
    masking_tonal = []
    masking_noise = []

    # Vectorized operations for tonal masking
    for i in range(table.subsize):
        zi = table.bark[i]
        zj = table.bark[table.map[tonal.tonal_components]]
        dz = zi - zj
        avtm = -1.525 - 0.275 * zj - 4.5
        vf = np.where(dz < -1, 17 * (dz + 1) - (0.4 * x[tonal.tonal_components] + 6),
                      np.where(dz < 0, dz * (0.4 * x[tonal.tonal_components] + 6),
                               np.where(dz < 1, -17 * dz, -(dz - 1) * (17 - 0.15 * x[tonal.tonal_components]) - 17)))
        masking_tonal.append(x[tonal.tonal_components] + vf + avtm)

    # Vectorized operations for noise masking
    for i in range(table.subsize):
        zi = table.bark[i]
        zj = table.bark[table.map[tonal.noise_components]]
        dz = zi - zj
        avnm = -1.525 - 0.175 * zj - 0.5
        vf = np.where(dz < -1, 17 * (dz + 1) - (0.4 * x[tonal.noise_components] + 6),
                      np.where(dz < 0, dz * (0.4 * x[tonal.noise_components] + 6),
                               np.where(dz < 1, -17 * dz, -(dz - 1) * (17 - 0.15 * x[tonal.noise_components]) - 17)))
        masking_noise.append(x[tonal.noise_components] + vf + avnm)

    # Global masking threshold
    masking_global = np.zeros(table.subsize)
    for i in range(table.subsize):
        masking_global[i] = add_db(np.concatenate(([table.hear[i]], masking_tonal[i], masking_noise[i])))

    # Compute the masking threshold for each subband by taking the minimum of the global masking threshold
    mask = np.zeros(N_SUBBANDS)
    for sb in range(N_SUBBANDS):
        first = table.map[int(sb * SUB_SIZE)]
        after_last = table.map[int((sb + 1) * SUB_SIZE - 1)] + 1
        mask[sb] = np.min(masking_global[first:after_last])

    # Compute the signal-to-mask ratio
    smr = subband_spl - mask

    subband_bit_allocation = smr_bit_allocation(params, smr)
    return subband_bit_allocation
