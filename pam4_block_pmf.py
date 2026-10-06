"""Approximate PAM4 block bit-error PMF from pairwise competitors.

SNR means Es/N0 at the input of a unit-energy, real two-tap channel.
Symbols are [-3, -1, 1, 3], so Es=5 and sample noise variance=N0/2.
First and last symbols differ; interior symbols may match. Enumerated spans
may therefore contain several diverge/remerge events. Pairwise probabilities
overlap. The final convolution assumes independent identical experiments.
These are modeling approximations, not an exact MLSE error distribution.
"""

import itertools
import math
from pathlib import Path

import numpy as np

# Edit these parameters, then run: python3 pam4_block_pmf.py
SNR_DB = 18.3                 # Symbol SNR: Es/N0 in dB
CHANNEL_TAPS = [1, 0.85]      # Exactly two real taps; normalized automatically
BLOCK_LENGTH = 1000            # Number of independent experiments
MAX_SPAN = 8                  # Maximum competitor span in PAM4 symbols (positive integer)
SIMULATION_BLOCKS = 1_000_000  # Blocks used to estimate binomial p
RANDOM_SEED = 12345
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'


def pair_multiplicities():
    """Count valid ordered symbol pairs by normalized difference and bit weight.

    Aggregation is exactly equivalent to explicitly enumerating every pair,
    since distance depends on differences and Gray weight is additive.
    """
    gray = (0b00, 0b01, 0b11, 0b10)
    counts = np.zeros((7, 3), dtype=np.int64)
    for actual in range(4):
        for estimated in range(4):
            difference = actual - estimated
            bits = bin(gray[actual] ^ gray[estimated]).count('1')
            counts[difference + 3, bits] += 1
    return counts


def enumerate_differences(length):
    """Return difference sequences; only endpoints must be nonzero."""
    nonzero = (-3, -2, -1, 1, 2, 3)
    choices = [nonzero] if length == 1 else [nonzero] + [range(-3, 4)] * (length - 2) + [nonzero]
    return np.asarray(list(itertools.product(*choices)), dtype=np.int64)


def bit_weight_multiplicities(differences, counts):
    """For every difference sequence, count pairs at each total Gray bit weight."""
    weights = np.ones((len(differences), 1), dtype=np.int64)
    for column in differences.T:
        local = counts[column + 3]
        updated = np.zeros((len(differences), weights.shape[1] + 2), dtype=np.int64)
        for bits in range(3):
            updated[:, bits:bits + weights.shape[1]] += weights * local[:, bits, None]
        weights = updated
    return weights


def squared_output_distances(differences, taps):
    """Convolve symbol differences (2*epsilon) with both taps, including tails."""
    delta = 2.0 * differences
    output = np.zeros((len(delta), delta.shape[1] + 1))
    output[:, :-1] += taps[0] * delta
    output[:, 1:] += taps[1] * delta
    return np.sum(output * output, axis=1)


def pairwise_probabilities(distances_squared, snr_db):
    """E3 = Q(D/(2*sigma)); sigma^2 = 5/(2*EsN0)."""
    es_n0 = 10.0 ** (snr_db / 10.0)
    sigma_squared = 5.0 / (2.0 * es_n0)
    arguments = np.sqrt(distances_squared / (8.0 * sigma_squared))
    return np.fromiter((0.5 * math.erfc(float(x)) for x in arguments), float, count=len(arguments))


def experiment_pmf(snr_db, taps, max_span=6):
    """Group weighted E3 contributions by bit count; assign residual mass to zero.

    E2 is absorbed by valid-pair enumeration. Each transmitted span has prior
    probability 4**(-length). Returns PMF and a breakdown by span length.
    """
    counts = pair_multiplicities()
    pmf = np.zeros(2 * max_span + 1)
    by_span = []
    for length in range(1, max_span + 1):
        differences = enumerate_differences(length)
        weights = bit_weight_multiplicities(differences, counts)
        distances = squared_output_distances(differences, taps)
        probabilities = pairwise_probabilities(distances, snr_db)
        contribution = np.sum(probabilities[:, None] * weights, axis=0) / (4.0 ** length)
        pmf[:len(contribution)] += contribution
        by_span.append(contribution)
    mass = float(pmf[1:].sum())
    if mass > 1.0:
        raise ValueError(f"Pairwise error mass is {mass:.6g} > 1; cannot form a PMF. Increase SNR or revise the approximation.")
    pmf[0] = 1.0 - mass
    return pmf, by_span


def block_pmf(experiment, block_length):
    """N-fold convolution for the total errors from N iid experiments."""
    result = np.array([1.0])
    for _ in range(block_length):
        result = np.convolve(result, experiment)
    return result


def simulate_bit_error_rate(experiment, block_length, num_blocks, seed):
    """Sample num_blocks*N iid contributions; estimate errors/(2*N*num_blocks).

    Multinomial counts have exactly the same distribution as a histogram of
    individual samples. No channel/Viterbi simulation is performed here.
    """
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(block_length * num_blocks, experiment / experiment.sum())
    errors = sum(k * int(count) for k, count in enumerate(counts))
    probability = errors / (2.0 * block_length * num_blocks)
    return probability, errors


def binomial_pmf(trials, probability):
    """Binomial probabilities evaluated in log space without SciPy."""
    result = np.zeros(trials + 1)
    if probability == 0:
        result[0] = 1
    elif probability == 1:
        result[-1] = 1
    else:
        for k in range(trials + 1):
            log_p = (math.lgamma(trials + 1) - math.lgamma(k + 1)
                     - math.lgamma(trials - k + 1)
                     + k * math.log(probability)
                     + (trials - k) * math.log1p(-probability))
            result[k] = math.exp(log_p)
    return result


def plot_block_comparison(block, binomial, probability, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for pmf, label, style in [(block, 'Enumerated model + N-fold convolution', '-o'),
                              (binomial, f'Binomial(2N, p={probability:.4g})', '--s')]:
        support = np.flatnonzero(pmf > 1e-14)
        last = int(support[-1]) if len(support) else 0
        y = pmf[:last + 1].copy()
        y[y <= 0] = np.nan
        ax.semilogy(np.arange(last + 1), y, style, linewidth=1.5, markersize=3, label=label)
    ax.set_ylim(1e-14, 1.2)
    ax.set(xlabel='Number of bit errors in block', ylabel='Probability',
           title=f'Block PMF; N={BLOCK_LENGTH} PAM4 symbols, Es/N0={SNR_DB:g} dB')
    ax.grid(axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    if not isinstance(MAX_SPAN, int) or MAX_SPAN < 1:
        raise ValueError('MAX_SPAN must be a positive integer')
    if BLOCK_LENGTH < 1 or not math.isfinite(SNR_DB):
        raise ValueError('Require BLOCK_LENGTH >= 1 and finite SNR_DB')
    taps = np.asarray(CHANNEL_TAPS, dtype=float)
    if taps.shape != (2,):
        raise ValueError('CHANNEL_TAPS must contain exactly two real taps')
    if not np.all(np.isfinite(taps)) or np.linalg.norm(taps) == 0:
        raise ValueError('Channel taps must be finite and nonzero as a vector')
    taps /= np.linalg.norm(taps)
    if not isinstance(SIMULATION_BLOCKS, int) or SIMULATION_BLOCKS < 1:
        raise ValueError('SIMULATION_BLOCKS must be a positive integer')
    experiment, _ = experiment_pmf(SNR_DB, taps, MAX_SPAN)
    block = block_pmf(experiment, BLOCK_LENGTH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probability, sampled_errors = simulate_bit_error_rate(
        experiment, BLOCK_LENGTH, SIMULATION_BLOCKS, RANDOM_SEED)
    if not 0 <= probability <= 1:
        raise ValueError('Simulated bit-error rate is outside [0,1]; approximation is invalid')
    binomial = binomial_pmf(2 * BLOCK_LENGTH, probability)
    for name, pmf in [('block_pmf', block), ('binomial_pmf', binomial)]:
        np.savetxt(OUTPUT_DIR / f'{name}.csv', np.column_stack((np.arange(len(pmf)), pmf)), delimiter=',', header='bit_errors,probability', comments='')
    plot_block_comparison(block, binomial, probability, OUTPUT_DIR / 'pmf.png')
    exact_rate = float(np.dot(np.arange(len(experiment)), experiment)) / 2
    print(f'Simulated binomial p: {probability:.8g} ({sampled_errors} bit errors in {SIMULATION_BLOCKS} sampled blocks)')
    print(f'Model mean bit-error rate: {exact_rate:.8g}; binomial p has Monte Carlo sampling error')
    if sampled_errors == 0:
        print('No errors sampled: increase SIMULATION_BLOCKS to resolve the binomial comparison')
    print(f'Normalized taps: {taps.tolist()}')
    print(f'Block PMF sum: {block.sum():.12g}; binomial PMF sum: {binomial.sum():.12g}')
    print(f'Full block support: 0..{len(block)-1}; plot omits only the tail below 1e-14')
    print(f'Results saved to {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
