"""Fast, self-contained tests for the deterministic data pipeline.

No dependency on data/train.csv — uses pure functions plus the synthetic
generators, so it runs anywhere (including CI without the dataset).
"""
import random

import makers
import synth
from solvers import _apply_bit_op, solve_bit


def test_apply_bit_op_known_values():
    assert _apply_bit_op(0b11110111, "ROL", 1) == 0b11101111
    assert _apply_bit_op(0b00000001, "NOT") == 0b11111110
    assert _apply_bit_op(0b10000000, "REV") == 0b00000001
    assert _apply_bit_op(0b00000011, "SHL", 1) == 0b00000110


def test_to_roman():
    assert makers._to_roman(4) == "IV"
    assert makers._to_roman(38) == "XXXVIII"
    assert makers._to_roman(1994) == "MCMXCIV"


def test_detect_type():
    assert makers.detect_type("converted ... numeral system.\n11 -> XI") == "numeral"
    assert makers.detect_type("bit rule\n01010001 -> 11011101\nNow ... 00110100") == "bit"
    assert makers.detect_type("gravitational constant\nt = 1s distance = 5 m") == "gravity"


def test_synth_cipher_roundtrip():
    # Every generated cipher must be fully decodable by the maker (full-coverage trick).
    rng = random.Random(0)
    for _ in range(40):
        prompt, _trace, ans = synth.gen_cipher(rng)
        res = makers.make_cipher_cot(prompt)
        assert res is not None, prompt
        assert res[1] == ans


def test_synth_numeral_roundtrip():
    rng = random.Random(1)
    for _ in range(40):
        prompt, _trace, ans = synth.gen_numeral(rng)
        res = makers.make_numeral_cot(prompt)
        assert res is not None and res[1] == ans


def test_synth_bit_solver_recovers_most():
    # Holistic bit ops should be recoverable by solve_bit on nearly all samples.
    rng = random.Random(2)
    n, solved = 40, 0
    for _ in range(n):
        prompt, _trace, ans = synth.gen_bit(rng)
        if solve_bit(prompt) == ans:
            solved += 1
    assert solved >= int(0.9 * n)
