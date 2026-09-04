"""Field axiom tests.

The field is exhaustively small (101 elements), so these check the axioms (公理) over the whole field rather than sampling.
"""

import itertools

import pytest

from groth16.field import F101, Field, inv_egcd

F = F101
P = F.p


def test_worked_example_from_notes():
    # 3/2 = 3 * 51 = 153 == 52 (mod 101)
    assert F.inv(2) == 51
    assert F.div(3, 2) == 52


def test_results_are_canonical():
    for a, b in itertools.product(F.elements(), repeat=2):
        for r in (F.add(a, b), F.sub(a, b), F.mul(a, b)):
            assert 0 <= r < P


def test_additive_inverse():
    for a in F.elements():
        assert F.add(a, F.neg(a)) == 0
        assert F.sub(a, a) == 0
        assert F.add(a, P - a) % P == 0


def test_multiplicative_inverse():
    for a in F.nonzero_elements():
        assert F.mul(a, F.inv(a)) == 1


def test_inv_is_a_bijection_on_the_nonzero_elements():
    inverses = [F.inv(a) for a in F.nonzero_elements()]
    assert sorted(inverses) == list(F.nonzero_elements())


def test_inv_of_zero_raises():
    with pytest.raises(ZeroDivisionError):
        F.inv(0)
    with pytest.raises(ZeroDivisionError):
        F.div(1, 0)


def test_identities():
    for a in F.elements():
        assert F.add(a, 0) == a
        assert F.mul(a, 1) == a
        assert F.mul(a, 0) == 0


def test_commutativity():
    for a, b in itertools.product(F.elements(), repeat=2):
        assert F.add(a, b) == F.add(b, a)
        assert F.mul(a, b) == F.mul(b, a)


def test_associativity():
    step = range(0, P, 7)  # full triple product is 10^6 combos; stride instead
    for a, b, c in itertools.product(step, repeat=3):
        assert F.add(F.add(a, b), c) == F.add(a, F.add(b, c))
        assert F.mul(F.mul(a, b), c) == F.mul(a, F.mul(b, c))


def test_distributivity():
    step = range(0, P, 7)
    for a, b, c in itertools.product(step, repeat=3):
        assert F.mul(a, F.add(b, c)) == F.add(F.mul(a, b), F.mul(a, c))


def test_div_inverts_mul():
    for a, b in itertools.product(F.elements(), F.nonzero_elements()):
        assert F.div(F.mul(a, b), b) == a


def test_sub_inverts_add():
    for a, b in itertools.product(F.elements(), repeat=2):
        assert F.sub(F.add(a, b), b) == a


def test_pow():
    for a in F.nonzero_elements():
        assert F.pow(a, 0) == 1
        assert F.pow(a, 1) == a
        assert F.pow(a, 2) == F.mul(a, a)
        assert F.pow(a, P - 1) == 1  # Fermat
        assert F.pow(a, -1) == F.inv(a)


def test_no_zero_divisors():
    for a, b in itertools.product(F.nonzero_elements(), repeat=2):
        assert F.mul(a, b) != 0


def test_fermat_and_egcd_agree():
    for a in F.nonzero_elements():
        assert F.inv(a) == inv_egcd(a, P)


def test_negative_and_oversized_inputs_normalize():
    assert F.add(-1, 0) == 100
    assert F.sub(0, 1) == 100
    assert F.mul(-1, -1) == 1
    assert F.elem(202) == 0
    assert F.inv(-2) == F.inv(99)


def test_modulus_must_be_prime():
    with pytest.raises(ValueError):
        Field(100)
    with pytest.raises(ValueError):
        Field(1)


def test_larger_field_shares_the_code():
    # BN254 scalar field -- the modulus this all has to work at eventually.
    bn254 = Field(
        21888242871839275222246405745257275088548364400416034343698204186575808495617
    )
    for a in (2, 3, 7, bn254.p - 1):
        assert bn254.mul(a, bn254.inv(a)) == 1
        assert bn254.inv(a) == inv_egcd(a, bn254.p)
    assert bn254.div(3, 2) == bn254.mul(3, bn254.inv(2))
