"""Prime field arithmetic F_p.

Everything in Groth16 -- R1CS entries, QAP polynomial coefficients, elliptic
curve coordinates -- lives in a prime field. This module is the bottom layer;
if it is wrong, every layer above it is wrong in ways that are hard to locate.

Integers only. Never floats: rounding destroys associativity, distributivity
and the uniqueness of inverses, which are exactly the properties the rest of
the construction relies on.
"""

from __future__ import annotations


class Field:
    """Arithmetic mod a prime `p`.

    The modulus is a constructor argument so the toy field (p = 101) and a
    real curve's scalar field can share all the code above this layer.
    """

    def __init__(self, p: int):
        if p < 2:
            raise ValueError(f"modulus must be at least 2, got {p}")
        if not _is_prime(p):
            raise ValueError(f"modulus must be prime, got {p}")
        self.p = p

    # --- core operations -------------------------------------------------

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def neg(self, a: int) -> int:
        return (-a) % self.p

    def inv(self, a: int) -> int:
        """Multiplicative inverse: the unique x with a * x == 1 (mod p).

        Zero has no inverse -- raising here rather than returning 0 keeps a
        division by an accidentally-zero denominator from silently producing
        a plausible-looking wrong proof.
        """
        a %= self.p
        if a == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse in F_p")
        # Fermat's little theorem: a^(p-1) == 1, so a^(p-2) == a^-1.
        # Valid because p is prime (enforced in __init__).
        return pow(a, self.p - 2, self.p)

    def div(self, a: int, b: int) -> int:
        """a / b == a * b^-1 (mod p). Division is inverse-then-multiply."""
        return self.mul(a, self.inv(b))

    def pow(self, a: int, e: int) -> int:
        """a^e, with negative exponents meaning inv(a)^|e|."""
        if e < 0:
            return pow(self.inv(a), -e, self.p)
        return pow(a % self.p, e, self.p)

    # --- helpers ---------------------------------------------------------

    def elem(self, a: int) -> int:
        """Normalize any integer into the canonical range [0, p)."""
        return a % self.p

    def elements(self):
        """All field elements, 0 .. p-1."""
        return range(self.p)

    def nonzero_elements(self):
        """The multiplicative group F_p*, i.e. 1 .. p-1."""
        return range(1, self.p)

    def __repr__(self) -> str:
        return f"Field(p={self.p})"


def egcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclid: returns (g, x, y) with a*x + b*y == g == gcd(a, b)."""
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


def inv_egcd(a: int, p: int) -> int:
    """Inverse via extended Euclid.

    Kept alongside Field.inv as the general method: unlike Fermat it does not
    need p to be prime, only gcd(a, p) == 1. Field.inv uses Fermat because it
    is a one-liner; this is here to check that against, and for the day the
    modulus is not prime.
    """
    a %= p
    if a == 0:
        raise ZeroDivisionError("0 has no multiplicative inverse in F_p")
    g, x, _ = egcd(a, p)
    if g != 1:
        raise ZeroDivisionError(f"{a} is not invertible mod {p} (gcd = {g})")
    return x % p


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin -- the toy modulus is small, but a real
    scalar field modulus is 254 bits and trial division is not an option."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for sp in small_primes:
        if n % sp == 0:
            return n == sp
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # These bases are a proven-correct witness set for all n < 3.3 * 10^24.
    for base in small_primes:
        x = pow(base, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# The teaching field from step 0.
F101 = Field(101)
