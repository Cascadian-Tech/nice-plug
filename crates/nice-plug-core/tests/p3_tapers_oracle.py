"""Independent, 80-digit Decimal reference vectors; never imports production code.
The inputs are exact binary32 values, including subnormal/adjacent endpoints.
Regenerate with Python's standard library; checked-in CSV is consumed by Rust.
"""
from decimal import Decimal, getcontext
from pathlib import Path
import struct

getcontext().prec = 80

def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]

def bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", value))[0]

def from_bits(value: int) -> float:
    return struct.unpack("!f", struct.pack("!I", value))[0]

def decimal(value: float) -> Decimal:
    return Decimal.from_float(value)

def generate() -> str:
    domains = [(20, 400), (40, 500), (100, 2000), (500, 10000),
               (2000, 16000), (0.3, 5), (500, 8000), (0.5, 20)]
    ns = {bits(f32(i / 256)) for i in range(257)}
    ns.update([1, 0x00800000, 0x3f7fffff])
    for n in [0.1, 0.25, 0.5, 0.75, 10 / 11]:
        b = bits(f32(n))
        ns.update([b - 1, b, b + 1])
    rows = ['# kind,lower_input_bits,upper_input_bits,normalized_bits,plain_bits,inverse_bits']
    for kind, low, high in [('log', *p) for p in domains] + [('db', -60, 6)]:
        low, high = f32(low), f32(high)
        l, h = decimal(low), decimal(high)
        for nb in sorted(ns):
            n = decimal(from_bits(nb))
            if kind == 'log':
                # Geometric interpolation, independent of the runtime ln/exp form.
                plain = l ** (1 - n) * h ** n
                if n == 0: plain = l
                if n == 1: plain = h
                rounded = f32(float(plain))
                inv = (decimal(rounded).ln() - l.ln()) / (h.ln() - l.ln())
            else:
                plain = Decimal(0) if n == 0 else Decimal(10) ** ((l + n * (h - l)) / 20)
                rounded = f32(float(plain))
                inv = Decimal(0) if rounded == 0 else (20 * decimal(rounded).log10() - l) / (h - l)
            inv = min(Decimal(1), max(Decimal(0), inv))
            rows.append(f'{kind},{bits(low):08x},{bits(high):08x},{nb:08x},{bits(rounded):08x},{bits(f32(float(inv))):08x}')
    return '\n'.join(rows) + '\n'

if __name__ == '__main__':
    Path(__file__).with_suffix('.csv').write_text(generate())
