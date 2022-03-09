from typing import List

import mmh3

from simhash import Simhash

kMul: int = 0xC6A4A7935BD1E995
kMul2: int = 0x9E3779B97F4A7835

kMappingTable: List[int] = [0, 1, -1, 0]


def shift_mix(val):
    return val ^ (val >> 47)


def get_more_bits(hash1, hash2):

    hash1 = shift_mix(hash1) * kMul
    hash2 ^= hash1
    newhigh = shift_mix(hash1)
    newlow = shift_mix(hash2 * kMul2) * kMul2

    return newlow, newhigh


def murmurhash(token: str, feature_size: int = 512):

    hash_low = 0
    hash_high = 0
    hash_codes = []

    for i in range(0, feature_size, 64):
        if i == 0:
            hash_low, hash_high = mmh3.hash64(token, signed=False)
        else:
            hash_low, hash_high = get_more_bits(hash_low, hash_high)
        hash_codes.append(hash_low)
        hash_codes.append(hash_high)

    projection: List[int] = []
    for code in hash_codes:
        while code:
            if len(projection) >= feature_size // 2:
                break
            projection.append(kMappingTable[code & 3])
            code = code >> 2
        if len(projection) >= feature_size // 2:
            break
    return projection[: feature_size // 2]

def simhash_token(token, f=64):
    hsh = Simhash(token, f=f).value
    result = [r for r in bin(hsh)[2:].zfill(f)]
    #result = np.array([r for r in str(ternary(hsh)).zfill(41)])
    #result = result.replace('2', '-1')
    return result

class SimhashToken:
    
    def __init__(self, f):
        self.f = f
        self.mapping = {
            '0': '0', # '-1'
            '1': '1',
        }
        
    def __call__(self, token):
        hsh = Simhash(token, f=self.f).value
        result = [self.mapping[r] for r in bin(hsh)[2:].zfill(self.f)]
        return result
