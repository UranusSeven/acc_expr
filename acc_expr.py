import torch
import numpy as np
import tensor_init


def init_tensors(array_a: list[int], array_b: list[int], m, n, k, transposed=False):
    # padd to k elements
    array_a = array_a + [0] * (k - len(array_a))
    array_b = array_b + [0] * (k - len(array_b))

    # to np array
    array_a = np.array(array_a, dtype=np.uint8)
    array_b = np.array(array_b, dtype=np.uint8)

    # to tensor
    array_a = torch.from_numpy(array_a).cuda()
    array_b = torch.from_numpy(array_b).cuda()
    array_a_2d = array_a.view(1, -1)
    array_b_2d = array_b.view(1, -1)
    a_uint8 = array_a_2d.repeat(m, 1)
    b_uint8 = array_b_2d.repeat(n, 1)

    # convert to fp8
    a_fp8e4m3 = tensor_init.create_fp8_from_uint8(a_uint8)
    b_fp8e4m3 = tensor_init.create_fp8_from_uint8(b_uint8)

    if transposed:
        b_fp8e4m3 = b_fp8e4m3.T

    return a_fp8e4m3, b_fp8e4m3


def test_deep_gemm(array_a: list[int], array_b: list[int], m, k, n, print_ref=False):
    import deep_gemm    

    a, b = init_tensors(array_a, array_b, m, n, k, transposed=False)

    a_scale = torch.ones(m, (k + 127) // 128).cuda()
    b_scale = torch.ones((n + 127) // 128, (k + 127) // 128).cuda()
    out = torch.zeros(m, n, dtype=torch.bfloat16, device='cuda')
    deep_gemm.gemm_fp8_fp8_bf16_nt((a, a_scale), (b, b_scale), out)
    
    print(f"deep_gemm.gemm_fp8_fp8_bf16_nt: {out.shape} {out.dtype}")
    print(out)

    if print_ref:
        a_ref = a.to(torch.float32)
        b_ref = b.to(torch.float32)
        out_ref = torch.mm(a_ref, b_ref.T)
        print(f"Ref: {out_ref.shape} {out_ref.dtype}")
        print(out_ref)


def test_scaled_mm(array_a: list[int], array_b: list[int], m, k, n, print_ref=False):
    a, b = init_tensors(array_a, array_b, m, n, k, transposed=True)
    a_scale = torch.ones(1).cuda()
    b_scale = torch.ones(1).cuda()

    out = torch._scaled_mm(a, b, out_dtype=torch.bfloat16, scale_a=a_scale, scale_b=b_scale)
    print(f"torch._scaled_mm: {out.shape} {out.dtype}")
    print(out)

    out = torch._scaled_mm(a, b, out_dtype=torch.float32, scale_a=a_scale, scale_b=b_scale)
    print(f"torch._scaled_mm: {out.shape} {out.dtype}")
    print(out)
    
    if print_ref:
        a_ref = a.to(torch.float32)
        b_ref = b.to(torch.float32)
        out_ref = torch.mm(a_ref, b_ref)
        print(f"Ref: {out_ref.shape} {out_ref.dtype}")
        print(out_ref)


def binary_to_fp8e4m3(bits: str):
    sign_bit = bits[0]
    exponent_bits = bits[1:5]
    mantissa_bits = bits[5:8]
    
    # Convert sign: 0 -> +1, 1 -> -1
    sign = -1 if sign_bit == 1 else 1
    
    # Convert exponent with bias of 7
    exponent = int(exponent_bits, 2)
    
    # Special cases
    if exponent == 0:  # Zero or denormal
        if int(mantissa_bits, 2) == 0:
            return 0.0 * sign  # Signed zero
        else:
            mantissa_value = int(mantissa_bits, 2) / 2**3
            return sign * (2**(1-7)) * mantissa_value
    elif exponent == 15:  # All 1s
        if int(mantissa_bits, 2) == 0:
            return float('inf') * sign  # Infinity
        else:
            return float('nan')  # NaN
    
    exponent_unbiased = exponent - 7
    mantissa_value = 1.0 + (int(mantissa_bits, 2) / 2**3)
    
    return sign * mantissa_value * (2 ** exponent_unbiased)


def hex_to_bits(hex_value: int) -> list[int]:
    binary_str = bin(hex_value)[2:]
    binary_str = binary_str.zfill(8)
    return binary_str


def debug(array_a: list[int], array_b: list[int]):
    binary_a = [hex_to_bits(x) for x in array_a]
    binary_b = [hex_to_bits(x) for x in array_b]

    floats_a = [binary_to_fp8e4m3(x) for x in binary_a]
    floats_b = [binary_to_fp8e4m3(x) for x in binary_b]

    print(f"Decimal Ref: {sum([a * b for a, b in zip(floats_a, floats_b)])}")


def main(array_a: list[int], array_b: list[int]):
    m, k, n = 1, 128, 64
    test_deep_gemm(array_a, array_b, m, k, n)
    test_scaled_mm(array_a, array_b, m, k, n, print_ref=True)


if __name__ == "__main__":
    array_a_0 = [0x77, 0x77, 0x67, 0x47, 0x26, 0xF]
    array_b_0 = [0x60, 0x48, 0x38, 0x38, 0x38, 0x38]
    main(array_a_0, array_b_0)
    debug(array_a_0, array_b_0)

    print("\n----------------------------------------------------------------\n")

    array_a_1 = [0x77, 0x77, 0x67, 0x47, 0x26, 0xF, 0x4]
    array_b_1 = [0x60, 0x48, 0x38, 0x38, 0x38, 0x38, 0x38]
    main(array_a_1, array_b_1)
    debug(array_a_1, array_b_1)

    print("\n----------------------------------------------------------------\n")

    array_a_2 = [0x77, 0x57]
    array_b_2 = [0x38,0x38]
    main(array_a_2, array_b_2)
    debug(array_a_2, array_b_2)
