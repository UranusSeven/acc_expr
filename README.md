# Acc Expr
Experimental exploration of Tensor Core internal accumulation precision.

## Requirements
- deep-gemm
- cuda-toolkit >= 12.4

## Usage
```bash
CUDA_HOME=/path/to/cuda-12.4/ python acc_expr.py
```

## Results
### Nvidia H100
```
A: [0x77, 0x77, 0x67, 0x47, 0x26, 0xF]
B: [0x60, 0x48, 0x38, 0x38, 0x38, 0x38]
Out (fp8_fp8_bf16): 8704.
Ref(fp32_fp32_fp32): 8703.9980
Decimal Ref: 8703.998046875
```

```
A: [0x77, 0x77, 0x67, 0x47, 0x26, 0xF, 0x4]
B: [0x60, 0x48, 0x38, 0x38, 0x38, 0x38, 0x38]
Out(fp8_fp8_bf16): 8704.
Ref(fp32_fp32_fp32): 8704.0059
Decimal Ref: 8704.005859375
```

```
A:[0x77,0x57]
B:[0x38,0x38]
Out(fp8_fp8_bf16): 255.
Ref(fp32_fp32_fp32): 255.
Decimal Ref: 255.0
```
