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
A: [0X77, 0X77, 0X67, 0X47, 0X26, 0XF]
B: [0X60, 0X48, 0X38, 0X38, 0X38, 0X38]
Out (fp8_fp8_bf16): 8704.
Ref(fp32_fp32_fp32): 8703.9980
Decimal Ref: 8703.998046875
```

```
A: [0X77, 0X77, 0X67, 0X47, 0X26, 0XF, 0X4]
B: [0X60, 0X48, 0X38, 0X38, 0X38, 0X38, 0X38]
Out(fp8_fp8_bf16): 8704.
Ref(fp32_fp32_fp32): 8704.0059
Decimal Ref: 8704.005859375
```
