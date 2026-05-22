# Format Notes - Production Mixed-Rate Allocation

## Current Tensor-Level Format

The current artifact is a standard GGUF file with per-tensor quantization types.

For each tensor:

```text
tensor_i := payload_from(selected_source_i)
```

where `selected_source_i` is one of the GGUF source labels used by a selector
run.

The current Gemma 4 E2B-it knapsack artifact uses:

- `Q2_K`
- `Q3_K_S`
- `Q3_K_M`
- `Q3_K_L`
- `IQ4_XS`
- `Q4_K_M`

The current Gemma 4 E2B-it frontier artifact contains:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `Q2_K` | `397` | `2637615244` |
| `Q3_K_M` | `84` | `233001984` |
| `Q4_K_M` | `56` | `119282688` |
| `IQ4_XS` | `40` | `83140608` |
| `Q3_K_L` | `24` | `21356544` |

This requires no custom runtime path because GGUF already stores tensor-level quantization types.

The earlier public-calibrated Qwen3-1.7B artifact contains:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `158` | `431355904` |
| `Q2_K_L` | `60` | `174391296` |
| `IQ4_XS` | `62` | `236191744` |
| `Q3_K_S` | `16` | `63078400` |
| `Q3_K_M` | `15` | `50724864` |

The released Huihui Qwen3.5 4B abliterated artifact uses `layer_family`
allocation with the `c2_calib_weight_blend_mixed` selector:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `67` | `650262528` |
| `Q3_K_S` | `212` | `785808896` |
| `Q3_K_M` | `19` | `118192128` |
| `Q3_K_L` | `37` | `82221568` |
| `IQ4_XS` | `77` | `320533248` |
| `Q4_K_M` | `14` | `42663936` |

The Ministral 3 8B Instruct primary artifact uses tensor allocation with the
`c2_calib_knapsack_mixed` selector:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `147` | `1370210304` |
| `IQ4_XS` | `67` | `1045954560` |
| `Q2_K_L` | `42` | `411500544` |
| `Q3_K_M` | `14` | `189792256` |
| `Q3_K_S` | `39` | `688455680` |

The compact 3.2 bpw Ministral artifact uses:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `172` | `1733935104` |
| `IQ4_XS` | `41` | `425852928` |
| `Q2_K_L` | `54` | `611057664` |
| `Q3_K_M` | `8` | `62390272` |
| `Q3_K_S` | `34` | `562298880` |

The Granite 4.1 8B Heretic artifact uses `layer_family` allocation with the
`c2_calib_knapsack_mixed` selector:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `110` | `585269248` |
| `Q2_K_S` | `40` | `516259840` |
| `Q2_K` | `56` | `359530496` |
| `Q3_K_S` | `62` | `1035780096` |
| `Q3_K_M` | `46` | `423198720` |
| `IQ4_XS` | `48` | `676839424` |

The artifact carries custom `pmra.*` metadata fields with the method name,
variant, source labels, payload bytes, source mix, and source result hash.
`general.file_type` remains inherited from the metadata donor because current
GGUF metadata has no single enum for mixed tensor allocations.

Historical seed-8 Q3_K_M-budget artifact:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ3_XS` | `160` | `523472896` |
| `Q3_K_M` | `81` | `310444032` |
| `IQ4_XS` | `70` | `231735296` |

## Byte Accounting

Current accounting:

- payload bytes: exact sum of selected tensor payload bytes
- file bytes: GGUF metadata, tensor info, alignment, and payloads
- no sidecar or external index

Gemma knapsack exact file:

```text
payload bytes: 3094397068
file bytes:    3110215968
overhead:        15818900
```

Qwen3.5 abliterated weight-blend exact file:

```text
payload bytes: 1999682304
file bytes:    2010651904
overhead:        10969600
```

Ministral 3 8B Instruct primary exact file:

```text
payload bytes: 3705913344
file bytes:    3713801312
overhead:         7887968
```

Ministral 3 8B Instruct compact exact file:

```text
payload bytes: 3395534848
file bytes:    3403422816
overhead:         7887968
```

Granite 4.1 8B Heretic exact file:

```text
payload bytes: 3596877824
file bytes:    3600448224
overhead:         3570400
```

## Why Page/Block Allocation Is Next

Tensor-level allocation is coarse. It can only promote or demote an entire tensor, so the byte budget is chunky.

The next plausible frontier is page/block allocation:

```text
page_j := format_tag_j + payload_j_from(selected_source_j)
```

This could spend stronger formats on sensitive blocks inside a tensor and keep easier blocks lower-bit.

Risk: metadata overhead. If each block needs a tag and offset table, the overhead can erase the gain unless pages are large enough or tags are packed efficiently.

## Compatibility Constraint

Tensor-level mixed allocation is already GGUF-native. Page/block-level mixed allocation is not necessarily GGUF-native unless encoded as a custom quantization type or represented through tensor splitting. Tensor splitting may increase metadata overhead and runtime scheduling cost.
