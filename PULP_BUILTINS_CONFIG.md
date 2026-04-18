# PULP Builtin Configuration

The generated code supports both compiler builtins (`__builtin_pulp_*`) and inline assembly implementations for PULP operations.

## Using Assembly Implementations

To use assembly implementations instead of compiler builtins, define `PULPNN_USE_ASM` before including the generated kernels header:

```c
#define PULPNN_USE_ASM
#include "mymodel_kernels.h"
```

This is useful when:
- The PULP compiler does not have full builtin support
- You need explicit control over assembly generation
- You want to verify assembly code generation

## Default Behavior

By default (without `PULPNN_USE_ASM` defined), the code uses compiler builtins:
- `__builtin_pulp_sdotusp4()` - Signed/Unsigned dot product
- `__builtin_pulp_clipu_r()` - Clipping with rounding
- `__builtin_pulp_fl1()` - Find last set bit
- `__builtin_pulp_maxu4()`, `__builtin_pulp_avg4()` - Vector operations
- And many others...

## Compilation Flags

When using assembly implementations, compile with PULP-specific flags:

```bash
gcc -march=pulpv2 -DPULPNN_USE_ASM -c mymodel.c -o mymodel.o
```

## Assembly Instructions Used

The assembly implementations use PULP-V instruction set extensions:
- `p.bextract` / `p.bextractu` - Bit extraction
- `p.pack4` - Pack 4 values
- `p.max4` / `p.min4` / `p.avg4` - Vector operations
- `p.fl1` / `p.clb` - Bit counting
- `p.dotsp4` / `p.sdotusp4` / `p.dotup4` - Dot products
- `p.mac` - Multiply-accumulate
- `p.sra` / `p.srl` / `p.sll` - Shift operations

## Configuration Macros

The following macros can be used directly in code:

| Macro | Builtin | Assembly |
|-------|---------|----------|
| `PULP_SDOTUSP4(a,b)` | `__builtin_pulp_sdotusp4()` | `asm_sdotusp4()` |
| `PULP_CLIPU_R(x,max)` | `__builtin_pulp_clipu_r()` | `asm_clipu_r()` |
| `PULP_FL1(x)` | `__builtin_pulp_fl1()` | `asm_fl1()` |
| `PULP_MAXU4(a,b)` | `__builtin_pulp_maxu4()` | `asm_maxu4()` |
| `PULP_AVG4(a,b)` | `__builtin_pulp_avg4()` | `asm_avg4()` |

And many more - see the generated `_kernels.c` file for complete list.
