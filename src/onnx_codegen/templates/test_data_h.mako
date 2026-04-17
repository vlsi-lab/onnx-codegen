/* Auto-generated test vectors for MCU deployment validation. */
#ifndef ${guard}
#define ${guard}

#include <stdint.h>
#include <stddef.h>

#define ${prefix_upper}_TEST_NUM_CASES ${num_cases}
#define ${prefix_upper}_TEST_NUM_INPUTS ${num_inputs}
#define ${prefix_upper}_TEST_NUM_OUTPUTS ${num_outputs}

% for case in cases:
/* ---- ${case["name"]} ---- */
% for inp in case["inputs"]:
static const ${inp["ctype"]} ${inp["symbol"]}[${inp["numel"]}] = {${inp["values"]}};
% endfor
% for out in case["outputs"]:
static const ${out["ctype"]} ${out["symbol"]}[${out["numel"]}] = {${out["values"]}};
% endfor

% endfor
/* Convenience pointer tables */
% for i in range(num_inputs):
static const void* const ${prefix}_test_inputs_${i}[${num_cases}] = {
% for ci, case in enumerate(cases):
    ${case["inputs"][i]["symbol"]}${"," if ci < num_cases - 1 else ""}
% endfor
};
% endfor

% for i in range(num_outputs):
static const void* const ${prefix}_test_golden_${i}[${num_cases}] = {
% for ci, case in enumerate(cases):
    ${case["outputs"][i]["symbol"]}${"," if ci < num_cases - 1 else ""}
% endfor
};
% endfor

#endif /* ${guard} */
