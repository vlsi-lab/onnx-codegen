#ifndef ${guard}
#define ${guard}

#include <stddef.h>
#include <stdint.h>

#define ${prefix_upper}_NUM_INPUTS ${num_inputs}
#define ${prefix_upper}_NUM_OUTPUTS ${num_outputs}
% for line in io_size_macros:
${line}
% endfor
% for line in io_type_macros:
${line}
% endfor

extern const char* ${prefix}_input_names[${num_inputs}];
extern const size_t ${prefix}_input_sizes[${num_inputs}];
extern const char* ${prefix}_output_names[${num_outputs}];
extern const size_t ${prefix}_output_sizes[${num_outputs}];

int ${prefix}_run_graph(const void* inputs[${num_inputs}], void* outputs[${num_outputs}]);
int ${prefix}_infer(const void* inputs[${num_inputs}], void* outputs[${num_outputs}]);

#endif /* ${guard} */
