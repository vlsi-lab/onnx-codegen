#ifndef ${guard}
#define ${guard}

#include <stdint.h>

% for line in weight_defs:
${line}
% endfor

#endif /* ${guard} */
