#ifndef _FUNC_TEST_H_
#define _FUNC_TEST_H_

#include "../ann.h"

typedef gdouble (*p2d2)[2];

void init_weights (PlaneBin *pb);
p2d2 init_inputs (PlaneBin *pb, gsize *n);
gdouble *init_expects (PlaneBin *pb);

#endif
