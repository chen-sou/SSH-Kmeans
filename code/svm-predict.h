#ifndef __SVM_PREDICT_H__
#define __SVM_PREDICT_H__

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"
#include "math.h"

#define Malloc0(type,n) (type *)malloc((n)*sizeof(type))

extern int max_index;/////////////////
// int print_null0(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;
struct predict_result  //
{
	int l;
	double *y;
	double *r_y;///////////真实的标签
	double *p;
	double *r_y_p; // 该类所属真实标签的概率
	struct svm_node **x;
};


static char *line0 = NULL;
static int max_line_len0;


static char* readline0(FILE *input);

void exit_input_error0(int line_num);

void load_test_data(char *argv);

void predict(const char *argv1, const char *argv2);


void exit_with_help0();

int predict111(char *argv,char *argv1,char *argv2);

#endif