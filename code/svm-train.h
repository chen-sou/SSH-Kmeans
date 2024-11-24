#ifndef __SVM_TRAIN_H__
#define __SVM_TRAIN_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include <time.h>
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s);

void exit_with_help();

void exit_input_error(int line_num);

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();


//struct svm_problem sub_prob;   // 子类的数据结构

static char *line = NULL;
static int max_line_len;

void do_cross_validation();

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);

// read in a problem (in svmlight format)

// void read_problem0(const char *filename);
void read_problem(const char *filename,int n);

#endif