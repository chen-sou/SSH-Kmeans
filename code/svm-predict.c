#include "svm-predict.h"

double *u2;

int print_null0(const char *s,...) {return 0;}


double ** svm_confidence; // 每个样本的每个类的预测概率 即第一维表示样本id，第二维表示第i个样本在每一个类上的预测概率
double *distance_score;

struct svm_node *x;
struct svm_node *x_space0;
//struct predict_result kmean_result; //自添加
struct predict_result test_data;

int max_nr_attr = 64;

struct svm_model* model0;
//int predict_probability=0;/////////////////////////
int predict_probability=1;


double SVM_ACC;

static char* readline0(FILE *input)
{
	int len;

	if(fgets(line0,max_line_len0,input) == NULL)
		return NULL;

	while(strrchr(line0,'\n') == NULL)
	{
		max_line_len0 *= 2;
		line0 = (char *) realloc(line0,max_line_len0);
		len = (int) strlen(line0);
		if(fgets(line0+len,max_line_len0-len,input) == NULL)
			break;
	}
	return line0;
}

void exit_input_error0(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void load_test_data(char *argv)
{
	int total = 0;
	int i,ii,jj,elements;
	double target_label;
	char *idx,*val,*label,*endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

	FILE *input;
	input = fopen(argv,"r");
    if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv);
		exit(1);
	}		
	jj=0;
	elements = 0;
	test_data.l = 0;
	max_line_len0 = 1024;
	line0 = (char *)malloc(max_line_len0*sizeof(char));
		
	///////////////////////////////////////////////////////////////////////
	while(readline0(input)!=NULL) //先读一遍文件，获得样本数量
	{
		char *p = strtok(line0," \t"); // label
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++jj;
	}
	rewind(input);

	test_data.l=jj;
	test_data.y = Malloc0(double,test_data.l);
	test_data.r_y = Malloc0(double,test_data.l);
	test_data.p = Malloc0(double,test_data.l);
	test_data.r_y_p = Malloc0(double, test_data.l);
	test_data.x = Malloc0(struct svm_node *,test_data.l);
	x_space0 = Malloc0(struct svm_node,elements);

	for(ii=0;ii<test_data.l;ii++)
	{
	//while(readline(input) != NULL)
	//{
		i = 0;
		inst_max_index = -1;
		if(readline0(input)== NULL) break;
		label = strtok(line0," \t\n");
		if(label == NULL) // empty line
			exit_input_error0(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error0(total+1);

		x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));///////////////////////
		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error0(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error0(total+1);

			++i;
		}
		x[i].index = -1;
		if(inst_max_index > max_index)
			max_index = inst_max_index;

		test_data.x[ii]=x;
		test_data.r_y[ii]=target_label;
		test_data.y[ii]=target_label;
		//printf("  %.0f ",test_data.y[ii]); 
	}
	/*/printf("\n%d\n",max_index); 
	printf("\n Test class num : "); 
	for(i=0;i<model0->nr_class;i++)
	{
	  int t=0;
	  for(jj=0;jj<test_data.l;jj++)
	  {
		  if((int)test_data.y[jj]==model0->label[i])
			  t++;
	  }
	  printf("    %d",t); 
	}*/
	free(line0);
	fclose(input);
}

void predict(const char *argv1,const char *argv2)
{
	int index; // 小类别在model->laberl上对应的id
	int j,ii,correct = 0;
	int total = 0;
    int max=0;
	int svm_type,nr_class;
	double max_p,sub_max_p,error = 0;
	double labelPro; // 测试样本所属的真实标签的概率
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	double *prob_estimates=NULL;
	double predict_label;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
	//x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));///////////////////////////
	
	FILE *output= fopen(argv1,"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv1);
		exit(1);
	}
	if((model0=svm_load_model(argv2))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv2);
		exit(1);
	}	
	
	svm_type=svm_get_svm_type(model0);
	nr_class=svm_get_nr_class(model0);
	
	if(predict_probability)
	{
		if(svm_check_probability_model(model0)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model0)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}
	
	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model0));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model0,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}

	
	for (j = 0; j < nr_class; j++) {
		if (model0->label[j] == 1) {
			index = j;
			break;
		}
	}

	u2 = (double *) malloc(test_data.l*sizeof(double));
	for(ii=0;ii<test_data.l;ii++)
	{
	//while(readline(input) != NULL)
	//{
		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(model0,test_data.x[ii],prob_estimates);
			fprintf(output,"%g",predict_label);
			max_p=0;  /////////////////////
			sub_max_p=0;
			for(j=0;j<nr_class;j++)
			{
				fprintf(output," %g",prob_estimates[j]);
				if(max_p<prob_estimates[j]) 
					{
						max_p=prob_estimates[j];
				        max=j;//找到最大的概率
				    }
				//计算每个样本的u2[i]，即max_p-sub_max_p,最大减次大
				svm_confidence[ii][j] = prob_estimates[j];
			}
			for(j=0;j<nr_class;j++)
			{
				if((sub_max_p<prob_estimates[j])&&(j!=max)) 
					sub_max_p=prob_estimates[j];//找到次大的概率
				//计算每个样本的u2[i]，即max_p-sub_max_p,最大减次大
			}
			/*  这部分是保存样本真实标签对应的概率
			for (j = 0; j < nr_class; j++) {
				// printf("%d %lf\n", model0->label[j], prob_estimates[j]);
				if ((int)test_data.r_y[ii] == model0->label[j]) {
					test_data.r_y_p[ii] = prob_estimates[j]; // 保存测试样本真实标签的概率
					printf("---%.0f %d %lf\n", test_data.r_y[ii], model0->label[j], prob_estimates[j]);
					break;
				}
			}*/
			 
			// 这里使用测试样本在 小类别标签上的 预测概率
			test_data.r_y_p[ii] = prob_estimates[index]; // 保存测试样本在小类别下的概率
			// printf("---%.0f %d %lf\n", test_data.r_y[ii], model0->label[index], prob_estimates[index]);

			test_data.p[ii] = max_p;/////////////////////// 保存预测结果的概率
			u2[ii]=max_p-sub_max_p;
			fprintf(output,"\n");
		}
		else
		{
			//predict_label = svm_predict(model0,test_data.x[ii]);
			//fprintf(output,"%g\n",predict_label);
			predict_label = svm_predict_score(model0, test_data.x[ii], distance_score);
			// test_data.y[ii] = predict_label;
			fprintf(output, "%g\n", predict_label);
			// 方式1：仅仅保存样本到标签的距离，不做处理
			for (j = 0; j < nr_class * (nr_class - 1) / 2; j++) {
				test_data.r_y_p[ii] = distance_score[j]; // 保存测试样本真实标签的概率
				// printf("---%.0f %.0f %lf\n", test_data.r_y[ii], test_data.y[ii], distance_score[j]);
			}
			
			// 方式2：计算样本到标签的距离 并且保证小类别样本到超平面距离为正，否则为负

		}
		test_data.y[ii]=predict_label; // 此处为原始代码，改到else里面为了测试方便

		if(predict_label == test_data.r_y[ii])
			++correct;
		error += (predict_label-test_data.r_y[ii])*(predict_label-test_data.r_y[ii]);
		sump += predict_label;
		sumt += test_data.r_y[ii];
		sumpp += predict_label*predict_label;
		sumtt += test_data.r_y[ii]*test_data.r_y[ii];
		sumpt += predict_label*test_data.r_y[ii];
		++total;
	}

	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
		       ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
		       ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
		       );
	}
	else
		printf("Accuracy = %g%% (%d/%d) (classification)\n",(double)correct/total*100,correct,total);
	if(predict_probability)
		free(prob_estimates);
	SVM_ACC = (double)correct / total;

	fclose(output);
}

void exit_with_help0()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

int predict111(char *argv,char *argv1,char *argv2)
//int main(int argc, char **argv)
{
	FILE *output;

	output = fopen(argv1,"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv1);
		exit(1);
	}

	if((model0=svm_load_model(argv2))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv2);
		exit(1);
	}

	//x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));///////////////////////////
	if(predict_probability)
	{
		if(svm_check_probability_model(model0)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model0)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	//predict(output);
	//svm_free_and_destroy_model(&model0);!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//free(x);////////////////////////////
	//free(line0);
	//fclose(input);
	fclose(output);
	return 0;
}

