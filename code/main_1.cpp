#include "svm.h"
#include "math.h"
#include "svm-train.h"
#include "svm-predict.h"
// #include "svm-train.c"
// #include "svm-predict.c"
//#include "exValidity.cpp"
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <vector>
#include <set>
using namespace std;
#define labelnumber 15
double noise_per=0;
bool train_with_noise;//�Ƿ���������
bool * estimate_noise;//Ԥ���Ƿ�Ϊ����
bool *flag_noise;//�Ƿ�Ϊ����
double min_del=0.3;
double del;//Ԥ�����������Ŷ���ֵ
double split_m=0.01;
double weight = 0.7;//Ŀ�꺯����׼ȷ�ʵ�Ȩ��
double weight1 = 0.15,weight2 = 0.15;
double total_sse,old_total_sse;
double train_acc,old_train_acc;
double total_UC,old_total_UC;
double train_f1, test_f1; // ÿ���ص�f1-value, ������
double nmi,RI;
//double AVG_coef;
double test_acc,old_test_acc;
double avg_train_acc;//ѵ�������и������ƽ��׼ȷ��
double avg_cluster_acc;//ѵ�������и����ص�ƽ��׼ȷ��
double avg_cluster_UC;//ѵ�������и����ص�ƽ��UC
double sum_SSE,sum_ACC,sum_UC, sum_FVP; // �������ۺ�ָ��
//int max_train_num=1000;
int maxnum=30;//�ص��������
int initial_train_size;
int old_k;
int *predict_class_num;  //Ԥ�����и��������������
int *test_class_num;  //���Լ��и��������������
int *train_class_num;  //���Լ��и��������������
int*num;//�����Ѵ��а���ÿ�������������
svm_node **sum_update;// = Malloc(struct svm_node *, kpara->k);
svm_node *sum_for_train;// = Malloc(struct svm_node, max_index + 1);

extern struct svm_parameter param;		// set by parse_command_line
extern int cross_validation;
extern struct svm_problem prob;		// set by read_problem
extern struct svm_model *model;
extern int predict_probability;
extern struct svm_problem k_prob;
extern struct predict_result test_data;
extern struct svm_node *x_space0;
extern struct svm_node *x_space;
extern struct svm_model* model0;
extern double ** svm_confidence;
extern double SVM_ACC;
extern double *u2;
extern int class_num;
extern struct svm_problem sub_prob; 

struct kmean_param
{
	int k;// �ص�����
	int *k_c;//ÿ���ص���������
	int *noise;//ÿ���ص���������
	double *y_c;//ÿ���ص����
	double* diameter; //���ֱ��
	struct svm_node **x_c;//ÿ���ص�����
	double * w;  // �������ʱ���Ȩ��
	double *subclass_y; // ��Ϊ����ı�ǩ
	int *use; // �ô��Ƿ�ɾ��
	int *size; // �ôص�����������ѵ����+���Լ�
	int *pos; // positive flag  pos:1 neg:0
};

double **kmeans_confidence;
struct kmean_param k_param;
//struct svm_node *zhixin;
//struct kmean_predict_result *kkk;
double *c;//=new double[max_train_num];//ѵ��������Ԥ���ǩ
double *old_c;//=new double[max_train_num];//ѵ��������Ԥ���ǩ
double *u;//=new double[max_train_num];//ѵ��������Ԥ�����
int * cluster;//=new int[max_train_num];//ѵ�����������Ĵ����
int * old_cluster;
//double * coef_train; //ѵ������������ϵ��
//double * coef_test; //ѵ������������ϵ��
svm_node **Center; //�ɵ�����
double * old_yc;

double *c1, *old_c1;;//=new double[test_data.l];//����������Ԥ���ǩ
double *u1;//=new double[test_data.l];//����������Ԥ�����
double *u3; // kmeans�������ϵ�Ԥ�����
int * class_acc;//��������Ԥ����ȷ��ѵ������������
int * class_total;//�������е�ѵ������������
int * cluster1;//=new int[test_data.l];//�������������������Ĵ����
int * count1;//=new int[kpara->k]; ��������ѵ������������
int * count_acc;//��������Ԥ����ȷ��ѵ������������
int * old_count;//
int * old_count_acc;//
int * cluster_classnum=new int[maxnum]; //�������а������������
double * cluster_UC=new double[maxnum];//��������ѵ��������׼ȷ��
double * cluster_acc=new double[maxnum];//��������ѵ��������׼ȷ��
double * old_cluster_acc=new double[maxnum];
double *d = new double[maxnum];
double *d1 = new double[maxnum];
svm_node **sum;
svm_node *result;//=Malloc(struct svm_node,max_index+1);////////����������ӵĽ��/!!!
bool * select_flg;//�Ƿ�ѡΪ��������
double * sse=new double[maxnum];
double * sse_labeled=new double[maxnum];
double * sse_labeled_acc=new double[maxnum];
//double * coef = new double[maxnum];//����ϵ��(Silhouette Coefficient)
double * history_acc=new double[500]; //���ڱ���ÿ�ε�����׼ȷ�ʽ��
double * history_SSE=new double[500];
double * history_FVP=new double[500];
double class_dis;
double * history_UC=new double[500];
//double SSE_min=0,SSE_max=0;
double * old_sse=new double[maxnum];
int * parent=new int[maxnum];
int * new_split=new int[maxnum]; //�Ƿ��·��ѵĴ� 0-���ǣ�1-�ǣ�2-���´ظ��ǵľɵĴ�
svm_node **tmp_Center = Malloc(struct svm_node *, maxnum); //
double * tmp_yc = new double[maxnum];
int * tmp_cluster;////
int * tmp_c;

// co-training�����ñ���
double kmeans_del;//kmeansԤ�����������Ŷ���ֵ
bool * kmeans_noise_flag;
double first_kmean_predict_testdata0; // ��¼��һ�δκ�����ACC�����浽first_kemans_test_acc��
double weight_kmeans;	//kmeansЭͬѵ��α��ǩ����Ȩ��
double new_weight = 0.3;

// ��ƽ��������ñ���
unordered_map<double, vector<double>> hashMap; // �ӱ�ǩ��ԭ��ǩ�Ķ��ձ�
vector<vector<double>> disVec; // ��¼������������ÿ�����ĵľ���
double **svm_ori_confidence; // ����svm��ԭʼ��ǩ�ϵ�Ԥ����ʣ����������ӣ�
double *svm_pro; // ���ڼ�¼��ֽ��δ�ϲ�ʱ��SVM��AUC
double *svm_pred; // ���ڼ�¼��ֽ��δ�ϲ�ʱ��SVM��Ԥ���ǩ
int oriClassNum; // ������������
int useClusterNum; // ʵ��ʹ�õĴصĴ�С
int trainPosCount, trainNegCount; // ѵ����������������������������
int testPosCount = 0, testNegCount = 0; // ������������������������������
double trainIR; // ѵ�������������������ı���
double testIR; // ���������������������ı���
double trainPosPro, trainNegPro; // ѵ���������������ռȫ�������ı���
double avgSubclass; // �����ƽ����С
double *sub_c; // ѵ��������Ԥ���ǩ(����)
int *sub_cluster; // ѵ�����������Ĵ����(����)
int *sub_count1; // ��������ѵ������������(����)
int proCombination = 0;
int maxSubNegCluster; // ���ĸ���صĴ�С
double afterChangeZ; // ������������Z
extern double Z;
extern double* posSubLabel;
struct sortSample
{
	int id; // ����
	double d; // ����
};
struct aucSample
{
	int rank; // �����λ��
	double label; // ��ǩ
	double p; // ����
};

// �������ļ�
int INDEX = 0;
map<int, vector<double>> resultMap;
/*
	����ָ�꣺F1-score  AUC-ROC
		map[0]:��ʼSVM�ĸ�ָ��
		map[1]:��ֽ��SVM��ָ��
		map[2]:��ලkmeans�ĸ�ָ��
		map[3]:��Ϻ�SVM�ĸ�ָ��
	 ��ָ�꣺map[4]
		��ֽ�󣬶��������������Դص�����
		���������������԰������б�ǩ��������
		��IR
*/

// ����ָ��ģ��
vector<int> tp(maxnum, 0);
vector<int> fp(maxnum, 0);
vector<int> fn(maxnum, 0);
vector<int> tn(maxnum, 0);
int TP = 0; // �����ԣ�Ԥ��Ϊ����ʵ��ҲΪ��
int FP = 0; // �����ԣ�Ԥ��Ϊ����ʵ��Ϊ��
int FN = 0; // �����ԣ�Ԥ���븺��ʵ��Ϊ��
int TN = 0; // �����ԣ�Ԥ��Ϊ����ʵ��ҲΪ��
double precision; // ��ȷ��
double recall; // �ٻ��� TPR ROC���ߵ�����
double FPR; // �پ�����	ROC���ߵĺ���
double TNR; // ����ȣ����ڼ���G-mean
double AUC_ROC; // AUC_ROC(area under the roc curve)
double AUC_PR; // AUC-PR(area under the pr curve)
double tAUC_ROC; // �״�svm�Ľ��
double tAUC_PR; // �״�svm�Ľ��

double tGamma, tC, tweight0, tweight1;

/***************************************   ����ָ�����  ************************************************/
void setAllTPN()
{
	// ��ʼ��
	TP = FP = FN = TN = 0;
	for (int i = 0; i < model->nr_class; i++) {
		TP += tp[i];
		FP += fp[i];
		FN += fn[i];
		TN += tn[i];
	}
}

void setConfusionMatrix()
{
	// ��ʼ��Ϊ0
	for (int i = 0; i < maxnum; i++) {
		tp[i] = 0;
		fp[i] = 0;
		tn[i] = 0;
		fn[i] = 0;
	}
	for (int i = 0; i < model->nr_class; i++) {
		for (int j = 0; j < test_data.l; j++) {
			if (test_data.r_y[j] == (double)model->label[i] && test_data.y[j] == (double)model->label[i])
				tp[i]++;
			if (test_data.y[j] == (double)model->label[i] && test_data.r_y[j] != (double)model->label[i])
				fp[i]++;
			if (test_data.r_y[j] == (double)model->label[i] && test_data.y[j] != (double)model->label[i])
				fn[i]++;
			if (test_data.r_y[j] != (double)model->label[i] && test_data.y[j] != (double)model->label[i])
				tn[i]++;
		}
	}
	/*TP = FP = FN = TN = 0;
	for (int i = 0; i < test_data.l; i++) {
		if (test_data.r_y[i] == test_data.y[i] && test_data.r_y[i] == 1)
			TP++;
		if (test_data.r_y[i] != test_data.y[i] && test_data.r_y[i] == -1)
			FP++;
		if (test_data.r_y[i] != test_data.y[i] && test_data.r_y[i] == 1)
			FN++;
		if (test_data.r_y[i] == test_data.y[i] && test_data.r_y[i] == -1)
			TN++;
	}*/
	//cout << "TP:" << TP << "  " << "FP:" << FP << "  " << "FN:" << FN << endl;
	setAllTPN();
}

void setClusterConfusionMatrix(int flag) // �������������ж�, flag=1:ѵ������ָ��  flag=0:���Լ���ָ��
{
	// ��������
	double getTargetClassFScore(int idx);
	// ��ʼ��Ϊ0
	for (int i = 0; i < maxnum; i++) {
		tp[i] = 0;
		fp[i] = 0;
		tn[i] = 0;
		fn[i] = 0;
	}
	if (flag == 1) { // ����ѵ����
		for (int i = 0; i < model->nr_class; i++) {
			for (int j = 0; j < test_data.l; j++) {
				if (prob.r_y[j] == (double)model->label[i] && c[j] == (double)model->label[i])
					tp[i]++;
				if (c[j] == (double)model->label[i] && prob.r_y[j] != (double)model->label[i])
					fp[i]++;
				if (prob.r_y[j] == (double)model->label[i] && c[j] != (double)model->label[i])
					fn[i]++;
				if (prob.r_y[j] != (double)model->label[i] && c[j] != (double)model->label[i])
					tn[i]++;
			}
		}
	}
	else { // ���ڲ��Լ�
		for (int i = 0; i < model->nr_class; i++) {
			for (int j = 0; j < test_data.l; j++) {
				if (test_data.r_y[j] == (double)model->label[i] && c1[j] == (double)model->label[i])
					tp[i]++;
				if (c1[j] == (double)model->label[i] && test_data.r_y[j] != (double)model->label[i])
					fp[i]++;
				if (test_data.r_y[j] == (double)model->label[i] && c1[j] != (double)model->label[i])
					fn[i]++;
				if (test_data.r_y[j] != (double)model->label[i] && c1[j] != (double)model->label[i])
					tn[i]++;
			}
		}
	}
	//for (int i = 0; i < model->nr_class; i++) {
	//	printf("\n%d  :  tp:%d tn:%d fp:%d fn:%d\n", model->label[i], tp[i], tn[i], fp[i], fn[i]);
	//}
	setAllTPN();
	int id;
	for(int i=0; i<model->nr_class; i++)
		if (model->label[i] == 1) {
			id = i;
			break;
		}
	double tf1 = getTargetClassFScore(id); // ����ֱ�Ӽ������1��f1-value
	if (flag == 1) train_f1 = tf1;
	else test_f1 = tf1;
}

double getPrecision() // ��ȷ��
{
	double p  = (double)TP / (TP + FP);
	return p;
}

double getTargetClssPrecision(int& idx)
{
	return (double)tp[idx] / (tp[idx] + fp[idx]);
}

double getRecall() // �ٻ���
{
	double r = (double)TP / (TP + FN);
	return r;
}

double getTargetClsaaRecall(int& idx)
{
	return (double)tp[idx] / (tp[idx] + fn[idx]);
}

double getFPR() // �پ�����
{
	double fpr = (double)FP / (FP + TN);
	return fpr;
}

double getTNR()
{
	double tnr = (double)TN / (TN + FP);
	return tnr;
}

double getFScore() // f-score
{
	double x = 2 * precision * recall;
	double y = precision + recall;
	return x / y;
}

double getTargetClassFScore(int idx)
{
	double x = 2 * getTargetClssPrecision(idx) * getTargetClsaaRecall(idx);
	double y = getTargetClssPrecision(idx) + getTargetClsaaRecall(idx);
	return x / y;
}

double getGmean() // G-mean
{
	TNR = getTNR();
	return sqrt(recall * TNR);
}

bool cmp2(aucSample& x, aucSample& y) // ����getAUC_ROC() -- ��������
{
	if (x.p != y.p) return x.p < y.p;
	return x.rank < y.rank;
}

bool cmp3(aucSample& x, aucSample& y) // ����getAUC_PR() -- �ݼ�����
{
	if (x.p != y.p) return x.p > y.p;
	return x.rank < y.rank;
}

double selectMaxP(int start, vector<double>& allPrecision)
{
	if (start == test_data.l - 1)
		return allPrecision[start];
	double max = allPrecision[start];
	for (int i = start + 1; i < test_data.l; i++) {
		if (max < allPrecision[i])
			max = allPrecision[i];
	}
	return max;
}

double getAUC_ROC(int flag=0) // ΢Ԫ������ROC������� -- AUC-ROC
{
	int posCount = 0, negCount = 0; // �������������� -- ���ﶨ��С�������Ϊ������������������Ϊ������
	double posRankSum = 0.0; // ��������rank��
	double posSum; // ��Ӧ��ʽ�� -- M(M+1)/2
	double PNprocuct; // ��Ӧ��ʽ�ķ�ĸ -- �������������ĳ˻�
	double posLabel = 1; // ��¼��������ǩ

	// flag=1:������ֽ��δ�ϲ���SVM�����Ĭ��flag=0
	if (flag == 1) {
		for (int i = 0; i < test_data.l; i++) {
			test_data.r_y_p[i] = svm_pro[i];
		}
	}

	vector<aucSample> aucTable(test_data.l);
	for (int i = 0; i < test_data.l; i++) {
		aucTable[i].label = test_data.r_y[i]; // ������������ʵ��ǩ
		aucTable[i].p = test_data.r_y_p[i]; // ��ʵ��ǩ��Ӧ��Ԥ�����
		aucTable[i].rank = i;
		if (test_data.r_y[i] == posLabel) posCount++;
		else negCount++;
	}
	// cout << posCount << " " << negCount << endl;
	if (posCount > negCount) posLabel = 1;
	//posCount = posCount < negCount ? posCount : negCount; // С�������Ϊ������
	sort(aucTable.begin(), aucTable.end(), cmp2); // ���ݸ��ʴ�С��������
	for (int i = 0; i < test_data.l; i++) // ���Ķ�Ӧ��rank
		aucTable[i].rank = i + 1;
	//for (int i = 0; i < test_data.l; i++) {
	//	printf("%.0f %lf %d\n", aucTable[i].label, aucTable[i].p, aucTable[i].rank);
	//}
	// cout << "�������ı�ǩ�ǣ�" << posLabel << endl;
	// ������������rank��
	for (int i = 0; i < test_data.l; i++) { // ����ȵĸ��ʵ����
		if (i == 0) {
			if (aucTable[i].label == posLabel && aucTable[i + 1].p != aucTable[i].p)
				posRankSum += aucTable[i].rank;
		}
		else if (i != test_data.l - 1){
			if (aucTable[i].label == posLabel && aucTable[i + 1].p != aucTable[i].p && aucTable[i - 1].p != aucTable[i].p)
				posRankSum += aucTable[i].rank;
		}
		else {
			if (aucTable[i].label == posLabel && aucTable[i - 1].p != aucTable[i].p)
				posRankSum += aucTable[i].rank;
		}
	}
	// 2.����ȸ��ʵ�rankֵ
	int sameNum, start;
	double tRankSum;
	for (int i = 0; i < test_data.l; i++) {
		tRankSum = 0;
		sameNum = 0;
		start = i;
		if (aucTable[i].label == posLabel) {
			if (i != 0) {
				for (int j = i - 1; j >= 0; j--) {
					if (aucTable[j].p == aucTable[i].p) {
						tRankSum += aucTable[j].rank;
						sameNum++;
					}
					else
						break;
				}
			}
			if (i != test_data.l - 1) {
				for (int j = i + 1; j < test_data.l; j++) {
					if (aucTable[j].p == aucTable[i].p) {
						tRankSum += aucTable[j].rank;
						sameNum++;
						start = j;
					}	
					else
						break;
				}
			}
		}
		if (tRankSum != 0) {
			tRankSum += aucTable[i].rank;
			tRankSum /= (sameNum + 1);
		}
		posRankSum += tRankSum;
		//i = start;
	}
	posSum = posCount * (posCount + 1) * 1.0 / 2;
	PNprocuct = posCount * negCount;
	AUC_ROC = (posRankSum * 1.0 - posSum) / PNprocuct;
	// printf("AUC-ROC:%lf\n", AUC_ROC);
	aucTable.clear();

	return AUC_ROC;
}
     
double getAUC_PR() // AUC-PR����AP(average precision)
{
	int posCount = 0; // ������������
	double posLabel = 1; // ���ﶨ���������ı�ǩ
	vector<aucSample> apTable(test_data.l);
	for (int i = 0; i < test_data.l; i++) {
		apTable[i].rank = i + 1;
		apTable[i].p = test_data.r_y_p[i];
		apTable[i].label = test_data.r_y[i];
		if (test_data.r_y[i] == posLabel) posCount++;
	}
	sort(apTable.begin(), apTable.end(), cmp3); // �����ʵݼ�����
	
	int sampCount = 0, posNum = 0;
	double tempPrecision;
	vector<double> allPrecision;
	vector<double> maxPrecision; // ���ڴ洢ÿ��recall��Ӧ������precision
	vector<int> boundary; // ��¼ÿ�����������ֵ�λ��
	for (int i = 0; i < test_data.l; i++) {
		if (apTable[i].label == posLabel) {
			boundary.push_back(i);
		}
	}
	for (int i = 0; i < test_data.l; i++) {
		sampCount++;
		if (apTable[i].label == posLabel)
			posNum++;
		tempPrecision = posNum * 1.0 / sampCount;
		allPrecision.push_back(tempPrecision);
	}
	for (int i = 0; i < boundary.size(); i++) {
		tempPrecision = selectMaxP(boundary[i], allPrecision);
		maxPrecision.push_back(tempPrecision);
	}

	/*int id = 0, sampCount = 0, end;
	int tpNum = 0; // Ŀǰ����������Ŀ
	double tempPrecision, tempRecall;
	vector<double> maxPrecision; // ���ڴ洢ÿ��recall��Ӧ������precision
	for (int i = 0; i < test_data.l;) {
		vector<double> precArr; // ����ÿһ�׶ε�precision
		if (id + 1 < posCount) end = boundary[id + 1];
		else end = test_data.l;
		for (int j = i; j < end; j++) {
			sampCount++;
			if (apTable[j].label == posLabel) tpNum++;
			tempPrecision = tpNum * 1.0 / sampCount;
			tempRecall = tpNum * 1.0 / posCount;
			precArr.push_back(tempPrecision);
		}
		//i = j;
		id++;
		if(id < posCount) i = boundary[id];
		tempPrecision = precArr[0];
		for (int i = 1; i < precArr.size(); i++) {
			if (tempPrecision < precArr[i])
				tempPrecision = precArr[i];
		}
		maxPrecision.push_back(tempPrecision);
		if (id >= posCount) break;
	}*/
	//for (double num : maxPrecision)
	//	cout << num << " ";
	//cout << endl;

	double sum = 0.0;
	for (int i = 0; i < maxPrecision.size(); i++)
		sum += maxPrecision[i];
	AUC_PR = sum / maxPrecision.size();
	// printf("AUC-PR:%lf\n", AUC_PR);

	return AUC_PR;
}

// libsvm�ٷ����㷽��
// for auc and ap
class Comp {
	const double *dec_val;
public:
	Comp(const double *ptr) : dec_val(ptr) {}
	bool operator()(int i, int j) const {
		return dec_val[i] > dec_val[j];
	}
};

double auc(int flag=0)
{
	// flag=1:������ֽ��δ�ϲ���SVM�����Ĭ��flag=0
	if (flag == 1) {
		for (int i = 0; i < test_data.l; i++) {
			test_data.r_y_p[i] = svm_pro[i];
		}
	}

	vector<double> dec_values;
	vector<int> ty;
	for (int i = 0; i < test_data.l; i++) {
		ty.push_back((int)test_data.r_y[i]);
		dec_values.push_back(test_data.r_y_p[i]);
	}

	double roc = 0;
	size_t size = dec_values.size();
	size_t i;
	std::vector<size_t> indices(size);

	for (i = 0; i < size; ++i) indices[i] = i;

	std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

	int tp = 0, fp = 0;
	for (i = 0; i < size; i++) {
		if (ty[indices[i]] == 1) tp++;
		else if (ty[indices[i]] == -1) {
			roc += tp;
			fp++;
		}
	}

	if (tp == 0 || fp == 0)
	{
		fprintf(stderr, "warning: Too few postive true labels or negative true labels\n");
		roc = 0;
	}
	else
		roc = roc / tp / fp;

	printf("AUC = %g\n", roc);

	return roc;
}

double ap() 
{
	vector<double> dec_values;
	vector<int> ty;
	for (int i = 0; i < test_data.l; i++) {
		ty.push_back((int)test_data.r_y[i]);
		dec_values.push_back(test_data.r_y_p[i]);
	}

	size_t size = dec_values.size();
	size_t i;
	std::vector<size_t> indices(size);

	for (i = 0; i < size; ++i) indices[i] = i;
	std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

	int p = 0, tp = 0;
	double prev_recall = 0, area = 0;

	for (i = 0; i < size; ++i) p += (ty[i] == 1);

	if (p == 0) {
		fprintf(stderr, "warning: Too few postive labels\n");
		return 0;
	}

	for (i = 0; i < size; ++i) {
		tp += (ty[indices[i]] == 1);

		if (i + 1 < size && dec_values[indices[i]] == dec_values[indices[i + 1]])
			continue;

		double recall = (double)tp / p;
		double precision = (double)tp / (double)(i + 1);

		area += precision * (recall - prev_recall);
		prev_recall = recall;
	}

	printf("AP = %g\n", area);
	return area;
}

double sskmeans_auc(double* u1)
{
	vector<double> dec_values;
	vector<int> ty;
	for (int i = 0; i < test_data.l; i++) {
		ty.push_back((int)test_data.r_y[i]);
		dec_values.push_back(u1[i]);
	}

	double roc = 0;
	size_t size = dec_values.size();
	size_t i;
	std::vector<size_t> indices(size);

	for (i = 0; i < size; ++i) indices[i] = i;

	std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

	int tp = 0, fp = 0;
	for (i = 0; i < size; i++) {
		if (ty[indices[i]] == 1) tp++;
		else if (ty[indices[i]] == -1) {
			roc += tp;
			fp++;
		}
	}

	if (tp == 0 || fp == 0)
	{
		fprintf(stderr, "warning: Too few postive true labels or negative true labels\n");
		roc = 0;
	}
	else
		roc = roc / tp / fp;

	printf("AUC = %g\n", roc);

	return roc;
}

/**********************************   ��������۱�׼   **********************************************************/
void getPerformance(int id, int flag=0)
{
	//AUC_ROC = getAUC_ROC();
	//AUC_PR = getAUC_PR();
	// double tAUC_ROC = AUC_ROC;
	// double tAUC_PR = AUC_PR;
	printf("AUC-ROC: %lf       AUC-PR:%lf\n", AUC_ROC, AUC_PR);

	// flag=1:������ֽ��δ�ϲ���SVM�����Ĭ��flag=0
	if (flag == 1) {
		for (int i = 0; i < test_data.l; i++) {
			test_data.y[i] = svm_pred[i];
		}
	}

	setConfusionMatrix();
	/*precision = getPrecision();
	recall = getRecall();
	double f_score = getFScore();
	double Gmean = getGmean();
	cout << "FP:" << FP << "  " << "TP:" << TP << " " << "FN:" << FN << "  " << "TN:" << TN << endl;
	cout << "p:" << precision << "  " << "r:" << recall << "  " << "f:" << f_score << " " << "G-mean:" << Gmean << endl;
	resultMap[INDEX].push_back(f_score);*/
	precision = recall = 0;
	double f_score = 0;
	double Gmean = 0;
	for (int i = 0; i < model->nr_class; i++) {
		precision += getTargetClssPrecision(i);
		recall += getTargetClsaaRecall(i);
		f_score += getTargetClassFScore(i);
	}
	cout << "p:" << precision / model->nr_class << "  " << "r:" << recall / model->nr_class << "  " << "f:" << f_score / model->nr_class << " " << endl;

	for (int i = 0; i < model->nr_class; i++) {
		double tP = getTargetClssPrecision(i);
		double tR = getTargetClsaaRecall(i);
		double tFS = getTargetClassFScore(i);
		//cout << "���:" << model->label[i] << " FP:" << fp[i] << "  " << " TP:" << tp[i] << " " << " FN:" << fn[i] << "  " << " TN:" << tn[i] << endl;
		printf("���%d��precision��%.4f, recall:%.4f, fscore:%f\n", model->label[i], tP, tR, tFS);
		if (model->label[i] == 1)
			resultMap[id].push_back(tFS); // ��¼F1-score
	}

	// ����ÿ����Ĳ�ȫ��
	set<int> hashSet;
	for (int i = 0; i < model->nr_class; i++) {
		hashSet.insert(model->label[i]);
	}
	model->nr_class = hashSet.size();
	int acc = 0, classNum;
	for (int i = 0; i < model->nr_class; i++) {
		acc = 0;
		classNum = 0;
		for (int j = 0; j < test_data.l; j++) {
			if (test_data.r_y[j] == model->label[i]) classNum++;
			if (test_data.r_y[j] == model->label[i] && test_data.y[j] == test_data.r_y[j]) acc++;
		}
		printf("��ǩΪ %d �����׼ȷ��(��ȷԤ�����������%d / �ܵ�������%d)Ϊ:%f\n", model->label[i], acc, classNum, acc * 1.0 / classNum);
		//if (model->label[i] == 1)
		//	resultMap[INDEX].push_back(acc * 1.0 / classNum);
	}
}

/********************************************  end  *************************************************************/

/********************************************* util(������)  ***************************************************************/
long long C(long long n,long long m)//����ֲ�
{
	long long ans=1;
	for(int i=1;i<=m;i++)
	{
		ans=ans*(n-m+i)/i;
	}
	return ans;
}
void copy(svm_node *x,svm_node *y)//ʵ��X->Y����������
{
		int t=0;
		while(x[t].index!=-1)
		{
			y[t].index=x[t].index;
			y[t].value=x[t].value;//����ƽ��ֵ
			t++;
		}
		y[t].index=-1;	 
}

double distance1(const svm_node *px, const svm_node *py)
{
	double ssum = 0;
	int i=0,j=0;
	while(px[i].index != -1 || py[j].index != -1)
	{
		if(px[i].index == py[j].index)
		{
			ssum += (px[i].value - py[j].value)*(px[i].value - py[j].value);	
			++i;	++j;
		}
		else if(px[i].index > py[j].index)
		{
			if( py[j].index!=-1)
		    {
              ssum +=  py[j].value * py[j].value;	
			  ++j;
			}
			else
		    {
		       ssum += px[i].value * px[i].value;	
			   ++i;
		    }
		}
		else
		{
		   if( px[i].index!=-1)
		   {
			   ssum += px[i].value * px[i].value;	
			   ++i;
		   }
		   else
		   {
              ssum +=  py[j].value * py[j].value;	
			  ++j;
		    }
		}
	}
	return sqrt(ssum);
}

svm_node * add(const svm_node *x,const svm_node *y)	//ʵ���������������
{
	int i=0,j=0,t=0;
	result[0].index = -1;
	result[0].value = 0;
	while(x[i].index!=-1 || y[j].index!=-1)
	{
	  if(x[i].index==y[j].index)
	  {
		  result[t].index=x[i].index;
		  result[t].value=x[i].value+y[j].value;
		  i++;j++;t++;
	  }
	  else if(x[i].index<y[j].index) 
	  {
		if(x[i].index!=-1)
		{
		  result[t].index=x[i].index;
		  result[t].value=x[i].value;
		  i++;t++;
		}
		else
        {
		  result[t].index=y[j].index;
		  result[t].value=y[j].value;
		  j++;t++;
		}
	  }
	  else
	  {
		if(y[j].index!=-1)
		{
		  result[t].index=y[j].index;
		  result[t].value=y[j].value;
		  j++;t++;
		}
		else
		{
		  result[t].index=x[i].index;
		  result[t].value=x[i].value;
		  i++;t++;
		}
	  }
	}
	result[t].index=-1;
	return result;
}

svm_node * diminish(const svm_node *x, const svm_node *y)	//ʵ���������������
{
	int i = 0, j = 0, t = 0;
	result[0].index = -1;
	result[0].value = 0;
	while (x[i].index != -1 || y[j].index != -1)
	{
		if (x[i].index == y[j].index)
		{
			result[t].index = x[i].index;
			result[t].value = x[i].value - y[j].value;
			i++; j++; t++;
		}
		else if (x[i].index < y[j].index)
		{
			if (x[i].index != -1)
			{
				result[t].index = x[i].index;
				result[t].value = x[i].value;
				i++; t++;
			}
			else
			{
				result[t].index = y[j].index;
				result[t].value = y[j].value;
				j++; t++;
			}
		}
		else
		{
			if (y[j].index != -1)
			{
				result[t].index = y[j].index;
				result[t].value = y[j].value;
				j++; t++;
			}
			else
			{
				result[t].index = x[i].index;
				result[t].value = x[i].value;
				i++; t++;
			}
		}
	}
	result[t].index = -1;
	return result;
}

void multiply(double factor, svm_node *x) // ʵ�� double * svm_node 
{
	int t = 0;
	while (x[t].index != -1)
	{
		x[t].value = factor * x[t].value;//����ƽ��ֵ
		t++;
	}
}

double getOriLabel(double label) // ����ԭʼ���ǩ
{
	double oriLabel = 1;
	bool flag = false;
	for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
		for (int k = 0; k < it->second.size(); k++) {
			if (it->second[k] == label) {
				// cout << final_result[i] << " " << it->first << " " << test_data.r_y[i] << endl;
				oriLabel = it->first;
				flag = true;
				break;
			}
		}
		if (flag) break;
	}
	return oriLabel;
}

void mergeSubPro() // ���ԭ��ǩ��Ԥ�����
{
	//svm_ori_confidence.clear(); // ���ԭ����
	//vector<vector<double>> svm_ori_confidence(model->nr_class);
	/*
		svm_ori_confidence[i][0]: ����ǩΪ����ĸ���
		svm_ori_confidence[i][1]������ǩΪ����ĸ���
	*/
	for (int i = 0; i < test_data.l; i++) {
		svm_ori_confidence[i][0] = svm_ori_confidence[i][1] = 0;
		for (int j = 0; j < model->nr_class; j++) {
			if (getOriLabel(model->label[j]) == 1) {
				svm_ori_confidence[i][0] += svm_confidence[i][j];
			}
			if (getOriLabel(model->label[j]) == -1) {
				svm_ori_confidence[i][1] += svm_confidence[i][j];
			}
		}
	}
}

/************************************  util end   ***********************************************************************************/

bool kmean_select_traindata(int num, char *filename) //num��ѵ�����Ĵ�С,�����ݼ���ѡ�񲿷���Ϊѵ����
{
	int i,ii,rnd;
	srand((unsigned)time(NULL)); //Ϊ������ʹ��rand���������
	FILE *fp1 = fopen("mytrain","w");//ѵ����д��haberman3
	FILE *fp2 = fopen("mytest","w");//���Լ�д��haberman4
	read_problem(filename); 
	bool * s_flg = new bool[prob.l];//
	for (i = 0; i < prob.l; i++)  s_flg[i] = false;
	for (i = 0; i < num; )//num��ѵ�����Ĵ�С
	{
		rnd = (int)(prob.l*rand() / (RAND_MAX + 1.0)) % prob.l;//rnd��1��prob.l�������
		if (s_flg[rnd] == false)
		{
			s_flg[rnd] = true;//��ѡ�������ѵ�����������ϱ�ǩ
			i++;
		}
	}
	//printf("  %d",prob.l);  
	for (i = 0; i < prob.l; i++)//l�����ݼ���С�������ݼ��У�flag=true���������Ƶ�ѵ����������Ͱ��������Ƶ����Լ�
	{
		if (s_flg[i] == true)//ѵ����
		{
			fprintf(fp1,"%.0f",prob.y[i]);//������ǩ  
			ii=0;
            while( prob.x[i][ii].index!=-1)
			{
				fprintf(fp1, " "); 
				fprintf(fp1,"%d",(prob.x[i][ii].index));//����ֵ���	
			    fprintf(fp1,":");
			    fprintf(fp1,"%.4f",prob.x[i][ii].value);	//����ֵ				
			    ii++;
			}
			fprintf(fp1, "\n");
		}
		else//���Լ�
		{
			fprintf(fp2,"%.0f",prob.y[i]);	
			ii = 0;
			while( prob.x[i][ii].index!=-1)
			{
				fprintf(fp2," ");
			    fprintf(fp2,"%d",(prob.x[i][ii].index));//����ֵ���	
			    fprintf(fp2,":");
			    fprintf(fp2,"%.4f",(prob.x[i][ii].value));	//����ֵ
				ii++;
			}
			fprintf(fp2,"\n");	
		}
	}
    fclose(fp1);
	fclose(fp2);
	/*/�ж�ѵ�������Ƿ�����������������
	double num1[100];//��ű�ǩ�����ݼ����ж����ֱ�ǩ
	int k = 1;
	num1[0] = prob.y[0];
	for (int j = 1; j<prob.l; j++)
	{
	int flag = 0;
	for (int i = 0; i<k; i++)
	if (prob.y[j] != num1[i])
	{
	num1[k] = prob.y[j];
	}
	else
	flag++;
	if (flag == 0)
	{
	num1[k] = prob.y[j];
	k++;
	}
	}

	int m = 0;
	double num2[100];
	for (int k = 0; k<sizeof(num1) / sizeof(num1[0]); k++)//��ʼ��num2����,sizeof(num1)/sizeof(num1[0])��num1�ĳ��ȣ�Ҳ���Ǳ�ǩ�ĸ���
	num2[k] = 0;
	for (m = 0; m<prob.l; m++)//������������
	{
	if (select_flg[m] == true)//ɸѡ������ǵ�����
	for (int k = 0; k<sizeof(num1) / sizeof(num1[0]); k++)//���ҵ�ǰ���������ĸ�num1
	if (num1[k] == prob.y[m])
	num2[k]++;//ÿ������������
	}
	for (int k = 0; k<sizeof(num1) / sizeof(num1[0]); k++)//���ҵ�ǰ���и���Ϊ0������
	if (num2[k] == 0)//������������Ϊ0
	for (int i = 0; i<prob.l; i++)
	if (prob.y[i] == num1[k])
	{
	select_flg[i] = true;
	break;
	}
	*/
	return 0;
}
void del_noise(svm_problem * kprob,int k)//��������
{
	int i,j,t,dif,Acc=0,e_num=0;
	int index1,index2,index3;
	double min1,min2,min3,dis=0;
	estimate_noise = Malloc(bool,kprob->l);
	for(i=0;i<kprob->l;i++)//�ҵ��������������ѵ���������ж�����Ƿ�һ��
	{
		estimate_noise[i]=false;
		dif=0;//���һ�µ��ڽ���������
		min1=100000;min2=100001;min3=100002;
		index3=index2=index1=-1;
		for(j=i+1;j<kprob->l;j++)
		{
			dis=distance1(kprob->x[i], kprob->x[j]);;//
			if(dis<=min1)
			{min3=min2;min2=min1;min1=dis;index3=index2;index2=index1;index1=j;}
			else if(dis<=min2)
			{min3=min2;min2=dis;index3=index2;index2=j;}
			else if(dis<=min3)
			{min3=dis;index3=j;}
		}
		if(kprob->y[i]!=kprob->y[index1]) dif++;
		if(kprob->y[i]!=kprob->y[index2]) dif++;
		if(kprob->y[i]!=kprob->y[index3]) dif++;
		printf("%f  %f  %f  ",min1,min2,min3);
		if(dif>=2) {
			estimate_noise[i] =true;
			e_num++;
		    if(estimate_noise[i]==flag_noise[i]) Acc++;
		}
	}
	/*///////////////////////////////////////////////////////////////////

	dis=0;
	k_param.k=k;	
	svm_node *sum0=Malloc(struct svm_node,max_index+1);
	class_acc = new int[k];
	class_total = new int[k];
	total_sse=0; train_acc=1; total_UC=1;
	for(i=0;i<k;i++) 
	{ 
		k_param.k_c[i]=0; 
		k_param.y_c[i]=model->label[i]; 
		count[i]=0; count_acc[i]=0;
		sse[i]=0;
		k_param.noise[i]=0;
		k_param.diameter[i]=0;
		train_class_num[i]=0;
	}
	for(i=0;i<k_param.k;i++)//����ÿ���أ�����
	{
		sum0->index=-1;///
		sum0->value=0;
		for(j=0;j<kprob->l;j++)
		{
			if(kprob->y[j]==k_param.y_c[i] && estimate_noise[j]==false)//������������������
			{
				add(sum0,kprob->x[j]);
				copy(result,sum0);
				k_param.k_c[i]++;
				count[i]++;
				cluster[j]=i;
				c[j]=kprob->y[j];				
			    for(t=j+1;t<kprob->l;t++)/////////////Ѱ�Ҵ�i�����ֱ��
			    {
				  if(kprob->y[t]==k_param.y_c[i] && k_param.diameter[i]<distance(kprob->x[j],kprob->x[t]))//
                     k_param.diameter[i]=distance(kprob->x[j],kprob->x[t]);	
				} ///////
			}
		}
		t=0;
		while(sum0[t].index!=-1)
		{
			k_param.x_c[i][t].index=sum0[t].index;
			k_param.x_c[i][t].value=sum0[t].value/k_param.k_c[i];//����ƽ��ֵ
			t++;
		}
		k_param.x_c[i][t].index=-1;	  
	}

	printf("\n train class num :");
	for (i = 0; i<k_param.k; i++)//����ÿ���أ�����
	{
		printf("   %d", k_param.k_c[i]);
		for(j=0;j<kprob->l;j++)
		{
		   if(kprob->y[j]==k_param.y_c[i]) 
              sse[i]+=distance(kprob->x[j], k_param.x_c[i])*distance(kprob->x[j], k_param.x_c[i]);	
		}
		sse[i]=sse[i]/count[i];
		count_acc[i] = count[i];
		cluster_acc[i]=1;
		train_class_num[i]= count[i];
	}

	printf("\n Test class num : "); //ͳ�Ʋ���������ÿ�������������
	for(i=0;i<k_param.k;i++)
	{
	  test_class_num[i]=0;
	  for(j=0;j<test_data.l;j++)
	  {		  
		  if((int)test_data.r_y[j]==model->label[i])
			  test_class_num[i]++;
	  }
	  printf("    %d",test_class_num[i]); 
	}
	free(sum0);
	sum0=NULL;
	/////////////////////////////
	double min_d;int min_i;
	e_num=Acc=0;
	for(i=0;i<kprob->l;i++)//�ҵ����������������������ĵľ���
	{
		min_d=100000; min_i=-1;
		if(estimate_noise[i]==true){
			for(j=0;j<k_param.k;j++){
				dis=distance(k_param.x_c[j], kprob->x[i]);
				if(dis<min_d){min_d=dis; min_i=j;} 
			}
			if(k_param.y_c[min_i]==kprob->y[i]) 
				estimate_noise[i]=false;
		}
		if(estimate_noise[i]==true ){
			e_num++;
		    if(true==flag_noise[i]) Acc++;
		}
	}
	/*for(i=0;i<kprob->l;i++)//ɾ������������ѵ������
	{
		//
		if(estimate_noise[i]==true){
			for(j=i;j<kprob->l-1;j++)
			{
				kprob->y[j]=kprob->y[j+1];
				kprob->r_y[j]=kprob->r_y[j+1];
				copy(kprob->x[j+1],kprob->x[j]);
			}
		}
		kprob->l--;
	}*/
}

void k_center(svm_problem * kprob, int k)//��ʼ������
{
	int i, j, t;
	double dis = 0;
	k_param.k = k;
	svm_node *sum0 = Malloc(struct svm_node, max_index + 1);
	//class_acc = new int[k];
	class_total = new int[k];
	total_sse = 0; train_acc = 1; total_UC = 1;
	for (i = 0; i < k; i++)
	{
		k_param.k_c[i] = 0;
		k_param.y_c[i] = model->label[i];
		count1[i] = 0; count_acc[i] = 0;
		sse[i] = 0;
		//k_param.noise[i]=0;
		//k_param.diameter[i]=0;
		//train_class_num[i]=0;
	}
	for (i = 0; i < k_param.k; i++)//����ÿ���أ�����
	{
		if (kprob->use[i] == 0) continue;
		sum0->index = -1;///
		sum0->value = 0;
		for (j = 0; j < kprob->l; j++)
		{
			if (kprob->y[j] == k_param.y_c[i])
			{
				add(sum0, kprob->x[j]);
				copy(result, sum0);
				k_param.k_c[i]++;
				count1[i]++;
				cluster[j] = i;
				c[j] = kprob->y[j];
			}
		}
		t = 0;
		while (sum0[t].index != -1)
		{
			k_param.x_c[i][t].index = sum0[t].index;
			k_param.x_c[i][t].value = sum0[t].value / k_param.k_c[i];//����ƽ��ֵ
			t++;
		}
		k_param.x_c[i][t].index = -1;
	}

	printf("\n train class num :");
	for (i = 0; i < k_param.k; i++)//����ÿ���أ�����
	{
		printf("   %d", k_param.k_c[i]);
		//class_prob[i]= (double)count[i]/kprob->l;�ĵ�load_problem��ͳ��
		for (j = 0; j < kprob->l; j++)
		{
			if (kprob->y[j] == k_param.y_c[i])
				sse[i] += distance1(kprob->x[j], k_param.x_c[i])*distance1(kprob->x[j], k_param.x_c[i]);
		}
		sse[i] = sse[i] / count1[i];
		count_acc[i] = count1[i];
		cluster_acc[i] = 1;
		//train_class_num[i]= count[i];
	}

	/*printf("\n Test class num : "); //ͳ�Ʋ���������ÿ�������������
	for(i=0;i<k_param.k;i++)
	{
	  test_class_num[i]=0;
	  for(j=0;j<test_data.l;j++)
	  {
		  if((int)test_data.r_y[j]==model->label[i])
			  test_class_num[i]++;
	  }
	  printf("    %d",test_class_num[i]);
	}*/
	//free(sum0);
	//sum0 = NULL;
}

void kmean_saveinformation(kmean_param * kpara,svm_problem * kprob)//�����ϴ�kmean�������Ϣ
{
	   int i,j;
	   old_total_sse = total_sse; old_train_acc = train_acc; /////��������ָ��
	    for(i=0;i<kprob->l;i++)////�����ϴεľ�����
	    {
		  old_cluster[i]=cluster[i];////
		  old_c[i]=c[i];
		}
		old_k = kpara->k;				
		for (j = 0; j < kpara->k; j++) //�����ϴε����������Ϣ
		{
			old_count[j] = count1[j];
			old_count_acc[j] = count_acc[j];
			copy(kpara->x_c[j], Center[j]);//�������ǰ������		
			old_yc[j] = kpara->y_c[j];
			old_cluster_acc[j] = cluster_acc[j];
			old_sse[j] = sse[j];
		}
}

double kmean_predict0(kmean_param * kpara, svm_problem * kprob)// ��ע����Ⱥ��ͱ�Ե
{
	int i, j, t = 0, acc = 0;//��ʼ������
	double min, sum = 0;
	//double *d = new double[kpara->k];
	for (j = 0; j < kpara->k; j++)
	{
		count1[j] = 0; count_acc[j] = 0; sse[j] = 0;
		kpara->k_c[j] = 0;  //kpara->diameter[j] = 0; kpara->noise[j] = 0; 		
		d[j] = 0;
		kpara->size[j] = 0;
	}
	for (i = 0; i < kprob->l; i++)///////////////////////
	{
		if (kprob->use[i] == 0) continue;
		//edge_flag0[i]=false;
		kmeans_noise_flag[i] = false;
		d[0] = min = distance1(kpara->x_c[0], k_prob.x[i]);
		sum = 1.0 / d[0];
		c[i] = kpara->y_c[0];
		cluster[i] = 0;
		for (j = 1; j < kpara->k; j++)
		{
			d[j] = distance1(kpara->x_c[j], kprob->x[i]);
			if (min > d[j])
			{
				min = d[j]; c[i] = kpara->y_c[j]; cluster[i] = j;
			}
			sum += 1.0 / d[j];
		}
		//����ÿ��������Ԥ����ʣ�����
		double sub_min;
		int index;
		if (cluster[i] == 0)
		{
			sub_min = d[1]; index = 1;
		}
		else
		{
			sub_min = d[0]; index = 0;
		}
		for (j = 0; j < kpara->k; j++)
			if (sub_min > d[j] && j != cluster[i])
			{
				sub_min = d[j]; index = j;
			}
		u[i] = 1 - min / sub_min;//sub_min-min;	    //�����δ���Ϊ���Ŷ�
		//u[i]=(1.0/d[cluster[i]])/sum;
		//if(u[i]<0.05) edge_flag0[i]=true;
		if (kpara->y_c[cluster[i]] != k_prob.y[i] && u[i] > 0.95)
			kmeans_noise_flag[i] = true;
		//if(kmeans_noise_flag[i]==false || kmeans_noise_flag[i]==false) )
		{
			count1[cluster[i]]++;//////////////////////
			kpara->k_c[cluster[i]]++;
			kpara->size[cluster[i]]++;
			sse[cluster[i]] += min;
			t++;
			if (c[i] == kprob->y[i]) acc++;
		}
	}
	train_acc = double(acc) / t;
	printf("\n Kmean train accuracy is %f,(%d/%d)", double(acc) / kprob->l, acc, kprob->l);
	//printf("\n cluster number: %d", kpara->k); 
	for (j = 0; j < kpara->k; j++)
		sse_labeled[j] = sse[j];
	avg_cluster_acc = 0;
	for (i = 0; i < kpara->k; i++)  //�������ڸô���Ԥ����ȷ��ѵ��������SSE
	{
		for (j = 0; j < kprob->l; j++)
		{
			if (kprob->use[i] == 0) continue;
			if (cluster[j] == i && kprob->y[j] == c[j])//&& edge_flag0[i]==false && kmeans_noise_flag[i]==false)
			{
				sse_labeled_acc[i] += distance1(kpara->x_c[i], kprob->x[j]); count_acc[i]++;
			}
		}
		sse_labeled_acc[i] = sse_labeled_acc[i] / count_acc[i];
		if (count1[i] == 0)
			cluster_acc[i] = 0;
		else
			cluster_acc[i] = (double)count_acc[i] / count1[i];
		avg_cluster_acc += cluster_acc[i];
		printf("\n Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", i, kpara->y_c[i], cluster_acc[i], count_acc[i], count1[i]);
		//printf("   Average SSE  %f", sse[i]/count[i]);
	}
	avg_cluster_acc = avg_cluster_acc / kpara->k;

	// ѵ�����������������F1-VALUE
	setClusterConfusionMatrix(1);
	printf("\nTrain data f1-value:%f\n", train_f1);

	/*for (i = 0; i < model->nr_class; i++)  //�������ڸ�����Ԥ����ȷ��ѵ������
	{
		train_class_num[i]=0;
		class_acc[i] = 0; class_total[i] = 0;
		for (j = 1; j < kpara->k; j++)//�������ڸ��������ѵ������
		{
			if (model->label[i] == kpara->y_c[j])
				train_class_num[i]+=kpara->k_c[j];
		}
		for (j = 1; j < kprob->l; j++)
		{
			if (model->label[i] == c[j])
			{
				class_total[i]++;
				if(kprob->y[j] == kpara->y_c[i])  class_acc[i]++;
			}
		}
		//printf("   %d", class_acc[i]);
	}
	//printf("\n   ���ذ��������������");

	bool appear;
	//printf("\n   ���ذ��������������");
	for(i = 0; i < kpara->k; i++)//�ҵ��ô��а���������������ִ��ģ�
	{
		cluster_classnum[i]=0;
		for (j = 0; j < model->nr_class; j++)
		{
			appear=false; //�����ѵ�������Ƿ����
			if(model->label[j]!=kpara->y_c[i])
			{
			  for (t = 0; t < kprob->l; t++)
			  {
				if (model->label[j] == kprob->y[t] && cluster[t]==i)// && edge_flag0[t]==false && kmeans_noise_flag[t]==false)
					appear=true;
			  }
			}
			if(appear==true) cluster_classnum[i]++;
		}
		//printf("   %d", cluster_classnum[i]);
	}
	total_UC=0;
	avg_cluster_UC=0;
	double aa,aa1,bb=0;
	for(i = 0; i < kpara->k; i++)//����UC
	{
		aa=(double)cluster_classnum[i]/model->nr_class;
		aa1=(double)cluster_classnum[i]/model->nr_class/2;
		bb+=(double)count_acc[i]/(kprob->l)*(1-pow(aa,3));
		//total_UC+=(double)count_acc[i]/(kprob->l)*(1-pow(aa,3));
		//total_UC+=pow((double)count_acc[i]/(kprob->l),(1+aa*aa*aa));
		//cluster_UC[i]=(double)kpara->k_c[i]/(kprob->l)*pow((double)count_acc[i]/kpara->k_c[i],(1+aa*aa));
		if(kpara->k_c[i]==0)
			cluster_UC[i]=0;
		else
			cluster_UC[i]=(double)kpara->k_c[i]/(kprob->l)*pow((double)count_acc[i]/kpara->k_c[i],(1+aa1));
		total_UC+=cluster_UC[i];
		avg_cluster_UC+=cluster_UC[j];
	}
	avg_cluster_UC=avg_cluster_UC/kpara->k;
	//delete [] d;
	//d=NULL;
	*/
	return train_acc;
}

double kmean_predict1(kmean_param * kpara, svm_problem * kprob)// ���б�ǩ����,��Ȩ׼ȷ��
{
	int i, j, t, acc0 = 0, acc = 0;//��ʼ������
	double tt, min, sum = 0;
	for (j = 0; j < kpara->k; j++)
	{
		count1[j] = 0; count_acc[j] = 0; sse[j] = 0;
		kpara->k_c[j] = 0;  //kpara->diameter[j] = 0; kpara->noise[j] = 0; 		
		d[j] = 0;
	}
	for (i = 0; i < kprob->l; i++)///////////////////////
	{
		d[0] = min = distance1(kpara->x_c[0], kprob->x[i]);
		sum = 1.0 / d[0];
		c[i] = kpara->y_c[0];
		cluster[i] = 0;
		for (j = 1; j < kpara->k; j++)
		{
			d[j] = distance1(kpara->x_c[j], kprob->x[i]);
			if (min > d[j])
			{
				min = d[j]; c[i] = kpara->y_c[j]; cluster[i] = j;
			}
			sum += 1.0 / d[j];
		}
		count1[cluster[i]]++;//////////////////////
		kpara->k_c[cluster[i]]++;
		sse[cluster[i]] += min;
		u[i] = (1.0 / d[cluster[i]]) / sum;
		if (c[i] == kprob->y[i])
		{
			//acc0++;
			if (i < initial_train_size)
				acc0++;
			else
				acc++;
		}
	}
	if (kprob->l == initial_train_size)
		train_acc = double(acc0) / initial_train_size;
	else
	{
		tt = double(acc) / (kprob->l - initial_train_size);
		train_acc = (double(acc0) / initial_train_size + tt * weight_kmeans) / (1 + weight_kmeans);
	}
	printf("\n Kmean train accuracy is %f,(%d+%d/%d+%d)", train_acc, acc0, acc, initial_train_size, kprob->l - initial_train_size);
	//delete [] d;
	//d=NULL;
	return train_acc;
}

double kmean_predict_testdata(kmean_param * kpara) // ���ޱ�ǩ��������Ԥ��
{
	int i, j, acc = 0;//��ʼ������
	double min, sub_min;
	for (i = 0; i < test_data.l; i++)
	{
		d1[0] = min = distance1(kpara->x_c[0], test_data.x[i]);
		c1[i] = kpara->y_c[0];
		cluster1[i] = 0;
		for (j = 1; j < kpara->k; j++)
		{
			d1[j] = distance1(kpara->x_c[j], test_data.x[i]);
			if (min > d1[j])
			{
				min = d1[j]; c1[i] = kpara->y_c[j]; cluster1[i] = j;
			}
		}
		sse[cluster1[i]] += min;
		//����ÿ��������Ԥ����ʣ�����
		sub_min = 100000;
		for (j = 0; j < kpara->k; j++)
			if (sub_min > d1[j] && fabs(min - d1[j]) >= 0.00001)
				sub_min = d1[j];
		u1[i] = sub_min - min;	    //��С����С��Ϊ���Ŷ�
		// printf("  %d",cluster1[i]);
	}

	kmeans_del = 0;
	int total = 0;//���ζȴ����������
	for (i = 0; i < test_data.l; i++)
		kmeans_del += u1[i];
	kmeans_del = kmeans_del / test_data.l; //delΪ���в���������ƽ�����Ŷ�
	for (i = 0; i < test_data.l; i++)
	{
		kpara->k_c[cluster1[i]]++;
		if (u1[i] > kmeans_del)
		{
			total++;
			//kpara->k_c[cluster1[i]]++;
			//final_num[cluster1[i]]++;
			if (c1[i] == test_data.r_y[i]) acc++;
		}
	}
	printf("\n Kmean confident-test accuracy is %f,(%d/%d)", double(acc) / total, acc, total);

	acc = 0; total_sse = 0;
	for (i = 0; i < test_data.l; i++)
		if (c1[i] == test_data.r_y[i]) acc++;
	printf("\n Kmean test accuracy is %f,(%d/%d)", double(acc) / test_data.l, acc, test_data.l);
	test_acc = double(acc) / test_data.l;
	for (j = 0; j < kpara->k; j++)
	{
		acc = 0;
		for (i = 0; i < test_data.l; i++)
		{
			if (cluster1[i] == j && c1[i] == test_data.r_y[i]) acc++;
		}
		//printf("\n Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", j, kpara->y_c[j], double(acc) / (kpara->k_c[j]-count[j]), acc, kpara->k_c[j] - count[j]);
		total_sse += sse[j];
		sse[j] = sse[j] / kpara->k_c[j];
		//printf(" AVG SSE  %lf ", sse[j]);
	}
	total_sse = total_sse / test_data.l;
	printf("\n Total_AVG SSE %lf \n", total_sse);
	//delete [] d;
	//d=NULL;
	//free(final_num);
	//final_num=NULL;	
	return test_acc;
}

double kmean_predict_testdata0(kmean_param * kpara) // ��ȫ���࣬���Ǳ�Ե�㣬��Ⱥ��
{
	int i, j, acc1, acc = 0;//��ʼ������
	int total = 0;//
	int index;
	old_total_sse = total_sse;
	total_sse = 0;
	double min, sub_min;
	double *d = new double[kpara->k];
	int *final_num = new int[kpara->k];
	for (j = 0; j < kpara->k; j++)  final_num[j] = 0;
	kmeans_del = 0;
	double * prob_kmeans = new double[model->nr_class];
	double sum, min_p;
	int t, min_index, sub_min_index;
	int id; // ������model->label�ϵ�����
	for (i = 0; i < test_data.l; i++)
	{
		d[0] = min = distance1(kpara->x_c[0], test_data.x[i]);
		c1[i] = kpara->y_c[0];
		cluster1[i] = 0; min_index = 0;
		//edge_flag[i]=false;	
		sum = d[0];
		for (j = 1; j < kpara->k; j++)
		{
			d[j] = distance1(kpara->x_c[j], test_data.x[i]);
			sum += d[j];
			if (min > d[j])
			{
				min = d[j]; c1[i] = kpara->y_c[j]; cluster1[i] = j; min_index = j;
			}
		}
		sse[cluster1[i]] += min;
		kpara->size[cluster1[i]]++;
		//u1[i]=1-min/sum;
		if (cluster1[i] == 0)
		{
			sub_min = d[1]; sub_min_index = 1;
		}
		else
		{
			sub_min = d[0]; sub_min_index = 0;
		}
		for (j = 0; j < kpara->k; j++)
			if (sub_min > d[j] && j != cluster1[i])
			{
				sub_min = d[j]; sub_min_index = j;
			}
		u1[i] = 1 - min / sub_min;//	    //�����δ���Ϊ���Ŷ�

		//////////////////////////////////// ���ݸ����ص����ľ������������Ԥ�����
		for (t = 0; t < model->nr_class; t++)
		{
			prob_kmeans[t] = 0;
			for (j = 0; j < kpara->k; j++)
			{
				if (fabs(kpara->y_c[j] - model->label[t]) < 1e-6)
					prob_kmeans[t] += d[j];
			}
			prob_kmeans[t] = (sum - prob_kmeans[t]) / sum;
			kmeans_confidence[i][t] = prob_kmeans[t];
			if (model->label[t] == 1) id = t;
		}
		/* ����������ۺϼ���
		min_p=prob_kmeans[0];
		min_index=0;
		for(t=1;t<model->nr_class;t++)
		{
				//prob_kmeans[t]=prob_kmeans[t]/sum;
				if(min_p>prob_kmeans[t])
				{
					min_p=prob_kmeans[t];
					min_index=t;
				}
		}
		c1[i]=model->label[min_index];
		u1[i]=1-prob_kmeans[min_index];
		*///////////////////////////////////////////////////	
		total++;
		kmeans_del += u1[i];
		if (c1[i] == test_data.r_y[i]) acc++;
		//printf("  %f",c1[i]);

	}
	printf("\n Kmean test accuracy is %f,(%d/%d)", double(acc) / test_data.l, acc, test_data.l);
	first_kmean_predict_testdata0 = double(acc) / test_data.l; // ��¼��һ��keamnsδ���ѵ�ACC

	// ����kmeans�������ϵ�Ԥ�����
	for (int i = 0; i < test_data.l; i++) {
		u3[i] = kmeans_confidence[i][id];
	}

	kmeans_del = kmeans_del / total; //delΪ���в���������ƽ�����Ŷ�
	total = 0;
	acc = 0;
	for (i = 0; i < test_data.l; i++)
	{
		//if(edge_flag[i]==false)
		{
			kpara->k_c[cluster1[i]]++;
			if (u1[i] > kmeans_del)
			{
				total++;
				//kpara->k_c[cluster1[i]]++;
				final_num[cluster1[i]]++;
				if (c1[i] == test_data.r_y[i]) acc++;
			}
		}
	}
	printf("\n Kmean confident-test accuracy is %f,(%d/%d)", double(acc) / total, acc, total);

	int nn;
	for (j = 0; j < kpara->k; j++)
	{
		acc = 0; nn = 0;
		for (i = 0; i < test_data.l; i++)
		{
			if (cluster1[i] == j)// && edge_flag[i]==false)
			{
				nn++;
				if (c1[i] == test_data.r_y[i]) acc++;
				//if ( cluster1[i] == j && c1[i] == test_data.y[i] && edge_flag[i]==false) acc++;
			}
		}
		printf("\n Cluster %d (Label %f) accuracy is %f,(%d/%d)", j, kpara->y_c[j], double(acc) / nn, acc, nn);
		total_sse += sse[j];
		sse[j] = sse[j] / kpara->k_c[j];
		//printf("     %lf", sse[j]);
	}
	printf("\n %lf", total_sse);
	
	// ���Լ����������������Լ�f1-value
	setClusterConfusionMatrix(0);
	printf("\nTestdata f1-value:%f\n", test_f1);

	free(final_num);
	final_num = NULL;
	return total_sse - old_total_sse; //SSE�ı仯���
}

void kmean_final_predict(kmean_param * kpara) // ����ͬ�����صľ���ͽ���Ԥ��
{
	int i, j, t, n, acc = 0;//��ʼ������
	double max, sum0, sum;
	// sum0:�������������д����ĵľ����
	// sum:����������������ͬ��ǩ�Ĵ����ĵľ����
	for (i = 0; i < test_data.l; i++)
	{
		sum0 = 0;
		for (t = 0; t < kpara->k; t++)
		{
			sum0 += distance1(kpara->x_c[t], test_data.x[i]);
		}
		max = -1;
		for (j = 0; j < model->nr_class; j++)
		{
			sum = 0;
			for (t = 0; t < kpara->k; t++) {
				if (kpara->y_c[t] == model->label[j])
					sum += distance1(kpara->x_c[t], test_data.x[i]) / sum0;
			}
			sum = (1 - sum) / (model->nr_class - 1);
			if (max < sum)
			{
				max = sum; //c1[i] = model->label[j]; // ���Ϊ�˼���sskmeans��mf��ע��
			}
		}
		u1[i] = max; // ����i��Ԥ��Ϊ���c1[i]�ĸ���
		if (c1[i] == test_data.y[i]) acc++;
	}
	printf("\n Kmean test accuracy is %f,(%d/%d)", double(acc) / test_data.l, acc, test_data.l);

	// �����ලkmeans��AUC��F1ָ��
	//tAUC_ROC = getAUC_ROC();
	double tAuc = sskmeans_auc(u1);
	// getPerformance(); // ��ȡ��ʹ��SVM�����ܲ���
	setClusterConfusionMatrix(0); 
	printf("\nsskmeans��AUC��%f  mF1score: %f\n", tAuc, test_f1);
	resultMap[3].push_back(tAuc);
	resultMap[3].push_back(test_f1);
}

void kmean_update113(kmean_param * kpara, svm_problem * kprob)//���¸����ص�����
{
	int i, j, t;
	svm_node *sum2 = Malloc(struct svm_node, max_index + 1);
	double update_del = 0;
	for (i = 0; i < kpara->k; i++)
		copy(kpara->x_c[i], Center[i]);
	for (i = 0; i < kpara->k; i++)//����ÿ���أ���/////////��ʼ����������
	{
		kpara->k_c[i] = 0;
		sum2[0].index = -1;
		sum2[0].value = 0;
		for (j = 0; j < k_prob.l; j++)//����ѵ������
		{
			if (k_prob.use[j] == 0) continue;
			if (cluster[j] == i && c[j] == kprob->y[j])
			{
				(kpara->k_c[i])++;
				add(sum2, k_prob.x[j]);
				copy(result, sum2);
			}
		}
		/*for(j=0;j<test_data.l;j++)//���ڲ�������
		{
			if(cluster1[j]==i && u1[j]>del)//&& c1[j] == test_data.y[j])////////////�ҵ����ŶȸߵĲ���
			//if (cluster1[j] == i &&  c1[j] == test_data.y[j])
			{
			  (kpara->k_c[i])++;
			  add(sum2,test_data.x[j]);
			  copy(result,sum2);
			}
		}*/
		t = 0;
		while (sum2[t].index != -1)
		{
			kpara->x_c[i][t].index = sum2[t].index;
			kpara->x_c[i][t].value = sum2[t].value / kpara->k_c[i];//����ƽ��ֵ
			t++;
		}
		kpara->x_c[i][t].index = -1;
	}
	//for(i=0;i<kpara->k;i++)
	  // update_del+=distance1(Center[i],kpara->x_c[i]);
	//free(sum2);
}

bool kmean_delete_centroid(kmean_param * kpara, svm_problem * kprob, int no)//ɾ����No����
{
	int i,j;
	bool f=false;
	for(i=0;i<kpara->k;i++)//�����Ƿ��и�no����ǩһ���Ĵ�
	{
		if(kpara->y_c[i]==kpara->y_c[no] && i!=no)
			f=true;
	}
	if(!f)  return false;//�����ĳ����Ψһ�Ĵأ�����ɾ
	(kpara->k)--;
	//old_k--;

	/*int minId;
	double minDis;
	for (i = 0; i < kprob->l; i++) {
		minDis = 999999;
		minId = -1;
		if (cluster[i] == no) {
			for (j = 0; j < kpara->k; j++) {
				if (j == no) continue;
				if (kpara->y_c[j] == kprob->r_y[i]) { // ��ͬ�����������Ĵ�
					double dis = distance1(kpara->x_c[j], kprob->x[i]);
					if (minDis > dis) {
						minDis = dis;
						minId = j;
					}
				}
			}
			if (minId != -1)// && kpara->y_c[minId] == kprob->r_y[i]) 
			{
				printf("ѵ������%d�Ӵ�%d-->��%d\n", i, cluster[i], minId);
				cluster[i] = minId;
				kpara->size[minId]++;
				count1[minId]++;
			}
			else {
				cluster[i] = -1;
				prob.use[i] = 0;
				kprob->use[i] = 0;
			}
		}
	}*/

	printf("\n Delete cluster %d;", no);
	for(i=no;i<kpara->k;i++)//����ÿ���أ�ǰ��һλ////
	{	
		kpara->k_c[i]=kpara->k_c[i+1];
		kpara->y_c[i]=kpara->y_c[i+1];
		copy(kpara->x_c[i+1],kpara->x_c[i]);
		//kpara->diameter[i] = kpara->diameter[i + 1];
		//kpara->noise[i] = kpara->noise[i + 1];
		kpara->size[i] = kpara->size[i + 1];
		kpara->pos[i] = kpara->pos[i + 1];
		count1[i] = count1[i+1]; 	
		old_count[i] = old_count[i+1];
		cluster_acc[i]=cluster_acc[i+1]; 
		old_cluster_acc[i]=old_cluster_acc[i+1]; 
		sse[i] =sse[i+1]; old_sse[i] =old_sse[i+1]; 
		parent[i] =parent[i+1];
		new_split[i]=new_split[i+1];
		count_acc[i]=count_acc[i+1];
		old_count_acc[i] = old_count_acc[i+1];		
		//coef[i] = coef[i + 1];
	}

	/*for (j = 0; j<k_prob.l; j++)//����ѵ������
	{
		if (cluster[j] == no)
		{
			cluster[j] = maxnum - 1; old_cluster[j] = maxnum - 1;////��ʾ��Ⱥ��
		}
		else if (cluster[j] > no)  //&& (c[j] == k_prob.y[j]))
		{
			cluster[j] = cluster[j] - 1; old_cluster[j] = old_cluster[j] - 1;
		}
	}*/
	/*for (j = 0; j<test_data.l; j++)//���ڲ�������
	{
		if (cluster1[j] == no)   cluster1[j] = maxnum -1;
		if (cluster1[j] > no)  
			cluster1[j] = cluster1[j] - 1;
	}*/
	return true;
}

bool kmean_delete_centroid_samples(kmean_param * kpara, svm_problem * kprob, int no) // ɾ����no���أ������е�����
{
	int i, j;
	bool f = false;
	for (i = 0; i < kpara->k; i++) //�����Ƿ��и�no����ǩһ���Ĵ�
	{
		if (kpara->y_c[i] == kpara->y_c[no] && i != no)
			f = true;
	}
	if (!f)  return false;//�����ĳ����Ψһ�Ĵأ�����ɾ

	(kpara->k)--; // Ӧ����ǰ�Ƹ��ǣ�Ȼ���ټ����ɣ�����
	//old_k--;
	printf("\n Delete cluster %d;", no);

	//printf("\n��%d��ɾ��", no);
	// ��ɾ���ô�ǰ����ɾ���ô��ڵ�����ѵ������
	for (i = 0; i < prob.l; i++) {
		if (cluster[i] == no) {
			prob.use[i] = 0;
			k_prob.use[i] = 0;
			printf("\nѵ������ %d ɾ��", i);
		}
	}

	//kpara->use[no] = 0;
	for (i = no; i < kpara->k; i++) //����ÿ���أ�ǰ��һλ////
	{
		kpara->k_c[i] = kpara->k_c[i + 1];
		kpara->y_c[i] = kpara->y_c[i + 1];
		kpara->size[i] = kpara->size[i + 1];
		kpara->pos[i] = kpara->pos[i + 1];
		copy(kpara->x_c[i + 1], kpara->x_c[i]);
		//kpara->diameter[i] = kpara->diameter[i + 1];
		//kpara->noise[i] = kpara->noise[i + 1];		
		count1[i] = count1[i + 1];
		old_count[i] = old_count[i + 1];
		cluster_acc[i] = cluster_acc[i + 1];
		old_cluster_acc[i] = old_cluster_acc[i + 1];
		sse[i] = sse[i + 1]; old_sse[i] = old_sse[i + 1];
		parent[i] = parent[i + 1];
		new_split[i] = new_split[i + 1];
		count_acc[i] = count_acc[i + 1];
		old_count_acc[i] = old_count_acc[i + 1];
		//coef[i] = coef[i + 1];
	}

	for (j = 0; j < k_prob.l; j++)//����ѵ������
	{
		if (cluster[j] == no)
		{
			cluster[j] = maxnum - 1; old_cluster[j] = maxnum - 1;////��ʾ��Ⱥ��
		}
		else if (cluster[j] > no)  //&& (c[j] == k_prob.y[j]))
		{
			cluster[j] = cluster[j] - 1; old_cluster[j] = old_cluster[j] - 1;
		}
	}
	/*for (j = 0; j<test_data.l; j++)//���ڲ�������
	{
		if (cluster1[j] == no)   cluster1[j] = maxnum -1;
		if (cluster1[j] > no)
			cluster1[j] = cluster1[j] - 1;
	}*/

	return true;
}

double kmean_update_train(kmean_param * kpara, svm_problem * kprob, int no)//����ѵ���������µ�No���ص�����
{
	int j, t;
	double del = 0;
	//svm_node *sum_for_train = Malloc(struct svm_node, max_index + 1);
	//svm_node *Center = Malloc(struct svm_node, max_index + 1);
	sum_for_train->index = -1;	  sum_for_train->value = 0;
	//copy(kpara->x_c[no], Center);//����ԭ����
	count1[no] = 0;
	for (j = 0; j<k_prob.l; j++)//����ѵ������
	{
		if ((cluster[j] == no) && (c[j] == k_prob.y[j]))
		{
			(count1[no])++;
			add(sum_for_train, k_prob.x[j]);
			copy(result, sum_for_train);
		}
	}
	t = 0;
	while (sum_for_train[t].index != -1)
	{
		kpara->x_c[no][t].index = sum_for_train[t].index;
		kpara->x_c[no][t].value = sum_for_train[t].value / count1[no];//����ƽ��ֵ
		t++;
	}
	kpara->x_c[no][t].index = -1;
	//del = distance(Center, kpara->x_c[no]);
	//free(sum_for_train);
	//free(Center);
	return del;
}
double kmean_update(kmean_param * kpara, svm_problem * kprob)//ȫ���������
{
	int i, j, t;
	double del = 0;
	for (i = 0; i<kpara->k; i++)
	{
		sum_update[i]->index = -1;	  sum_update[i]->value = 0;
		copy(kpara->x_c[i], Center[i]);
		kpara->k_c[i] = 0;
	}

	for (j = 0; j<k_prob.l; j++)//����ѵ������
	{
		kpara->k_c[cluster[j]]++;
		add(sum_update[cluster[j]], k_prob.x[j]);
		copy(result, sum_update[cluster[j]]);
	}
	for (j = 0; j<test_data.l; j++)//���ڲ�������
	{
		kpara->k_c[cluster1[j]]++;
		add(sum_update[cluster1[j]], test_data.x[j]);
		copy(result, sum_update[cluster1[j]]);
	}
	for (i = 0; i<kpara->k; i++)
	{
		t = 0;
		while (sum_update[i][t].index != -1)
		{
			kpara->x_c[i][t].index = sum_update[i][t].index;
			kpara->x_c[i][t].value = sum_update[i][t].value / kpara->k_c[i];//����ƽ��ֵ
			t++;
		}
		kpara->x_c[i][t].index = -1;
	}//

	for (i = 0; i<kpara->k; i++)
	{
		del += distance1(Center[i], kpara->x_c[i]); 
	}
	return del;
}

double kmean_update1(kmean_param * kpara,svm_problem * kprob)//ֻ�������ŶȸߵĲ��ֽ�����¸����ص�����
{
    int i,j,t;
	svm_node *sum2=Malloc(struct svm_node,max_index+1);
	double update_del=0;
	for (i = 0; i < kpara->k; i++)  
		copy(kpara->x_c[i], Center[i]);
	for(i=0;i<kpara->k;i++)//����ÿ���أ���/////////��ʼ����������
	{				  
		  kpara->k_c[i]=0;
		  sum2[0].index=-1;
		  sum2[0].value=0;		  
		  for(j=0;j<k_prob.l;j++)//����ѵ������
		  {
			if(cluster[j]==i && c[j]==kprob->y[j])
			{
				(kpara->k_c[i])++;
				add(sum2,k_prob.x[j]);
				copy(result,sum2); 
			}
		  }
		  for(j=0;j<test_data.l;j++)//���ڲ�������
		  {
			  if(cluster1[j]==i && u1[j]>del)//&& c1[j] == test_data.y[j])////////////�ҵ����ŶȸߵĲ���
		      //if (cluster1[j] == i &&  c1[j] == test_data.y[j])
			  {
				(kpara->k_c[i])++;
				add(sum2,test_data.x[j]);
				copy(result,sum2);	
			  }
		  }
		  t=0;
		  while(sum2[t].index!=-1)
		  {
			kpara->x_c[i][t].index=sum2[t].index;
			kpara->x_c[i][t].value=sum2[t].value/kpara->k_c[i];//����ƽ��ֵ
			t++;
	 	  }
		  kpara->x_c[i][t].index=-1;
	}
	for(i=0;i<kpara->k;i++)
	   update_del+=distance1(Center[i],kpara->x_c[i]);
	free(sum2);
	return update_del;
}

bool kmean_split1(kmean_param * kpara, svm_problem * kprob, int no) //�Ե�ǰ�ذ���������ĸ������ѵ���������з��ѡ�       cluster[j]=kpara->k;û��
{
	/*if (count1[no] <= 1)
	{
		kmean_delete_centroid(kpara, &k_prob, no);
		old_k--;
		return true;//��Ϊ�����б仯
	}*/
	int i, j, k, t, n = 0;
	double del = 0;
	bool flg = false;
	int cluster_size = count1[no];
	/*double sse01=0;
	for (j = 0; j<kprob->l; j++)  //�������ڸôص�������SSE
	{
		if (cluster[j] == no) sse01+=distance(kpara->x_c[no],kprob->x[j]);
	}
	sse01=sse01/cluster_size;*/

	for (k = 0; k < model->nr_class; k++)//�ж������ĸ����
	{
		num[k] = 0;
		sum[k]->index = -1;
		sum[k]->value = 0;
		for (j = 0; j < kprob->l; j++) //����ô��и�����������ġ����ġ�
		{
			if (cluster[j] == no && kprob->y[j] == model->label[k])//  && (kmeans_noise_flag[j] == false || svm_noise_flag[j] == false) )
			{
				(num[k])++;
				add(sum[k], kprob->x[j]);
				copy(result, sum[k]);
			}
		}
	}
	printf("\n Split cluster %d: ", no);
	bool del_flg = false;
	for (k = 0; k < model->nr_class; k++) 
	{
		printf("  Label-%d: %d", model->label[k], num[k]);
		if (model->label[k] == kpara->y_c[no]) //�ȰѸô���Ԥ����ȷ��������Ϊһ����
		{
			if (num[k] > 0)
			{
				t = 0;
				while (sum[k][t].index != -1)
				{
					kpara->x_c[no][t].index = sum[k][t].index;
					kpara->x_c[no][t].value = sum[k][t].value / num[k];//����ƽ��ֵ
					t++;
				}
				kpara->x_c[no][t].index = -1;
				//count[no] = num[k];
				parent[no] = no;
				new_split[no] = 2;
			}
			else
				del_flg = true;
		}
		else if (num[k] < 2 && model->pos[k] == 0) { // ��ѵ����ɾ����Щ��������
			for (i = 0; i < k_prob.l; i++) {
				if (cluster[i] == no && k_prob.pos[i] == 0) {
					printf("\nɾ����%d��ѵ����������ǩ��:%.0f", i, k_prob.r_y[i]);
					prob.use[i] = 0;
					k_prob.use[i] = 0;
				}
			}
		}
		//else if ((num[k] > 2 || (double)num[k] / cluster_size > 0.05) && kpara->k < maxnum)//������������������������ж��Ƿ���Ҫ����/// if((double)num[k]/kpara->k_c[k]>split_m)
		else if((num[k] > 2 || model->pos[k] == 1) && kpara->k < maxnum)
		{
			flg = true; //��Ҫ����
			t = 0;
			while (sum[k][t].index != -1)
			{
				kpara->x_c[kpara->k][t].index = sum[k][t].index;
				kpara->x_c[kpara->k][t].value = sum[k][t].value / num[k];//����ƽ��ֵ
				t++;
			}
			kpara->x_c[kpara->k][t].index = -1;
			kpara->y_c[kpara->k] = model->label[k];
			//count[kpara->k]=num[k];//////
			parent[kpara->k] = no;
			new_split[kpara->k] = 1;
			(kpara->k)++;
		}
	}
	if (del_flg)//�����ȷ��ѵ����������Ϊ0����ѵ�no���ظ��ǵ�
	{
		copy(kpara->x_c[kpara->k - 1], kpara->x_c[no]);
		kpara->y_c[no] = kpara->y_c[kpara->k - 1];
		parent[no] = 1;
		new_split[no] = 1;
		(kpara->k)--;
	}
	/*/����SSE�ж��Ƿ���Ҫ�����ڸôص���ȷ�������ѳ�������
	int cc=0;double sse00=0;
	for (j = 0; j<kprob->l; j++)  //�������ڸôص�������SSE
	{
		if (cluster[j] == no && kprob->y[j]==kpara->y_c[no])
		{
			sse00+=distance(kpara->x_c[no],kprob->x[j]);  cc++;
		}
	}
	sse00=sse00/cc;
	if(sse00>sse01) //������ڸôص�������SSE ���ڸôص�ƽ��SSE������Ѹò�������
		kmean_split5(kpara, &k_prob, no);
	*/
	//for (k = 0; k<model->nr_class; k++) { free(sum[k]); }
	//free(sum);
	//delete [] num;
	return true;
}

bool kmean_split5(kmean_param * kpara, svm_problem * kprob, int no)//���ϴεĴأ��������������Զ��������t���ҳ���t������max֮�ڵĵ㣬���������ģ���������Ϊ��һ��������
{
	int j, t, n1 = 0, n2 = 0;//
	svm_node * tmp1, *tmp2;
	double max = -1, dis = 0;
	tmp1 = Malloc(struct svm_node, max_index + 1);
	tmp2 = Malloc(struct svm_node, max_index + 1);
	tmp1->index = -1; tmp2->index = -1;
	tmp1->value = 0; tmp2->value = 0;

	for (j = 0; j < kprob->l; j++)//�ҳ��ôؾ���������Զ�������㣬t
	{
		select_flg[j] = false;
		if ((old_cluster[j] == no) && kpara->y_c[no] == kprob->y[j])// (kpara->y_c[no]== kprob->y[j]))      //kpara->y_c[no]== kprob->y[j] ???
		{
			if (max < distance1(kprob->x[j], kpara->x_c[no])) ///////////////????
			{
				max = distance1(kprob->x[j], kpara->x_c[no]);
				t = j;
			}
		}
	}
	for (j = 0; j < kprob->l; j++)//�ҳ��ôؾ���t����һ��������
	{
		if (old_cluster[j] == no && kpara->y_c[no] == kprob->y[j] && select_flg[j] == false)
		{
			if (distance1(kprob->x[j], kprob->x[t]) < max) ///////////////????
			{
				select_flg[j] = true;//��ʾ�������ѱ�ѡ��
				add(tmp1, kprob->x[j]);
				copy(result, tmp1);
				n1++;
				cluster[j] = no;
				old_cluster[j] = no;
			}
		}
	}
	for (j = 0; j < kprob->l; j++)//ͳ��ʣ���һ��������
	{
		if (old_cluster[j] == no && kpara->y_c[no] == kprob->y[j] && select_flg[j] == false)
		{
			add(tmp2, kprob->x[j]);
			copy(result, tmp2);
			n2++;
			old_cluster[j] = kpara->k;
			cluster[j] = kpara->k;
		}
	}
	/////////n1,n2�������㣡
	if (n1 < 1 || n2 < 1) return false;
	printf("\n split cluster %d ; ", no);
	if (n1 > 0)
	{
		t = 0;
		while (tmp1[t].index != -1)
		{
			kpara->x_c[no][t].index = tmp1[t].index;
			kpara->x_c[no][t].value = tmp1[t].value / n1;//����ƽ��ֵ
			t++;
		}
		kpara->x_c[no][t].index = -1;
		//count[no] = n1;
		parent[no] = no;
		new_split[no] = 1;
	}
	if (n2 > 0)
	{
		t = 0;
		//printf(" %d ", kpara->k);
		while (tmp2[t].index != -1)
		{
			//printf("%d:%lf ", kpara->x_c[kpara->k][t].index, kpara->x_c[kpara->k][t].value);
			kpara->x_c[kpara->k][t].index = tmp2[t].index;
			kpara->x_c[kpara->k][t].value = tmp2[t].value / n2;//����ƽ��ֵ
			t++;
		}
		//printf("\n");
		kpara->x_c[kpara->k][t].index = -1;
		kpara->y_c[kpara->k] = kpara->y_c[no];////�������´�
		//count[kpara->k] = n2;
		parent[kpara->k] = no;
		new_split[kpara->k] = 1;
		(kpara->k)++;
	}
	return true;
}

bool kmean_split2(kmean_param * kpara, svm_problem * kprob, int no)//�Ե�ǰ�ذ���������ĸ������ѵ���������з��ѡ�       cluster[j]=kpara->k;û��
{
	if (count1[no] <= 1)
	{
		kmean_delete_centroid(kpara, &k_prob, no);
		old_k--;
		return true;//��Ϊ�����б仯
	}
	int i, j, k, t, n = 0;
	double del = 0;
	bool flg = false;
	int cluster_size = count1[no];
	/*double sse01=0;
	for (j = 0; j<kprob->l; j++)  //�������ڸôص�������SSE
	{
		if (cluster[j] == no) sse01+=distance(kpara->x_c[no],kprob->x[j]);
	}
	sse01=sse01/cluster_size;*/

	for (k = 0; k < model->nr_class; k++) //��ʼ��
	{
		num[k] = 0;
		sum[k]->index = -1;
		sum[k]->value = 0;
	}
	for (j = 0; j < kprob->l; j++)//����ô��и�����������ġ����ġ�
	{
		if (cluster[j] == no)
		{
			for (k = 0; k < model->nr_class; k++)//�ж������ĸ����
			{
				if (kprob->y[j] == model->label[k])
				{
					(num[k])++;
					add(sum[k], kprob->x[j]);
					copy(result, sum[k]);
				}
			}
		}
	}
	printf("\n Split cluster %d: ", no);
	bool del_flg = false;
	double splt_del = 0.01;
	if (train_with_noise)		splt_del = 0.1;
	for (k = 0; k < model->nr_class; k++)
	{
		printf("  Label-%d: %d", model->label[k], num[k]);
		if (model->label[k] == kpara->y_c[no])//�ȰѸô���Ԥ����ȷ��������Ϊһ����
		{
			if (num[k] > 0)
			{
				t = 0;
				while (sum[k][t].index != -1)
				{
					kpara->x_c[no][t].index = sum[k][t].index;
					kpara->x_c[no][t].value = sum[k][t].value / num[k];//����ƽ��ֵ
					t++;
				}
				kpara->x_c[no][t].index = -1;
				//count[no] = num[k];
				parent[no] = no;
				new_split[no] = 2;
			}
			else
				del_flg = true;
		}
		else if ((double)num[k] / cluster_size > splt_del && kpara->k < maxnum)//������������������������ж��Ƿ���Ҫ����/// if((double)num[k]/kpara->k_c[k]>split_m)
		{
			flg = true;//��Ҫ����
			t = 0;
			while (sum[k][t].index != -1)
			{
				kpara->x_c[kpara->k][t].index = sum[k][t].index;
				kpara->x_c[kpara->k][t].value = sum[k][t].value / num[k];//����ƽ��ֵ
				t++;
			}
			kpara->x_c[kpara->k][t].index = -1;
			kpara->y_c[kpara->k] = model->label[k];
			//count[kpara->k]=num[k];//////
			parent[kpara->k] = no;
			new_split[kpara->k] = 1;
			(kpara->k)++;
		}
	}
	if (del_flg)//�����ȷ��ѵ����������Ϊ0����ѵ�no���ظ��ǵ�
	{
		copy(kpara->x_c[kpara->k - 1], kpara->x_c[no]);
		kpara->y_c[no] = kpara->y_c[kpara->k - 1];
		parent[no] = 1;
		new_split[no] = 1;
		(kpara->k)--;
	}
	return flg;
}

void iterative_update2(int n)//����co-trainingĿ�꺯���������ĵ���
{
	int i, n1 = n + 1;
	double update_del = 100;
	double tmp_sum_SSE, tmp_sum_ACC; //����ָ����ܺ�	
	double tmp_sum_FVP;
	//if(n==0)
	//{
	//kmean_predict1(&k_param, &k_prob);
	//kmean_predict_testdata(&k_param);
	kmean_predict0(&k_param, &k_prob);
	kmean_predict_testdata0(&k_param);
	//}
	tmp_sum_ACC = history_acc[n] = train_acc; //�������ĸ���ǰ�Ľ��
	tmp_sum_SSE = history_SSE[n] = total_sse;
	tmp_sum_FVP = history_FVP[n] = train_acc;
	printf("\n ��ʼ���ĵ���!\n");
	do
	{
		for (i = 0; i < k_param.k; i++)//����ɵ�����
			copy(k_param.x_c[i], tmp_Center[i]);
		update_del = kmean_update(&k_param, &k_prob);//////////////////////////////////////
		//kmean_predict1(&k_param, &k_prob);
		//kmean_predict_testdata(&k_param);//
		kmean_predict0(&k_param, &k_prob);
		kmean_predict_testdata0(&k_param);
		history_acc[n1] = train_acc; history_SSE[n1] = total_sse; history_FVP[n1] = train_f1;
		tmp_sum_ACC += history_acc[n1];////
		tmp_sum_SSE += history_SSE[n1];
		tmp_sum_FVP += history_FVP[n1];
		//printf("\n %lf   %lf",history_acc[n1-1],history_acc[n1]);printf("\n %lf   %lf",history_SSE[n1-1],history_SSE[n1]);
		//printf("\n %lf   %lf",history_acc[n1-1]/sum_ACC,history_acc[n1]/sum_ACC);printf("\n %lf   %lf",history_SSE[n1-1]/sum_SSE,history_SSE[n1]/sum_SSE);
		//double t1 = weight * history_acc[n1 - 1] / tmp_sum_ACC - (1 - weight)*history_SSE[n1 - 1] / tmp_sum_SSE;
		//double t2 = weight * history_acc[n1] / tmp_sum_ACC - (1 - weight)*history_SSE[n1] / tmp_sum_SSE;
		double t1 = weight * history_FVP[n1 - 1] / tmp_sum_ACC - (1 - weight)*history_SSE[n1 - 1] / tmp_sum_SSE;
		double t2 = weight * history_FVP[n1] / tmp_sum_ACC - (1 - weight)*history_SSE[n1] / tmp_sum_SSE;
		if (t1 > t2)
		{
			printf("\n �����������ĵ������ָ�ԭ����!\n");  //�������ε������ָ�ԭ����
			for (i = 0; i < k_param.k; i++)
				copy(tmp_Center[i], k_param.x_c[i]);
			break;
		}
		n1++;
	} while (update_del > 0.001);//�����Ĳ��ٷ����仯	
}

bool needSplit(int no) // �жϴ˴��Ƿ���Ҫ���ѣ���������������Ԥ���������Ҫ�ٷ���
{
	int accNum = 0;
	int t = 0;
	for (int i = 0; i < k_prob.l; i++) {
		if (k_prob.use[i] == 0) continue;
		if (cluster[i] == no && k_prob.r_y[i] == k_param.y_c[no])
			accNum++;
		if(cluster[i] == no)
			t++;
	}
	if (t - accNum <= 2 && accNum > t / 2) 
		return false;
	else
		return true;	
}

int kmean_train_new111(int No, kmean_param * kpara, svm_problem * kprob)  // C
{
	int i, j, dif, tmp_k, n = 0;
	bool restore, split_flg = true;
	double t1, t2, d, update_del = 100;
	bool *need_split = new bool[maxnum];
	sum_SSE = sum_ACC = 0; //����ָ��d���ܺ�
	sum_FVP = 0; // ������ָ��
	do
	{
		///////////////////��һ���������ϴεľ�����   //////////////////////////////////////////////////////
		for (i = 0; i < kprob->l; i++)
		{
			old_cluster[i] = cluster[i];////
			old_c[i] = c[i];
		}
		for (i = 0; i < test_data.l; i++)
			old_c1[i] = c1[i];//�ϴ�Ԥ��Ľ��
		///////////////////�ڶ��������в���   //////////////////////////////////////////////////////		
		kmean_predict0(kpara, kprob);//�б�ǩ����������Ԥ��
		kmean_predict_testdata0(kpara);//�ޱ�ǩ����������Ԥ��	

		//////////////������������////////////////////////
		if (train_acc >= 1)
			break;
		dif = 0;
		for (i = 0; i < test_data.l; i++)	//ͳ������Ԥ��Ĳ���
			if (old_c1[i] != c1[i]) dif++;
		if (n != 0 && dif < 1)
			break;

		old_k = kpara->k;

		split_flg = false;
		for (i = 0; i < maxnum; i++) //��ʼ�����дصķ���״̬
		{
			parent[i] = -1;	new_split[i] = 0;  need_split[i] = false;
		}

		for (i = 0; i < kprob->l; i++)////�ҳ���Ҫ���ѵĴ�
		{
			if (kprob->r_y[i] != c[i])
				//need_split[old_cluster[i]] = true;
				//need_split[cluster[i]] = true;
				if (kpara->size[old_cluster[i]] > kpara->size[cluster[i]])
					need_split[old_cluster[i]] = true;
				else
					need_split[cluster[i]] = true;
		}

		/* // ����ʵ��Ƚ�
		for (i = 0; i < kprob->l; i++)////�ҳ���Ҫ���ѵĴ�
		{
			if (kprob->r_y[i] != c[i]) {
				need_split[old_cluster[i]] = true;
				//need_split[cluster[i]] = true;
				if (kpara->size[old_cluster[i]] > kpara->size[cluster[i]])
					need_split[old_cluster[i]] = true;
				else
					need_split[cluster[i]] = true;
			}
		}
		*/

		for (i = 0; i < old_k; i++)// ���Ѻ������Ĵ�
		{
			new_split[i] = 0;
			if (kpara->k >= maxnum)
				break;
			if (need_split[i] == true)//&&(double)(kpara->noise[i])/count[i] > split_m)
			{
				if (kmean_split5(kpara, &k_prob, i)) split_flg = true;
			}
		}
		n++;
		printf("\n���ִظ���Ϊ:%d\n", kpara->k);
	} while (split_flg == true && kpara->k < maxnum);
	kmean_predict0(kpara, kprob);
	kmean_predict_testdata0(kpara);

	//iterative_update2(0);//////////
	for (i = 0; i < kpara->k; i++)// ���ο����·��ѵĴأ��Ƿ���Ҫ����
	{
		//printf("%f\n", cluster_acc[i]);
		if (new_split[i] == 1)//�·��ѳ����Ĵ�
		{
			if (cluster_acc[i] < 1.0 / model->nr_class)// || coef[i]< avg1)//train_acc || cluster_acc[i] < 0.5)//(double)class_acc[t]/ class_total[t])//(cluster_acc[i] <= old_cluster_acc[parent[i]] && sse[i] >= old_sse[parent[i]])// 
			{
				if (kmean_delete_centroid(kpara, &k_prob, i))
				{
					i--; restore = true;
				}
			}
		}
		/*if (cluster_acc[i] < 0.5)// && kpara->y_c[i] == 1) 
		{
			kmean_split2(kpara, &k_prob, i);
		}*/
	}

	iterative_update2(0);//////////
	return n;
}

int kmean_train_new113(int No, kmean_param * kpara, svm_problem * kprob)  // C
{
	int i, j, dif, tmp_k, n = 0;
	bool restore, split_flg = true;
	double t1, t2, d, update_del = 100;
	bool *need_split = new bool[maxnum];
	sum_SSE = sum_ACC = 0; //����ָ��d���ܺ�
	sum_FVP = 0; // ������ָ��
	do
	{
		///////////////////��һ���������ϴεľ�����   //////////////////////////////////////////////////////
		for (i = 0; i < kprob->l; i++)
		{
			old_cluster[i] = cluster[i];////
			old_c[i] = c[i];
		}
		for (i = 0; i < test_data.l; i++)
			old_c1[i] = c1[i];//�ϴ�Ԥ��Ľ��
		///////////////////�ڶ��������в���   //////////////////////////////////////////////////////		
		kmean_predict0(kpara, kprob);//�б�ǩ����������Ԥ��
		kmean_predict_testdata0(kpara);//�ޱ�ǩ����������Ԥ��	

		//////////////������������////////////////////////
		if (train_acc >= 1)
			break;
		dif = 0;
		for (i = 0; i < test_data.l; i++)	//ͳ������Ԥ��Ĳ���
			if (old_c1[i] != c1[i]) dif++;
		if (n != 0 && dif < 1)
			break;
		old_k = kpara->k;
		//
		split_flg = false;
		for (i = 0; i < maxnum; i++) //��ʼ�����дصķ���״̬
		{
			parent[i] = -1;	new_split[i] = 0;  need_split[i] = false;
		}
		for (i = 0; i < old_k; i++) //�ҳ���Ҫ���ѵĴ�,�����з���,   �ģ�old_kû����ֵΪkpara->k(0705)
		{
			// if ((needSplit(i) && cluster_acc[i] < 0.99) && kpara->k < maxnum)
			if (cluster_acc[i] < 1 && kpara->k < maxnum)
			{
				need_split[i] = true;
				if (kmean_split1(kpara, &k_prob, i)) split_flg = true;
			}
		}
		n++;
	} while (split_flg == true && kpara->k < maxnum);
	kmean_predict0(kpara, kprob);
	kmean_predict_testdata0(kpara);

	for (i = 0; i < kpara->k; i++)// ���ο����·��ѵĴأ��Ƿ���Ҫ����
	{
		if (new_split[i] == 1)//�·��ѳ����Ĵ�
		{
			if (cluster_acc[i] < 1.0 / model->nr_class)// || coef[i]< avg1)//train_acc || cluster_acc[i] < 0.5)//(double)class_acc[t]/ class_total[t])//(cluster_acc[i] <= old_cluster_acc[parent[i]] && sse[i] >= old_sse[parent[i]])// 
			//if(cluster_acc[i] < avg_cluster_acc)
			{
				if (kmean_delete_centroid(kpara, &k_prob, i))
				{
					i--; restore = true;
				}
			}
		}
	}
	iterative_update2(0);//////////
	return n;
}

int kmean_train_new112(int No, kmean_param * kpara, svm_problem * kprob)  //����split1���µ�Ŀ�꺯������UC
{
	int i, j, dif, tmp_k, n = 0;
	//int cishu=1;//��������
	bool restore, split_flg = true;
	double t1, t2, d, update_del = 100;
	bool *need_split = new bool[maxnum];
	//k_center(&k_prob,svm_get_nr_class(model));	//����ѵ�������б�ǩ�������������ı�ǩ��ʼ��k������
	sum_SSE = sum_ACC = 0; //����ָ��d���ܺ�
	sum_FVP = 0; // ������ָ��
	do
	{
		///////////////////��һ���������ϴεľ�����   //////////////////////////////////////////////////////
		for (i = 0; i < kprob->l; i++)
		{
			old_cluster[i] = cluster[i];////
			old_c[i] = c[i];
		}
		for (i = 0; i < test_data.l; i++)
			old_c1[i] = c1[i];//�ϴ�Ԥ��Ľ��
		///////////////////�ڶ��������в���   //////////////////////////////////////////////////////		
		kmean_predict0(kpara, kprob);//�б�ǩ����������Ԥ��
		kmean_predict_testdata0(kpara);//�ޱ�ǩ����������Ԥ��	
		history_acc[n] = train_acc; history_SSE[n] = total_sse; history_FVP[n] = train_f1;
		sum_ACC += history_acc[n];
		sum_SSE += history_SSE[n];
		sum_FVP += history_FVP[n];
		//////////////���������������Ч�����Ƿ���Ҫ��ԭ//////////////////////////////////////////	
		if (n != 0)
		{
			tmp_k = kpara->k;
			for (i = 0; i < kpara->k; i++)
			{
				copy(kpara->x_c[i], tmp_Center[i]);
				tmp_yc[i] = kpara->y_c[i];
			}
			for (i = 0; i < kprob->l; i++)
			{
				tmp_cluster[i] = cluster[i];////
				tmp_c[i] = c[i];
			}
			restore = false;
			/*for (i = 0; i < kpara->k; i++)// ���ο����·��ѵĴأ��Ƿ���Ҫ����
			{
				if (new_split[i] == 1)//�·��ѳ����Ĵ�
				{
					if (cluster_acc[i] < avg_cluster_acc)// || coef[i]< avg1)//train_acc || cluster_acc[i] < 0.5)//(double)class_acc[t]/ class_total[t])//(cluster_acc[i] <= old_cluster_acc[parent[i]] && sse[i] >= old_sse[parent[i]])// 
					{
						if (kmean_delete_centroid(kpara, &k_prob, i))
						{
							i--; restore = true;
						}
					}
				}
			}*/

			/*split_flg = false;
			for (i = 0; i < kpara->k; i++) // �Դ˿����·��ѵĴ��Ƿ񶼱�������
			{
				if (new_split[i] == 1) split_flg = true;
			}
			if (!split_flg && n != 0)
				break; // ����·��ѵĴض���������ֹͣѭ��
			if (restore == true) // ������·��ѵĴر������������²���
			{
				kmean_predict0(kpara, kprob);
				kmean_predict_testdata0(kpara);
				history_acc[n] = train_acc; history_SSE[n] = total_sse; history_FVP[n] = train_f1;
				sum_ACC += history_acc[n];
				sum_SSE += history_SSE[n];
				sum_FVP += history_FVP[n];

			// �ⲿ����ע��
				history_acc[n + 1] = train_acc; history_SSE[n + 1] = total_sse;
				d=(weight*history_acc[n] / sum_ACC - (1-weight)*history_SSE[n] / sum_SSE) - (weight * history_acc[n + 1] / sum_ACC - (1 - weight)*history_SSE[n + 1] / sum_SSE);
				if(d>=0) //�������ָ�ԭ����
				{
					printf("\n Restore clusters before deletting!\n");
					kpara->k = tmp_k; //��ԭ�Ľ������?
					for (i = 0; i<kpara->k; i++)
					{
						copy(tmp_Center[i], kpara->x_c[i]);
						kpara->y_c[i] = tmp_yc[i];
					}
					for(i=0;i<kprob->l;i++)
					{
					  cluster[i]=tmp_cluster[i];////
					  c[i]=tmp_c[i];
					}
					kmean_predict0(kpara, kprob);// kmean_predict_new(kpara, kprob);
					kmean_predict_testdata0(kpara);
				}
				else
				{
					  sum_ACC = sum_ACC - history_acc[n];////
					  sum_SSE = sum_SSE - history_SSE[n];
					  history_acc[n] = history_acc[n + 1]; history_SSE[n] = history_SSE[n + 1];
					  sum_ACC += history_acc[n];////
					  sum_SSE += history_SSE[n];
				}
				// �ⲿ����ע��
			} // end 	if (restore == true)	 
			printf("\n %lf   %lf", history_acc[n - 1], history_acc[n]); printf("\n %lf   %lf", history_SSE[n - 1], history_SSE[n]); printf("\n %lf   %lf", history_FVP[n - 1], history_FVP[n]);
			printf("\n %lf   %lf", history_acc[n - 1] / sum_ACC, history_acc[n] / sum_ACC); printf("\n %lf   %lf", history_SSE[n - 1] / sum_SSE, history_SSE[n] / sum_SSE); printf("\n %lf   %lf", history_FVP[n - 1] / sum_FVP, history_FVP[n] / sum_FVP);
			//t1 = weight * history_acc[n - 1] / sum_ACC - (1 - weight)*history_SSE[n - 1] / sum_SSE;
			//t2 = weight * history_acc[n] / sum_ACC - (1 - weight)*history_SSE[n] / sum_SSE;
			t1 = weight * history_FVP[n - 1] / sum_FVP - (1 - weight)*history_SSE[n - 1] / sum_SSE;
			t2 = weight * history_FVP[n] / sum_FVP - (1 - weight)*history_SSE[n] / sum_SSE;
			if (t1 >= t2) //�������η��ѣ��ָ�ԭ���� 
			{
				printf("\n Restore former clusters!\n");
				kpara->k = old_k;
				for (i = 0; i < kpara->k; i++)
				{
					copy(Center[i], kpara->x_c[i]);
					kpara->y_c[i] = old_yc[i];
				}
				break;
			}
			*/
		} // end if (n != 0 )//////////////////////////////////////////////////////////////////////////////////
	
		if (train_acc >= 1) break;
		dif = 0;
		for (i = 0; i < test_data.l; i++)	//ͳ������Ԥ��Ĳ���
			if (old_c1[i] != c1[i]) dif++;
		if (n != 0 && dif < 1)
			break;
		//kmean_update_once(kpara, kprob);  //Ӧ��ȫ�����£���������ѵ����������δ���Ѵص�����

		///////////////////���Ĳ��������ϴε����������Ϣ////////////////////////////////////////////////		

		//old_total_sse = total_sse; old_train_acc = train_acc; /////��������ָ��		
		old_k = kpara->k;
		for (j = 0; j < kpara->k; j++)
		{
			copy(kpara->x_c[j], Center[j]);//�������ǰ������		
			old_yc[j] = kpara->y_c[j];	  //�������ǰ�����ı�ǩ		
			old_sse[j] = sse[j];
			old_count[j] = count1[j];//��������ѵ�������ĸ���
			old_count_acc[j] = count_acc[j];  //��������Ԥ��׼ȷ��ѵ�������ĸ���
			old_cluster_acc[j] = cluster_acc[j]; //��������ѵ������Ԥ���׼ȷ��
		}
		//////////////������������////////////////////////
		split_flg = false;
		for (i = 0; i < maxnum; i++) //��ʼ�����дصķ���״̬
		{
			parent[i] = -1;	new_split[i] = 0;  need_split[i] = false;
		}
		for (i = 0; i < old_k; i++) //�ҳ���Ҫ���ѵĴ�,�����з���
		{
			// if ((needSplit(i) && cluster_acc[i] < 0.99) && kpara->k < maxnum)
			if (cluster_acc[i] < 1 && kpara->k < maxnum)
			{
				need_split[i] = true;
				if (kmean_split1(kpara, &k_prob, i)) split_flg = true;
			}
		}
		n++;
		//cishu++;
	} while (split_flg == true && kpara->k < maxnum);
	kmean_predict0(kpara, kprob);
	kmean_predict_testdata0(kpara);
	history_acc[n] = train_acc; history_SSE[n] = total_sse;
	sum_ACC += history_acc[n];////
	sum_SSE += history_SSE[n];
	//iterative_update2(n);//////////
	return n;
}

int kmean_train_new_ori(int No, kmean_param * kpara, svm_problem * kprob)
{
	int i, j, dif, tmp_k, n = 0;
	//int cishu=1;//��������
	bool restore, split_flg = true;
	double t1, t2, d, update_del = 100;
	bool *need_split = new bool[maxnum];
	k_center(&k_prob,svm_get_nr_class(model));	//����ѵ�������б�ǩ�������������ı�ǩ��ʼ��k������
	sum_SSE = sum_ACC = 0; //����ָ��d���ܺ�
	sum_FVP = 0; // ������ָ��
	do
	{
		///////////////////��һ���������ϴεľ�����   //////////////////////////////////////////////////////
		for (i = 0; i < kprob->l; i++)
		{
			old_cluster[i] = cluster[i];////
			old_c[i] = c[i];
		}
		for (i = 0; i < test_data.l; i++)
			old_c1[i] = c1[i];//�ϴ�Ԥ��Ľ��
		///////////////////�ڶ��������в���   //////////////////////////////////////////////////////		
		kmean_predict0(kpara, kprob);//�б�ǩ����������Ԥ��
		kmean_predict_testdata0(kpara);//�ޱ�ǩ����������Ԥ��	
		history_acc[n] = train_acc; history_SSE[n] = total_sse; history_FVP[n] = train_f1;
		sum_ACC += history_acc[n];
		sum_SSE += history_SSE[n];
		sum_FVP += history_FVP[n];
		//////////////���������������Ч�����Ƿ���Ҫ��ԭ//////////////////////////////////////////	
		if (n != 0)
		{
			tmp_k = kpara->k;
			for (i = 0; i < kpara->k; i++)
			{
				copy(kpara->x_c[i], tmp_Center[i]);
				tmp_yc[i] = kpara->y_c[i];
			}
			for (i = 0; i < kprob->l; i++)
			{
				tmp_cluster[i] = cluster[i];////
				tmp_c[i] = c[i];
			}
			restore = false;
			for (i = 0; i < kpara->k; i++)// ���ο����·��ѵĴأ��Ƿ���Ҫ����
			{
				if (new_split[i] == 1)//�·��ѳ����Ĵ�
				{
					if (cluster_acc[i] < avg_cluster_acc)// || coef[i]< avg1)//train_acc || cluster_acc[i] < 0.5)//(double)class_acc[t]/ class_total[t])//(cluster_acc[i] <= old_cluster_acc[parent[i]] && sse[i] >= old_sse[parent[i]])//
					{
						if (kmean_delete_centroid(kpara, &k_prob, i))
						{
							i--; restore = true;
						}
					}
				}
			}

			split_flg = false;
			for (i = 0; i < kpara->k; i++)// �Դ˿����·��ѵĴ��Ƿ񶼱�������
			{
				if (new_split[i] == 1) split_flg = true;
			}
			if (!split_flg && n != 0)
				break; // ����·��ѵĴض���������ֹͣѭ��
			if (restore == true) // ������·��ѵĴر������������²���
			{
				kmean_predict0(kpara, kprob);
				kmean_predict_testdata0(kpara);
				history_acc[n] = train_acc; history_SSE[n] = total_sse; history_FVP[n] = train_f1;
				sum_ACC += history_acc[n];
				sum_SSE += history_SSE[n];
				sum_FVP += history_FVP[n];
				/*history_acc[n + 1] = train_acc; history_SSE[n + 1] = total_sse;
				d=(weight*history_acc[n] / sum_ACC - (1-weight)*history_SSE[n] / sum_SSE) - (weight * history_acc[n + 1] / sum_ACC - (1 - weight)*history_SSE[n + 1] / sum_SSE);
				if(d>=0) //�������ָ�ԭ����
				{
					printf("\n Restore clusters before deletting!\n");
					kpara->k = tmp_k; //��ԭ�Ľ������?
					for (i = 0; i<kpara->k; i++)
					{
						copy(tmp_Center[i], kpara->x_c[i]);
						kpara->y_c[i] = tmp_yc[i];
					}
					for(i=0;i<kprob->l;i++)
					{
					  cluster[i]=tmp_cluster[i];////
					  c[i]=tmp_c[i];
					}
					kmean_predict0(kpara, kprob);// kmean_predict_new(kpara, kprob);
					kmean_predict_testdata0(kpara);
				}
				else
				{
					  sum_ACC = sum_ACC - history_acc[n];////
					  sum_SSE = sum_SSE - history_SSE[n];
					  history_acc[n] = history_acc[n + 1]; history_SSE[n] = history_SSE[n + 1];
					  sum_ACC += history_acc[n];////
					  sum_SSE += history_SSE[n];
				}*/
		} // end 	if (restore == true)	  
		printf("\n %lf   %lf", history_acc[n - 1], history_acc[n]); printf("\n %lf   %lf", history_SSE[n - 1], history_SSE[n]); printf("\n %lf   %lf", history_FVP[n - 1], history_FVP[n]);
		printf("\n %lf   %lf", history_acc[n - 1] / sum_ACC, history_acc[n] / sum_ACC); printf("\n %lf   %lf", history_SSE[n - 1] / sum_SSE, history_SSE[n] / sum_SSE); printf("\n %lf   %lf", history_FVP[n - 1] / sum_FVP, history_FVP[n] / sum_FVP);
		t1 = weight * history_acc[n - 1] / sum_ACC - (1 - weight)*history_SSE[n - 1] / sum_SSE;
		t2 = weight * history_acc[n] / sum_ACC - (1 - weight)*history_SSE[n] / sum_SSE;
		//t1 = weight * history_FVP[n - 1] / sum_FVP - (1 - weight)*history_SSE[n - 1] / sum_SSE;
		//t2 = weight * history_FVP[n] / sum_FVP - (1 - weight)*history_SSE[n] / sum_SSE;
		if (t1 >= t2) //�������η��ѣ��ָ�ԭ����
		{
			printf("\n Restore former clusters!\n");
			kpara->k = old_k;
			for (i = 0; i < kpara->k; i++)
			{
				copy(Center[i], kpara->x_c[i]);
				kpara->y_c[i] = old_yc[i];
			}
			break;
		}

		} // end if (n != 0 )//////////////////////////////////////////////////////////////////////////////////

		if (train_acc >= 1) break;
		dif = 0;
		for (i = 0; i < test_data.l; i++)	//ͳ������Ԥ��Ĳ���
			if (old_c1[i] != c1[i]) dif++;
		if (n != 0 && dif < 1)
			break;
		//kmean_update_once(kpara, kprob);  //Ӧ��ȫ�����£���������ѵ����������δ���Ѵص�����

		///////////////////���Ĳ��������ϴε����������Ϣ////////////////////////////////////////////////		

		//old_total_sse = total_sse; old_train_acc = train_acc; /////��������ָ��		
		old_k = kpara->k;
		for (j = 0; j < kpara->k; j++)
		{
			copy(kpara->x_c[j], Center[j]);//�������ǰ������		
			old_yc[j] = kpara->y_c[j];	  //�������ǰ�����ı�ǩ		
			old_sse[j] = sse[j];
			old_count[j] = count1[j];//��������ѵ�������ĸ���
			old_count_acc[j] = count_acc[j];  //��������Ԥ��׼ȷ��ѵ�������ĸ���
			old_cluster_acc[j] = cluster_acc[j]; //��������ѵ������Ԥ���׼ȷ��
		}
		//////////////������������////////////////////////
		split_flg = false;
		for (i = 0; i < maxnum; i++) //��ʼ�����дصķ���״̬
		{
			parent[i] = -1;	new_split[i] = 0;  need_split[i] = false;
		}
		for (i = 0; i < old_k; i++) //�ҳ���Ҫ���ѵĴ�,�����з���
		{
			if (cluster_acc[i] < 0.99 && kpara->k < maxnum)
			{
				need_split[i] = true;
				if (kmean_split1(kpara, &k_prob, i)) split_flg = true;
			}
		}
		n++;
		//cishu++;
	} while (split_flg == true && kpara->k < maxnum);
	kmean_predict0(kpara, kprob);
	kmean_predict_testdata0(kpara);
	history_acc[n] = train_acc; history_SSE[n] = total_sse;
	sum_ACC += history_acc[n];////
	sum_SSE += history_SSE[n];
	iterative_update2(n);//////////
	return n;
}

/*
	���Ƕ��ڶ�����Ĵأ����ôع�����ֱ��ɾ�������ò��ִؾ���С���ؽ��Ļ�������λ�ڷ��೬ƽ���ϵģ��ɿ���ֱ��ɾ��
										 ���ôؾ���С���ؽ�Զ�����������Ҳ��Զ�Ļ����ôؿ�����samll disjoint�����Բ�ɾ��
	    ����������Ĵأ���ɾ��
*/
void getEachClusterAcc(kmean_param &kpara, svm_problem &kprob); // ����
void deleteNegCluster() // ɾ�����������ٵĸ���� 
{
	int threshold = 2;// prob.l / k_param.k / 2;
	int negFlag, posFlag;
	useClusterNum = k_param.k;
	//for (int i = 0; i < k_param.k; i++) {
	//	cout << sub_count1[i] << endl;
	//}
	getEachClusterAcc(k_param, k_prob);
	for (int i = 0; i < k_param.k; i++) {
		printf("\ncluster[%d] acc is:%f", i, cluster_acc[i]);
		negFlag = 0, posFlag = 0;
		for (int j = 0; j < model->nr_class; j++) {
			if (k_param.y_c[i] == model->label[j] && model->pos[j] == 0) {
				negFlag = 1;
				break;
			}
		}
		if ((negFlag && count1[i] <= threshold && cluster_acc[i] < 0.9) || cluster_acc[i] <= 0) {
			k_param.use[i] = 0; // ����i����ɾ��
			useClusterNum--; // ʵ��ʹ�õĴ�����--
			for (int k = 0; k < prob.l; k++) { // ����ÿ��ѵ�������������ڸôص�����use��Ϊ0
				if (cluster[k] == i) {
					prob.use[k] = 0;
					k_prob.use[k] = 0;
				}	
			}
			printf("\ncluster:%d is deleted", i);
		}
		/*if (count1[i] <= threshold) {
			k_param.use[i] = 0; // ����i����ɾ��
			useClusterNum--; // ʵ��ʹ�õĴ�����--
			for (int k = 0; k < prob.l; k++) { // ����ÿ��ѵ�������������ڸôص�����use��Ϊ0
				if (cluster[k] == i) {
					prob.use[k] = 0;
					k_prob.use[k] = 0;
				}
			}
			printf("\ncluster:%d is deleted", i);
		}*/

		// ���ԣ�ɾ��������������Ϊ1�������ࣿ����
		/*for (int j = 0; j < model->nr_class; j++) {
			if (k_param.y_c[i] == model->label[j] && model->pos[j] == 1) {
				posFlag = 1;
				break;
			}
		}
		if (posFlag && sub_count1[i] <= 1) {
			k_param.use[i] = 0; // ����i����ɾ��
			useClusterNum--; // ʵ��ʹ�õĴ�����--
			for (int k = 0; k < prob.l; k++) { // ����ÿ��ѵ�������������ڸôص�����use��Ϊ0
				if (cluster[k] == i) {
					prob.use[k] = 0;
					k_prob.use[k] = 0;
				}
			}
			printf("\ncluster:%d is deleted", i);
		}*/
	}
}

void set_noise0(svm_problem *prob_noise,svm_problem *k_prob_noise)
{
	int a=0,b,c,d=0,e;
	int *flag_noise = Malloc(int,prob_noise->l);
	int num_class=1;
	double *label=Malloc(double,prob_noise->l);
	double label_0;
	for(int a=0;a<prob_noise->l;a++)
		label[a]=prob_noise->r_y[a];
	//label����
	for(int b=0;b<prob_noise->l-1;b++)
		for(int c=b+1;c<prob_noise->l;c++)
			if(label[b]>label[c])
			{
				label_0 = label[c];
				label[c] = label[b];
				label[b] = label_0;
			}
	//������num_class
	for(int d=0;d<prob_noise->l-1;d++)
			if(label[d]!=label[d+1])
				num_class++;
	printf("��������%d",num_class);
	//�õ���ǩ��Ӧ������count_label[]
	int *count_label = Malloc(int,num_class);
	count_label[0]=label[0];
	int num=0;
	for(int e=0;e<prob_noise->l-1;e++)
		if(label[e]!=label[e+1])
		{
			num++;
			count_label[num]=label[e+1];
		}
	for(int f=0;f<num_class;f++)
		printf("\n%d",count_label[f]);

	for(b=0;b<prob_noise->l;b++)
		flag_noise[b] = 0;//��ʼ������flag.
	//����������flag
	while(a<(int)(noise_per*prob_noise->l))//�����ĸ���
	{
		c = rand()%(prob_noise->l - 0) + 0;//���ѡ������
		if(flag_noise[c]!=1)
		{
			flag_noise[c]=1;
			a++;
		}
	}
	//��������������������ǩ������Ĵ��벻�����ѡ����һ������������ѡ����һ���뵱ǰ������ǩ��һ�µ��������Դ���Ϊ���������ı�ǩ
	for(d=0;d<prob_noise->l;d++)//����ѵ�����е���������
		if(flag_noise[d]==1)//�жϵ�ǰ�����Ƿ�Ϊ����
		{
			//e = rand()%(prob->l - 0) + 0;//���������һ��������Ӧ��ѵ�����е�ĳһ������
			//while(prob->r_y[e] == prob->r_y[d])//�����ѡ������������ʵ��ǩ���ڵ�ǰ������������ʵ��ǩ����ô����ʹ��rand������������һ������ֱ�����ѡ������������ʵ��ǩ����������������Ӧ����ʵ��ǩ
				//e = rand()%(prob->l - 0) + 0;
			//prob->y[d] = prob->r_y[e];//�������������Ϻ��Լ���ǩ��һ�µı�ǩ����Ϊ����
			srand(time(0));
			e=rand()%(num_class - 0) + 0;//�����������������
			while(count_label[e] == prob_noise->r_y[d])
				e = rand()%(num_class - 0) + 0;
			prob_noise->y[d] =count_label[e];
			//printf("%d",e);
		}
	//��prob���Ƶ�k_prob��
		for(int aaa=0;aaa<prob_noise->l;aaa++)
			k_prob_noise->y[aaa] = prob_noise->y[aaa];

}
void set_noise(svm_problem *prob_noise,svm_problem *k_prob_noise)
{
	int a=0,b,c,d=0,e;
	int num_class=1;
	noise_per = 0.3;
	flag_noise = Malloc(bool,prob_noise->l);	
	double *label=Malloc(double,prob_noise->l);
	double label_0;
	for(int a=0;a<prob_noise->l;a++)
		label[a]=prob_noise->r_y[a];
	//label����
	for(int b=0;b<prob_noise->l-1;b++)
		for(int c=b+1;c<prob_noise->l;c++)
			if(label[b]>label[c])
			{
				label_0 = label[c];
				label[c] = label[b];
				label[b] = label_0;
			}
	//������num_class
	for(int d=0;d<prob_noise->l-1;d++)
			if(label[d]!=label[d+1])
				num_class++;
	printf("��������%d",num_class);
	//�õ���ǩ��Ӧ������count_label[]
	int *count_label = Malloc(int,num_class);
	count_label[0]=label[0];
	int num=0;
	for(int e=0;e<prob_noise->l-1;e++)
		if(label[e]!=label[e+1])
		{
			num++;
			count_label[num]=label[e+1];
		}
	for(int f=0;f<num_class;f++)
		printf("\n%d",count_label[f]);

	for(b=0;b<prob_noise->l;b++)
		flag_noise[b] = false;//��ʼ������flag.
	//����������flag
	int aa=0;
	while(aa<(int)(noise_per*prob_noise->l))//�����ĸ���
	{
		c = rand()%(prob_noise->l - 0) + 0;//���ѡ������
		if(flag_noise[c]!=true)
		{
			flag_noise[c]=true;
			aa++;
		}
	}
	//��������������������ǩ������Ĵ��벻�����ѡ����һ������������ѡ����һ���뵱ǰ������ǩ��һ�µ��������Դ���Ϊ���������ı�ǩ
	for(d=0;d<prob_noise->l;d++)//����ѵ�����е���������
		if(flag_noise[d]==true)//�жϵ�ǰ�����Ƿ�Ϊ����
		{
			//e = rand()%(prob->l - 0) + 0;//���������һ��������Ӧ��ѵ�����е�ĳһ������
			//while(prob->r_y[e] == prob->r_y[d])//�����ѡ������������ʵ��ǩ���ڵ�ǰ������������ʵ��ǩ����ô����ʹ��rand������������һ������ֱ�����ѡ������������ʵ��ǩ����������������Ӧ����ʵ��ǩ
				//e = rand()%(prob->l - 0) + 0;
			//prob->y[d] = prob->r_y[e];//�������������Ϻ��Լ���ǩ��һ�µı�ǩ����Ϊ����
			srand(time(0));
			e=rand()%(num_class - 0) + 0;//�����������������
			while(count_label[e] == prob_noise->r_y[d])
				e = rand()%(num_class - 0) + 0;
			prob_noise->y[d] =count_label[e];
			//printf("%d",e);
		}
	//��prob���Ƶ�k_prob��
		for(int aaa=0;aaa<prob_noise->l;aaa++)
			k_prob_noise->y[aaa] = prob_noise->y[aaa];

}

void allocateSpace()
{
	int i, max_n = 0;
	int n1 = 30, n2 = 30;
	initial_train_size = k_prob.l;
	c = new double[k_prob.l];//ѵ��������Ԥ���ǩ
	old_c = new double[k_prob.l];//ѵ��������Ԥ���ǩ
	u = new double[k_prob.l];//ѵ��������Ԥ�����
	cluster = new int[k_prob.l];//ѵ�����������Ĵ����
	old_cluster = new int[k_prob.l];//ѵ�����������Ĵ����
	tmp_cluster = new int[k_prob.l];////
	tmp_c = new int[k_prob.l];
	//coef_train=new double[k_prob.l]; //ѵ������������ϵ��
	count1 = new int[maxnum];
	count_acc = new int[maxnum];
	old_count = new int[maxnum];
	old_count_acc = new int[maxnum];

	k_param.x_c = Malloc(struct svm_node*, maxnum);
	Center = Malloc(struct svm_node *, maxnum); //�ɵ�����
	old_yc = new double[maxnum]; //�ɵ�����
	for (i = 0; i < maxnum; i++)
	{
		k_param.x_c[i] = Malloc(struct svm_node, max_index + 1);
		Center[i] = Malloc(struct svm_node, max_index + 1);
	}
	k_param.k_c = Malloc(int, maxnum);
	k_param.noise = Malloc(int, maxnum);
	k_param.y_c = Malloc(double, maxnum);
	k_param.diameter = Malloc(double, maxnum);
	k_param.w = Malloc(double, maxnum);
	k_param.subclass_y = Malloc(double, maxnum);
	k_param.use = Malloc(int, maxnum);
	k_param.size = Malloc(int, maxnum);
	k_param.pos = Malloc(int, maxnum);

	old_c1 = new double[test_data.l];
	c1 = new double[test_data.l];//����������Ԥ���ǩ
	u1 = new double[test_data.l];//����������Ԥ�����
	u3 = new double[test_data.l]; //kmeans�ڲ������������Ԥ�����
	svm_pro = new double[test_data.l];
	svm_pred = new double[test_data.l];
	cluster1 = new int[test_data.l];//�������������������Ĵ����
	select_flg = new bool[prob.l];
	//SK_test_data_y = new double[test_data.l];
	//last_test_data_y = new double[test_data.l];

	for (i = 0; i < maxnum; i++)
		tmp_Center[i] = Malloc(struct svm_node, max_index + 1);

	kmeans_confidence = Malloc(double *, test_data.l);
	for (i = 0; i < test_data.l; i++)
		kmeans_confidence[i] = Malloc(double, maxnum);
	svm_confidence = Malloc(double *, test_data.l);
	for (i = 0; i < test_data.l; i++)
		svm_confidence[i] = Malloc(double, maxnum);

	svm_ori_confidence = Malloc(double *, test_data.l);
	for (i = 0; i < test_data.l; i++)
		svm_ori_confidence[i] = Malloc(double, maxnum);

	// Ϊ֮�󱣴��������ݽṹ��ǰ���ٴ洢�ռ�
	sub_prob.l = prob.l * 5 + test_data.l + test_data.l;
	sub_prob.y = Malloc(double, sub_prob.l);
	sub_prob.r_y = Malloc(double, sub_prob.l);
	sub_prob.pos = Malloc(int, sub_prob.l);
	sub_prob.use = Malloc(int, sub_prob.l);
	sub_prob.x = Malloc(struct svm_node *, sub_prob.l);
	for (i = 0; i < sub_prob.l; i++)
	{
		sub_prob.x[i] = Malloc(struct svm_node, max_index + 1);
		//Center[i] = Malloc(struct svm_node, max_index + 1);
	}
	sub_c = new double[sub_prob.l];//ѵ��������Ԥ���ǩ
	sub_cluster = new int[sub_prob.l];//ѵ�����������Ĵ����
	sub_count1 = new int[maxnum];

	// co-trainingʹ�ñ���
	kmeans_noise_flag = new bool[sub_prob.l];
	initial_train_size = prob.l;

	posSubLabel = Malloc(double, maxnum);
}

void allocateSpace2()
{
	int i;
	sum_for_train = Malloc(struct svm_node, max_index + 1);
	result = Malloc(struct svm_node, max_index + 1);
	k_param.k = model->nr_class;
	sum = Malloc(struct svm_node*, model->nr_class);
	for (i = 0; i < model->nr_class; i++)
		sum[i] = Malloc(struct svm_node, max_index + 1);
	sum_update = Malloc(struct svm_node *, maxnum);//�������ĵ��ۼӺͣ����ڸ������ĵ�
	for (i = 0; i < maxnum; i++)
		sum_update[i] = Malloc(struct svm_node, max_index + 1);
	predict_class_num = new int[model->nr_class];
	test_class_num = new int[model->nr_class];
	train_class_num = new int[model->nr_class];
	num = new int[model->nr_class];
}

void initParamDefault()
{
	param.svm_type = C_SVC;
	param.kernel_type = LINEAR;// LINEAR;//RBF;//
	param.degree = 3;
	param.gamma = 0;//tGamma;	// 1/num_features  Ĭ��Ϊ0
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;//tC; // Ĭ��Ϊ 1
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;//////////////////
	param.nr_weight = 0; // Ĭ�ϲ���
	param.weight_label = NULL; // Ĭ�ϲ���
	param.weight = NULL; // Ĭ�ϲ���
	cross_validation = 0;//�ǽ�����֤
	//cross_validation = 1;//������֤
	//nr_fold = 5;//������֤
	void(*print_func)(const char*) = NULL;	// default printing to stdout
	svm_set_print_string_function(print_func);
	predict_probability = 1; // �Ƿ�ʹ�ø���Ԥ��
}

void initParam()
{
	param.svm_type = C_SVC;
	param.kernel_type = RBF;// LINEAR;//RBF;//
	param.degree = 3;
	param.gamma = tGamma;	// 1/num_features  Ĭ��Ϊ0
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = tC; // Ĭ��Ϊ 1
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;//////////////////
	// param.nr_weight = 0; // Ĭ�ϲ���
	param.nr_weight = 2;
	// param.weight_label = NULL; // Ĭ�ϲ���
	param.weight_label = Malloc(int, 2);
	param.weight_label[0] = 1; param.weight_label[1] = -1;
	// param.weight = NULL; // Ĭ�ϲ���
	param.weight = Malloc(double, 2);
	param.weight[0] = tweight0; param.weight[1] = tweight1;
	cross_validation = 0;//�ǽ�����֤
	//cross_validation = 1;//������֤
	//nr_fold = 5;//������֤
	void(*print_func)(const char*) = NULL;	// default printing to stdout
	svm_set_print_string_function(print_func);
	predict_probability = 1; // �Ƿ�ʹ�ø���Ԥ��
}

void initParamAfterSplit(int n)
{
	param.svm_type = C_SVC;
	param.kernel_type = RBF;// LINEAR;//RBF;//
	param.degree = 3;
	param.gamma = tGamma;	// 1/num_features  Ĭ��Ϊ0
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = tC; // Ĭ��Ϊ 1
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;//////////////////
	// param.nr_weight = 0; // Ĭ�ϲ���
	param.nr_weight = n;
	// param.weight_label = NULL; // Ĭ�ϲ���
	param.weight_label = Malloc(int, n);
	param.weight = Malloc(double, n);
	// param.weight[0] = tweight0; param.weight[1] = tweight1;
	int id = 0;
	for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
		for (int i = 0; i < it->second.size(); i++) {
			if (it->first == 1) {
				param.weight_label[id] = it->second[i];
				param.weight[id] = tweight0;
				id++;
			}
			else {
				param.weight_label[id] = it->second[i];
				param.weight[id] = tweight1;
				id++;
			}
		}
	}
	// param.weight = NULL; // Ĭ�ϲ���
	cross_validation = 0;//�ǽ�����֤
	//cross_validation = 1;//������֤
	//nr_fold = 5;//������֤
	void(*print_func)(const char*) = NULL;	// default printing to stdout
	svm_set_print_string_function(print_func);
	predict_probability = 1; // �Ƿ�ʹ�ø���Ԥ��
}

void initParamAfterSplit2(int n)
{
	param.svm_type = C_SVC;
	param.kernel_type = RBF;// LINEAR;//RBF;//
	param.degree = 3;
	param.gamma = tGamma;	// 1/num_features  Ĭ��Ϊ0
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = tC; // Ĭ��Ϊ 1
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;//////////////////
	// param.nr_weight = 0; // Ĭ�ϲ���
	param.nr_weight = n;
	// param.weight_label = NULL; // Ĭ�ϲ���
	param.weight_label = Malloc(int, n);
	param.weight = Malloc(double, n);
	// param.weight[0] = tweight0; param.weight[1] = tweight1;
	int posNum = 0, negNum = 0;
	for (int i = 0; i < k_param.k; i++) {
		if (k_param.use[i] == 0) continue;
		if (k_param.y_c[i] == 1)
			posNum += sub_count1[i];
		if (k_param.y_c[i] == -1)
			negNum += sub_count1[i];
	}
	double IR = (negNum * 1.0 / posNum) * 10;
	int id = 0;
	for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
		for (int i = 0; i < it->second.size(); i++) {
			if (it->first == 1) {
				param.weight_label[id] = it->second[i];
				param.weight[id] = IR;
				id++;
			}
			else {
				param.weight_label[id] = it->second[i];
				param.weight[id] = 1;
				id++;
			}
			printf("\n��ǩΪ%d������Ȩ��Ϊ%f", param.weight_label[id-1], param.weight[id-1]);
		}
	}
	// param.weight = NULL; // Ĭ�ϲ���
	cross_validation = 0;//�ǽ�����֤
	//cross_validation = 1;//������֤
	//nr_fold = 5;//������֤
	void(*print_func)(const char*) = NULL;	// default printing to stdout
	svm_set_print_string_function(print_func);
	predict_probability = 1; // �Ƿ�ʹ�ø���Ԥ��
	// ��ʼ��������ֵZ��ֵ
	Z = (negNum - posNum) * 1.0 / (negNum + posNum + 2); //(negNum - posNum) * 1.0 / (negNum + posNum + (negNum + posNum))
	afterChangeZ = Z;
	printf("\n������ֵZ�Ľ���ǣ�%f(%d, %d)\n", Z, negNum, posNum);
}

void freeMemery()
{
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	free(k_prob.y);
	free(k_prob.x);
	free(x_space0);
	//Sleep(1000);
	free(line0);
}

void final_predict()
{
	//double * prob_kmeans = new double[model0->nr_class];
	//double * prob_svm = new double[model0->nr_class];
	double *final_result = new double[test_data.l];
	double max_p = -100;
	int max_index = -1, acc = 0;
	for (int i = 0; i < test_data.l; i++)
	{
		svm_predict_probability(model, test_data.x[i], svm_confidence[i]);
		max_p = svm_confidence[i][0];
		max_index = 0;
		for (int t = 1; t < model->nr_class; t++)
		{
			if (max_p < svm_confidence[i][t])
			{
				max_p = svm_confidence[i][t];
				max_index = t;
			}
		}
		test_data.y[i] = model->label[max_index];
		printf("  %d", (int)test_data.y[i]);
		if ((int)c1[i] == (int)test_data.y[i])
			final_result[i] = test_data.y[i];
		else
		{
			max_p = -100; max_index = -1;
			for (int t = 0; t < model->nr_class; t++)
			{
				if (max_p < kmeans_confidence[i][t] + svm_confidence[i][t])
				{
					max_p = kmeans_confidence[i][t] + svm_confidence[i][t];
					max_index = t;
				}
			}
			final_result[i] = model->label[max_index];
		}
		//printf("  %d", (int)final_result[i]);
		// ת����ǩ
		bool flag = false;
		for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
			for (int k = 0; k < it->second.size(); k++) {
				if (it->second[k] == final_result[i]) {
					// cout << final_result[i] << " " << it->first << " " << test_data.r_y[i] << endl;
					final_result[i] = it->first;
					flag = true;
					break;
				}		
			}	
			if (flag) break;
		}
		//printf("  %d", (int)final_result[i]);
		//printf("  %d\n", (int)test_data.r_y[i]);
		if ((int)final_result[i] == (int)test_data.r_y[i]) 	acc++;
	}
	printf("\n\nFinal Acc:%f (%d/%d)\n", (double)acc / test_data.l, acc, test_data.l);
	double final_acc = (double)acc / test_data.l;
}

double getSVMLabelProp(int flag); //// ��ȡSVM�ڲ���������������ǩ�Ϸֱ��Ԥ�����
double getSshKmeansPreLabelProp(int flag); // ��ȡ��ල�����ڲ���������������ǩ�Ϸֱ��Ԥ�����


void final_predict1()//���������ʣ�������С���Ľ�Ϸ�ʽ
{
	double *final_result = new double[test_data.l];
	double max_p = -100;
	int max_index = -1, acc = 0;
	for (int i = 0; i < test_data.l; i++)
	{
		svm_predict_probability(model, test_data.x[i], svm_confidence[i]);
		max_p = svm_confidence[i][0];
		max_index = 0;
		for (int t = 1; t < model->nr_class; t++)
		{
			if (max_p < svm_confidence[i][t])
			{
				max_p = svm_confidence[i][t];
				max_index = t;
			}
		}
		test_data.y[i] = getOriLabel(model->label[max_index]); // ת��Ϊԭʼ���ǩ
		printf("  %d", (int)test_data.y[i]);
		if ((int)c1[i] == (int)test_data.y[i])
			final_result[i] = test_data.y[i];
		else if (c1[i] == 1 && test_data.y[i] == -1)
			final_result[i] = c1[i];
		else
			final_result[i] = test_data.y[i];

		if ((int)final_result[i] == (int)test_data.r_y[i]) 	acc++;
		test_data.y[i] = final_result[i];
	}
	printf("\n\nFinal Acc:%f (%d/%d)\n", (double)acc / test_data.l, acc, test_data.l);
	double final_acc = (double)acc / test_data.l;
	SVM_ACC = final_acc;

	mergeSubPro(); // ������Ԥ�������ӣ���ø���Ԥ�����
	for (int i = 0; i < test_data.l; i++) {
		test_data.r_y_p[i] = svm_ori_confidence[i][0];
	}
}

void final_predict2() //���ݸ��ʣ���һ�����ȽϵĽ������Ԥ��
{
	double *final_result = new double[test_data.l];
	double predict_label_train, max_p = -100;
	int max_index = -1, acc = 0;
	int acc_kmeans = 0, acc_svm = 0;
	double svm_del = 0, new_weight = 0.3;
	/*for (int j = 0; j < k_prob.l; j++) 		//����ѵ������F1�������Ȩ��,�ⲿ����Ҫ�ģ�KmeansӦ�ü���ش���ǰ��F1
	{
		if ((int)c[j] == (int)prob.y[j])
			acc_kmeans++;
		predict_label_train = svm_predict_train(model, prob.x[j]);//ͳ�Ƶ�ǰ�������ڵ���	
		if ((int)predict_label_train == (int)k_prob.y[j])
			acc_svm++;
	}
	new_weight = (double)acc_kmeans / (acc_kmeans + acc_svm);*/

	for (int i = 0; i < test_data.l; i++)
	{
		svm_predict_probability(model, test_data.x[i], svm_confidence[i]);
		max_p = svm_confidence[i][0];
		max_index = 0;
		for (int t = 1; t < model->nr_class; t++)
		{
			if (max_p < svm_confidence[i][t])
			{
				max_p = svm_confidence[i][t];
				max_index = t;
			}
		}
		test_data.p[i] = max_p;
		svm_del += max_p;
		test_data.y[i] = getOriLabel(model->label[max_index]);
		// printf("  %d", (int)test_data.y[i]);
	}

	// ���test_data.p[i]תΪ���游��ĸ���    svm_del���Ƿ�Ҫ���ۼ���������ͬ���������ĸ��ʣ���
	mergeSubPro();
	svm_del = 0;
	for (int i = 0; i < test_data.l; i++) {
		int id = test_data.y[i] == 1 ? 0 : 1;
		test_data.p[i] = svm_ori_confidence[i][id];
		test_data.r_y_p[i] = svm_ori_confidence[i][0]; // ���ڼ���AUC
		svm_del += test_data.p[i];
	}

	for (int i = 0; i < test_data.l; i++)
	{
		if ((int)c1[i] == (int)test_data.y[i])
			final_result[i] = test_data.y[i];
		else if (new_weight*u1[i] / kmeans_del > (1 - new_weight)*test_data.p[i] / svm_del) // �����u1��ûȷ����
			final_result[i] = c1[i];
		else
			final_result[i] = test_data.y[i];

		if ((int)final_result[i] == (int)test_data.r_y[i]) 	acc++;
		test_data.y[i] = final_result[i]; 
		printf("  %d", (int)test_data.y[i]);
	}
	printf("\n\nFinal Acc:%f (%d/%d)\n", (double)acc / test_data.l, acc, test_data.l);
	double final_acc = (double)acc / test_data.l;
	SVM_ACC = final_acc;
}

void final_predict3() //���ݸ��ʽ�ϣ���һ�����Ľ������Ԥ��
{
	double *final_result = new double[test_data.l];
	double max_p = -100;
	int i, j, t, max_index = -1, acc = 0;
	double predict_label_train, svm_del = 0;
	for (i = 0; i < test_data.l; i++)
	{
		svm_predict_probability(model, test_data.x[i], svm_confidence[i]);
		max_p = svm_confidence[i][0];
		max_index = 0;
		for (t = 1; t < model->nr_class; t++)
		{
			if (max_p < svm_confidence[i][t])
			{
				max_p = svm_confidence[i][t];
				max_index = t;
			}
		}
		test_data.p[i] = max_p;
		svm_del += max_p;
		test_data.y[i] = getOriLabel(model->label[max_index]);
		//printf("  %d", (int)test_data.y[i]);
	}

	double svmTestPosPro = getSVMLabelProp(0); // ����SVM��������Ԥ�����
	double svmTestNegPro = 1 - svmTestPosPro;
	double kmeansTestPosPro = getSshKmeansPreLabelProp(0); // ����SSHKMEANS��������Ԥ�����
	double kmeansTestNegPro = 1 - kmeansTestPosPro;
	double svmErrorIR = fabs(trainPosPro - svmTestPosPro) + fabs(trainNegPro - svmTestNegPro); // SVMԤ�����
	double kmeansErrorIR = fabs(trainPosPro - kmeansTestPosPro) + fabs(trainNegPro - kmeansTestNegPro); // kmeansԤ�����
	// kmeansErrorIR *= kmeansErrorIR;
	// svmErrorIR *= svmErrorIR;
	new_weight = 1 - (kmeansErrorIR / (kmeansErrorIR + svmErrorIR));

	// ���test_data.p[i]תΪ���游��ĸ���
	mergeSubPro();
	for (int i = 0; i < test_data.l; i++) {
		int id = test_data.y[i] == 1 ? 0 : 1;
		test_data.p[i] = svm_ori_confidence[i][id];
		test_data.r_y_p[i] = svm_ori_confidence[i][0]; // ���ڼ���AUC
	}

	/*for (i = 0; i < test_data.l; i++) {
		for (j = 0; j < 2; j++) {
			if (getOriLabel(model->label[j]) == 1) {
				test_data.r_y_p[i] = (1 - new_weight)*svm_ori_confidence[i][t] + new_weight * kmeans_confidence[i][t];
			}
		}
	}*/
	

	for (i = 0; i < test_data.l; i++)
	{
		if ((int)c1[i] == (int)test_data.y[i])
			final_result[i] = test_data.y[i];
		else
		{
			max_p = 0; max_index = -1;
			int nrClass = 2;
			for (t = 0; t < nrClass; t++)
			{
				if (max_p < (1 - new_weight)*svm_ori_confidence[i][t] + new_weight * kmeans_confidence[i][t])
				{
					max_p = (1 - new_weight)*svm_ori_confidence[i][t] + new_weight * kmeans_confidence[i][t];
					max_index = t;
				}
			}

			final_result[i] = getOriLabel(model->label[max_index]);
			if (test_data.y[i] != final_result[i])
				printf("��������%dԤ���ǩ�б仯\n", i);
			test_data.y[i] = final_result[i];
			if (test_data.y[i] == 1)
				test_data.r_y_p[i] = (1 - new_weight)*svm_ori_confidence[i][t] + new_weight * kmeans_confidence[i][t];
		}

		if ((int)final_result[i] == (int)test_data.r_y[i]) 	acc++;
		else
			printf("��%d����������ǩ��%.0f\n", i, test_data.r_y[i]);
	}
	printf("\n\nFinal Acc:%f (%d/%d)\n", (double)acc / test_data.l, acc, test_data.l);
	double final_acc = (double)acc / test_data.l;
	SVM_ACC = final_acc;
}


void predict1(char *argv1, char *argv2)
{
	int j, ii, correct = 0;
	int total = 0;
	int max = 0;
	int svm_type, nr_class;
	double max_p, sub_max_p, error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	double *prob_estimates = NULL;
	double predict_label;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
	//x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));///////////////////////////
	FILE *output = fopen(argv1, "w");
	if (output == NULL)
	{
		fprintf(stderr, "can't open output file %s\n", argv1);
		exit(1);
	}
	if ((model0 = svm_load_model(argv2)) == 0)
	{
		fprintf(stderr, "can't open model file %s\n", argv2);
		exit(1);
	}

	svm_type = svm_get_svm_type(model0);
	nr_class = svm_get_nr_class(model0);

	if (predict_probability)
	{
		if (svm_check_probability_model(model0) == 0)
		{
			fprintf(stderr, "Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if (svm_check_probability_model(model0) != 0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	if (predict_probability)
	{
		if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n", svm_get_svr_probability(model0));
		else
		{
			int *labels = (int *)malloc(nr_class * sizeof(int));
			svm_get_labels(model0, labels);
			prob_estimates = (double *)malloc(nr_class * sizeof(double));
			fprintf(output, "labels");
			for (j = 0; j < nr_class; j++)
				fprintf(output, " %d", labels[j]);
			fprintf(output, "\n");
			free(labels);
		}
	}
	u2 = (double *)malloc(test_data.l * sizeof(double));
	for (ii = 0; ii < test_data.l; ii++)
	{
		//while(readline(input) != NULL)
		//{
		if (predict_probability && (svm_type == C_SVC || svm_type == NU_SVC))
		{
			predict_label = svm_predict_probability(model0, test_data.x[ii], prob_estimates);

			// ת����ǩ
			int flag = 0;
			for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
				for (int k = 0; k < it->second.size(); k++) {
					if (it->second[k] == predict_label) {
						// cout << predict_label << " " << it->first << " " << test_data.r_y[ii] << endl;
						predict_label = it->first;
						flag = true;
						break;
					}
				}
				if (flag) break;
			}
			//cout << predict_label << " " << test_data.r_y[ii] << endl;

			fprintf(output, "%g", predict_label);
			max_p = 0;/////////////////////
			sub_max_p = 0;
			for (j = 0; j < nr_class; j++)
			{
				fprintf(output, " %g", prob_estimates[j]);
				if (max_p < prob_estimates[j])
				{
					max_p = prob_estimates[j];
					max = j;//�ҵ����ĸ���
				}
				//����ÿ��������u2[i]����max_p-sub_max_p,�����δ�
				svm_confidence[ii][j] = prob_estimates[j];
			}
			for (j = 0; j < nr_class; j++)
			{
				if ((sub_max_p < prob_estimates[j]) && (j != max))
					sub_max_p = prob_estimates[j];//�ҵ��δ�ĸ���
				//����ÿ��������u2[i]����max_p-sub_max_p,�����δ�
			}
			test_data.p[ii] = max_p;/////////////////////// ����Ԥ�����ĸ���
			u2[ii] = max_p - sub_max_p;
			fprintf(output, "\n");
		}
		else
		{
			predict_label = svm_predict_compare(model0, test_data.x[ii]);

			// ת����ǩ
			bool flag = false;
			for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
				for (int k = 0; k < it->second.size(); k++) {
					if (it->second[k] == predict_label) {
						// cout << predict_label << " " << it->first << " " << test_data.r_y[ii] << endl;
						predict_label = it->first;
						flag = true;
						break;
					}
				}
				if (flag) break;
			}

			fprintf(output, "%g\n", predict_label);
		}
		test_data.y[ii] = predict_label;

		if (predict_label == test_data.r_y[ii])
			++correct;
		error += (predict_label - test_data.r_y[ii])*(predict_label - test_data.r_y[ii]);
		sump += predict_label;
		sumt += test_data.r_y[ii];
		sumpp += predict_label * predict_label;
		sumtt += test_data.r_y[ii] * test_data.r_y[ii];
		sumpt += predict_label * test_data.r_y[ii];
		++total;
	}

	if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n", error / total);
		printf("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt - sump * sumt)*(total*sumpt - sump * sumt)) /
			((total*sumpp - sump * sump)*(total*sumtt - sumt * sumt))
		);
	}
	else
		printf("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct / total * 100, correct, total);
	if (predict_probability)
		free(prob_estimates);

	fclose(output);
}

void predict2(char *argv1, char *argv2)
{
	int j, ii, correct = 0;
	int total = 0;
	int max = 0;
	int svm_type, nr_class;
	double max_p, sub_max_p, error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	double *prob_estimates = NULL;
	double predict_label;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
	//x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));///////////////////////////
	FILE *output = fopen(argv1, "w");
	if (output == NULL)
	{
		fprintf(stderr, "can't open output file %s\n", argv1);
		exit(1);
	}
	if ((model0 = svm_load_model(argv2)) == 0)
	{
		fprintf(stderr, "can't open model file %s\n", argv2);
		exit(1);
	}

	svm_type = svm_get_svm_type(model0);
	nr_class = svm_get_nr_class(model0);

	vector<double> posLabel; // �洢С����Ӧ���ӱ�ǩ
	for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
		if (it->first == 1) {
			for (j = 0; j < it->second.size(); j++) {
				posLabel.push_back(it->second[j]);
			}
		}
	}

	if (predict_probability)
	{
		if (svm_check_probability_model(model0) == 0)
		{
			fprintf(stderr, "Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if (svm_check_probability_model(model0) != 0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	if (predict_probability)
	{
		if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n", svm_get_svr_probability(model0));
		else
		{
			int *labels = (int *)malloc(nr_class * sizeof(int));
			svm_get_labels(model0, labels);
			prob_estimates = (double *)malloc(nr_class * sizeof(double));
			fprintf(output, "labels");
			for (j = 0; j < nr_class; j++)
				fprintf(output, " %d", labels[j]);
			fprintf(output, "\n");
			free(labels);
		}
	}
	u2 = (double *)malloc(test_data.l * sizeof(double));
	for (ii = 0; ii < test_data.l; ii++)
	{
		//while(readline(input) != NULL)
		//{
		if (!proCombination && predict_probability && (svm_type == C_SVC || svm_type == NU_SVC))
		{
			predict_label = svm_predict_probability(model0, test_data.x[ii], prob_estimates);
			fprintf(output, "%g", predict_label);
			max_p = 0;/////////////////////
			sub_max_p = 0;
			for (j = 0; j < nr_class; j++)
			{
				fprintf(output, " %g", prob_estimates[j]);
				if (max_p < prob_estimates[j])
				{
					max_p = prob_estimates[j];
					max = j;//�ҵ����ĸ���
				}
				//����ÿ��������u2[i]����max_p-sub_max_p,�����δ�
				svm_confidence[ii][j] = prob_estimates[j];
			}
			for (j = 0; j < nr_class; j++)
			{
				if ((sub_max_p < prob_estimates[j]) && (j != max))
					sub_max_p = prob_estimates[j];//�ҵ��δ�ĸ���
				//����ÿ��������u2[i]����max_p-sub_max_p,�����δ�
			}

			// ��ò���������С��������϶�Ӧ��Ԥ�����
			for (j = 0; j < nr_class; j++) {
				for (int k = 0; k < posLabel.size(); k++) {
					if (model->label[j] == (int)posLabel[k])
						test_data.r_y_p[ii] += prob_estimates[j];
				}
			}
			svm_pro[ii] = test_data.r_y_p[ii]; // ���ڼ�����ֽ���SVM��AUC
			// printf("---%.0f %lf\n", test_data.r_y[ii], test_data.r_y_p[ii]);
			//printf("%.0f  ", getOriLabel(predict_label));

			test_data.p[ii] = max_p;/////////////////////// ����Ԥ�����ĸ���
			u2[ii] = max_p - sub_max_p;
			fprintf(output, "\n");
		}
		else if (proCombination && predict_probability && (svm_type == C_SVC || svm_type == NU_SVC)) { // ����ֱ�ӽ�Ԥ����ʼ�������ֱ�ӵõ�����ǩ�ĸ���			
			predict_label = svm_predict_probability(model0, test_data.x[ii], prob_estimates);
			
			int flag = 0;
			unordered_map<double, double> tempProb;
			for (int i = 0; i < nr_class; i++) {
				flag = 0;
				for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
					for (j = 0; j < it->second.size(); j++) {
						if ((int)it->second[j] == model->label[i]) {
							tempProb[it->first] += prob_estimates[i];
							flag = 1;
							break;
						}
					}
					//prob_estimates[i] /= flag; // �Ƿ���Ҫ��ƽ����
					if (flag) break;
				}
			}
			int pos = 0, neg = 0;
			for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
				if (it->first == 1) {
					pos = it->second.size();
				}
				if (it->first == -1)
					neg = it->second.size();
			}
			for (auto it = tempProb.begin(); it != tempProb.end(); it++) {
				if (it->first == 1)
					it->second /= pos;
				if (it->second == -1)
					it->second /= neg;
			}

			double tMaxP = 0, tMaxY;
			for (auto it = tempProb.begin(); it != tempProb.end(); it++) {
				if (it->second > tMaxP) {
					tMaxP = it->second;
					tMaxY = it->first;
				}
			}
			predict_label = tMaxY;

			// ��ò���������С��������϶�Ӧ��Ԥ�����
			for (j = 0; j < nr_class; j++) {
				for (int k = 0; k < posLabel.size(); k++) {
					if (model->label[j] == (int)posLabel[k])
						test_data.r_y_p[ii] += prob_estimates[j];
				}
			}
			// printf("---%.0f %lf\n", test_data.r_y[ii], test_data.r_y_p[ii]);

			fprintf(output, "\n");
		}

		else
		{
			predict_label = svm_predict(model0, test_data.x[ii]);
			fprintf(output, "%g\n", predict_label);
		}
		// test_data.y[ii] = predict_label; // ԭʼ��λ��

		if (!proCombination) {
			// ת����ǩ
			int flag = 0;
			for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
				for (int k = 0; k < it->second.size(); k++) {
					if (it->second[k] == predict_label) {
						// cout << predict_label << " " << it->first << " " << test_data.r_y[ii] << endl;
						predict_label = it->first;
						flag = 1;
						break;
					}
				}
				if (flag) break;
			}
		}
		test_data.y[ii] = predict_label;
		svm_pred[ii] = predict_label;

		//printf("%.0f\n", predict_label);

		// ���ò��������ڰ�ලkmeans�ϵ�Ԥ�����
		// if (test_data.y[ii] != c1[ii] && test_data.y[ii] == -1) {
	    // test_data.y[ii] = k_param.pos[cluster1[ii]] == 1 ? 1 : -1;
		// }

		if (predict_label == test_data.r_y[ii])
			++correct;
		error += (predict_label - test_data.r_y[ii])*(predict_label - test_data.r_y[ii]);
		sump += predict_label;
		sumt += test_data.r_y[ii];
		sumpp += predict_label * predict_label;
		sumtt += test_data.r_y[ii] * test_data.r_y[ii];
		sumpt += predict_label * test_data.r_y[ii];
		++total;
	}

	if (svm_type == NU_SVR || svm_type == EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n", error / total);
		printf("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt - sump * sumt)*(total*sumpt - sump * sumt)) /
			((total*sumpp - sump * sump)*(total*sumtt - sumt * sumt))
		);
	}
	else
		printf("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct / total * 100, correct, total);
	if (predict_probability)
		free(prob_estimates);
	SVM_ACC = (double)correct / total;

	fclose(output);
}

void getEachClusterAcc(kmean_param &kpara, svm_problem &kprob, char* argv, int No, int f)
{
	int i, j;
	double t_acc = 0;
	FILE *fp_cluster_result;
	FILE *fp_cluster_summary;
	char saveFile[100] = "cluster_summary.txt";
	fp_cluster_summary = fopen(saveFile, "a");
	fp_cluster_result = fopen(argv, "w");
	for (int i = 0; i < kprob.l; i++) {
		if (c[i] == kprob.r_y[i]) t_acc++;
	}
	cout << "��ල������ѵ������ACC:" << (double)t_acc / kprob.l << endl;

	// �ֱ�����������׼ȷ��
	int countP = 0, accP = 0;
	int countN = 0, accN = 0;
	for (int i = 0; i < kprob.l; i++) {
		if (kprob.use[i] == 0) continue;
		if ((int)kprob.r_y[i] == 1) {
			countP++;
			if (c[i] == kprob.r_y[i]) accP++;
		}
		if ((int)kprob.r_y[i] == -1) {
			countN++;
			if (c[i] == kprob.r_y[i]) accN++;
		}
	}

	// �ֱ������������������׼ȷ��
	int tcountP = 0, taccP = 0;
	int tcountN = 0, taccN = 0;
	for (int i = 0; i < test_data.l; i++) {
		if ((int)test_data.r_y[i] == 1) {
			tcountP++;
			if (c1[i] == test_data.r_y[i]) taccP++;
		}
		if ((int)test_data.r_y[i] == -1) {
			tcountN++;
			if (c1[i] == test_data.r_y[i]) taccN++;
		}
	}

	// ��ø�������Ӧ�Ĵظ���
	unordered_map<int, int> labelCount;
	for (int i = 0; i < model->nr_class; i++) {
		for (int j = 0; j < kpara.k; j++) {
			if (kpara.use[j] == 0) continue;
			if (kpara.y_c[j] == model->label[i]) {
				labelCount[model->label[i]]++;
			}
		}
	}
	int p1 = 0, n1 = 0;
	for (auto it = labelCount.begin(); it != labelCount.end(); it++) {
		printf("��ǩΪ %d �Ĵصĸ���Ϊ��%d\n", it->first, it->second);
		if (it->first == 1)
			p1 = it->second;
		if (it->first == -1)
			n1 = it->second;
	}

	// ��¼�ش���֮�����������Դص����������������������԰������б�ǩ������������IR
	if (f == 1) {
		resultMap[4].push_back(p1);
		resultMap[4].push_back(n1);
		int posNum = 0, negNum = 0;
		for (i = 0; i < kpara.k; i++) {
			if (kprob.use[i] == 0) continue;
			if (kpara.pos[i] == 1)
				posNum += count1[i];
			else
				negNum += count1[i];
		}
		resultMap[4].push_back(posNum);
		resultMap[4].push_back(negNum);
		resultMap[4].push_back(negNum * 1.0 / posNum);
	}

	printf("\n");
	fprintf(fp_cluster_result, "\n%d\t%d", p1, n1); // �����С����Ӧ�صĸ���
	if(f == 0)
		fprintf(fp_cluster_summary, "\n%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%d\t%d\t%lf\t%lf", No, p1, n1, countP, countN, accP * 1.0 / countP, accN * 1.0 / countN, 
			tcountP, tcountN, taccP * 1.0 / tcountP, taccN * 1.0 / tcountN);
	else
		fprintf(fp_cluster_summary, "\t%d\t%d\t%d\t%d\t%d\t%lf\t%lf\t%d\t%d\t%lf\t%lf", No, p1, n1, countP, countN, accP * 1.0 / countP, accN * 1.0 / countN,
			tcountP, tcountN, taccP * 1.0 / tcountP, taccN * 1.0 / tcountN);

	for (i = 0; i < kpara.k; i++) // �������ڸô���Ԥ����ȷ��ѵ��������SSE
	{
		if (kpara.use[i] == 0) continue;
		int nn = 0, acc = 0;

		// ������Լ���ACC
		for (j = 0; j < test_data.l; j++)
		{
			if (cluster1[j] == i)// && edge_flag[i]==false)
			{
				nn++;
				if (c1[j] == test_data.r_y[j]) acc++;
			}
		}
		//printf("\n ���Լ���Cluster %d (Label %f) accuracy is %f,(%d/%d)", i, kpara.y_c[i], double(acc) / nn, acc, nn);
		//fprintf(fp_cluster_result, "\n ���Լ���Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", i, kpara.y_c[i], double(acc) / nn, acc, nn);

		// ����ѵ������ACC
		count_acc[i] = 0;
		for (j = 0; j < kprob.l; j++)
		{
			if (cluster[j] == i && kprob.y[j] == c[j])//(cluster[j] == i && kprob->y[j]==kpara->y_c[i])// 
			{
				count_acc[i]++;
			}
		}

		if (count1[i] == 0)
			cluster_acc[i] = 0;
		else
			cluster_acc[i] = (double)count_acc[i] / count1[i];
		//printf("\n ѵ������Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", i, kpara.y_c[i], cluster_acc[i], count_acc[i], count1[i]);
		//fprintf(fp_cluster_result, "\n ѵ������Cluster %d (Label %.0f) accuracy is %f,(%d/%d)\t���Լ�: accuracy is %f,(%d/%d)", i, kpara.y_c[i], cluster_acc[i], count_acc[i], count1[i], double(acc) / nn, acc, nn);
		fprintf(fp_cluster_result, "\n Cluster\t%d\t%.0f\t%f\t%d,%d\t%f\t%d\t%d", i, kpara.y_c[i], cluster_acc[i], count_acc[i], count1[i], double(acc) / nn, acc, nn);
		// fprintf(fp_cluster_result, "%d\t%.0f\t\%f\t%d\t%d\n", i, kpara.y_c[i], cluster_acc[i], count_acc[i], count1[i]);

	}
	fclose(fp_cluster_result);
}

void getEachClusterAcc(kmean_param &kpara, svm_problem &kprob)
{
	int i, j;
	double t_acc = 0;
	FILE *fp_cluster_result;
	char saveFile[100] = "cluster.txt";
	fp_cluster_result = fopen(saveFile, "a");
	for (int i = 0; i < kprob.l; i++) {
		if (c[i] == kprob.r_y[i]) t_acc++;
	}
	cout << "��ල������ѵ������ACC:" << (double)t_acc / kprob.l << endl;
	for (i = 0; i < kpara.k; i++) //�������ڸô���Ԥ����ȷ��ѵ��������SSE
	{
		if (kpara.use[i] == 0) continue;
		count_acc[i] = 0;
		for (j = 0; j < kprob.l; j++)
		{
			if (cluster[j] == i && kprob.y[j] == c[j])//(cluster[j] == i && kprob->y[j]==kpara->y_c[i])// 
			{
				count_acc[i]++;
			}
		}
		if (count1[i] == 0)
			cluster_acc[i] = 0;
		else
			cluster_acc[i] = (double)count_acc[i] / count1[i];
		printf("\n Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", i, kpara.y_c[i], cluster_acc[i], count_acc[i], count1[i]);
		fprintf(fp_cluster_result, "%d\t%.0f\t\%f\t%d\t%d\n", i, kpara.y_c[i], cluster_acc[i], count_acc[i], count1[i]);
	}
	fclose(fp_cluster_result);
}

void getAvgSubclass(kmean_param &kpara, svm_problem &kprob)
{
	int trueNum = 0;
	for (int i = 0; i < kpara.k; i++) {
		for (int j = 0; j < kprob.l; j++) {
			if (sub_cluster[j] == i && kprob.r_y[j] == sub_c[j]) {
				trueNum++;
			}
		}
	}
	double tSub = (double)trueNum / useClusterNum;
	avgSubclass = (tSub > 0.0) ? floor(tSub + 0.5) : ceil(tSub - 0.5);
	cout << endl << "�����ƽ����С�ǣ�" << avgSubclass << endl;
	//if (avgSubclass < 10) avgSubclass = prob.l / model->nr_class; // ���಻��̫С--��̫С��������չΪ��prob.l / model->nr_class
	//cout << endl << "����̫С�������䣬�����ƽ����С�ǣ�" << avgSubclass << endl;
}

void getSamplingRatio()
{
	int useNegNum = 0; // �����Ķ����������ĸ���
	int usePosClusterNum = 0; // ������������صĸ���
	for (int i = 0; i < k_param.k; i++) {
		for (int j = 0; j < k_prob.l; j++) {
			if (cluster[j] == i && k_prob.r_y[j] == c[j] && k_prob.use[i] == 1 && k_prob.pos[j] == 0) {
				useNegNum++;
			}
		}
		if (k_param.pos[i] == 1)
			usePosClusterNum++;
	}
	/*for (int j = 0; j < k_prob.l; j++) {
		if (k_prob.use[j] == 1 && k_prob.r_y[j] == -1)
			useNegNum++;
	}*/
	double tSub = (double)useNegNum / usePosClusterNum;
	avgSubclass = (tSub > 0.0) ? floor(tSub + 0.5) : ceil(tSub - 0.5);
	cout << endl << "�����ƽ����С�ǣ�" << avgSubclass << endl;
	//if (avgSubclass < 10) avgSubclass = prob.l / model->nr_class; // ���಻��̫С--��̫С��������չΪ��prob.l / model->nr_class
	//cout << endl << "����̫С�������䣬�����ƽ����С�ǣ�" << avgSubclass << endl;
}

void getMaxSubCluster()
{
	int maxId, maxNum = 0;
	for (int i = 0; i < k_param.k; i++) {
		if ((int)k_param.y_c[i] == -1) {
			if (sub_count1[i] > maxNum) {
				maxId = i;
				maxNum = sub_count1[i];
			}
		}
	}
	maxSubNegCluster = maxNum;
}

int setEachSubSample() // Ϊÿ��������Ӵ����ò�ͬ�Ĳ�����
{
	int maxId, maxNum = 0;
	for (int i = 0; i < k_param.k; i++) {
		if ((int)k_param.y_c[i] == 1) {
			if (sub_count1[i] > maxNum) {
				maxId = i;
				maxNum = sub_count1[i];
			}
		}
	}
	getMaxSubCluster(); // ������ĴصĴ�С
	return maxId;
}

void getEachSubclassAcc() // ���������Ϻ��׼ȷ��
{
	int i, j;
	double t_acc = 0;
	for (int i = 0; i < sub_prob.l; i++) {
		if (sub_c[i] == sub_prob.r_y[i]) t_acc++;
	}
	cout << "\n��ල������ѵ������ACC:" << (double)t_acc / sub_prob.l << endl;
	for (i = 0; i < k_param.k; i++) //�������ڸô���Ԥ����ȷ��ѵ��������SSE
	{
		count_acc[i] = 0;
		if (k_param.use[i] == 0) continue;
		for (j = 0; j < sub_prob.l; j++)
		{
			if (sub_prob.use[j] == 0) continue;
			if (sub_cluster[j] == i && sub_prob.r_y[j] == sub_c[j])//(cluster[j] == i && kprob->y[j]==kpara->y_c[i])// 
			{
				count_acc[i]++;
			}
		}
		if (sub_count1[i] == 0)
			cluster_acc[i] = 0;
		else
			cluster_acc[i] = (double)count_acc[i] / sub_count1[i];
		printf("\n Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", i, k_param.y_c[i], cluster_acc[i], count_acc[i], sub_count1[i]);
	}
}

bool cmp(sortSample& x, sortSample& y)
{
	if (x.d != y.d) return x.d < y.d;
	else return x.id < y.id;
}

void reSampling()
{
	int id = 0;
	// ��������������Ԥ����ȷ��ѵ������
	for (int i = 0; i < k_param.k; i++) {
		if (k_param.use[i] == 0) continue; // use = 0 : �ôر�ɾ��
		sub_count1[i] = 0;
		for (int j = 0; j < prob.l; j++) {
			if (cluster[j] == i && prob.r_y[j] == c[j]) { // ���ڸô���Ԥ����ȷ��ѵ������
				sub_prob.r_y[id] = prob.r_y[j];
				sub_prob.y[id] = prob.y[j];
				copy(prob.x[j], sub_prob.x[id]);
				sub_cluster[id] = cluster[j]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c[j]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				id++;
			}
		}
	}
	sub_prob.l = id; // �ݶ�����ĸ���
	cout << endl << "��������������" << sub_prob.l << endl;

	// ��Ŀǰ����׼ȷ��
	getEachSubclassAcc();
	getAvgSubclass(k_param, sub_prob);

	// ���濪ʼ�����ز���--����avgSubclass
	// step1: ��������ѡ�����ŶȸߵĲ�����������������
	//id = 0;
	// double *dis = new double[test_data.l]; // ������в������������ĵľ���
	for (int i = 0; i < k_param.k; i++) {
		if (sub_count1[i] == 0) continue; // ���ô�������Ϊ0��ɾ��
		if (k_param.use[i] == 0) continue;
		sortSample temp;
		vector<sortSample> dis;
		for (int j = 0; j < test_data.l; j++) {
			if (cluster1[j] == i) {
				// dis[id++] = distance1(k_param.x_c[i], test_data.x[j]);
				temp.id = j;
				temp.d = distance1(k_param.x_c[i], test_data.x[j]);
				dis.push_back(temp);
			}
		}
		sort(dis.begin(), dis.end(), cmp); // ��С��������
		int dif_sub_avg = sub_count1[i] - avgSubclass; // �ж������ǽ��й���������Ƿ����
		int dif = dis.size() - abs(dif_sub_avg); // �ж��������������������Ƿ񹻴ﵽ����ƽ��
		int dif_sample;
		if (dif_sub_avg < 0) { // sub_count1[i] < avgSubclass   ������
			//int dif_sample = abs(dif) - dif_sub_avg;
			//cout << abs(dif) << "-" << dif_sub_avg << "=" << dif_sample << " " << abs(dif) - (int)dif_sub_avg << endl;
			if (dif < 0) { // �������ڵĲ�����������������������  Ŀǰ�ݶ��ȰѲ��Լ�������
				dif_sample = dis.size();
				printf("����%d������������������������%d������ʹ��oversample\n", i, abs(dif));
				for (int j = 0; j < abs(dif); j++) {
					sub_prob.r_y[id] = k_param.y_c[i];
					sub_prob.y[id] = k_param.y_c[i];
					copy(k_param.x_c[i], sub_prob.x[id]);
					sub_cluster[id] = i; // ��id��ѵ�������Ĵ����
					sub_c[id] = k_param.y_c[i]; // ��id��ѵ��������Ԥ���ǩ
					sub_count1[i]++;
					k_param.k_c[i]++;
					id++;
				}
			} 	
			else {
				dif_sample = dif_sub_avg;
				printf("����%d�����������㹻������%d��\n", i, abs(dif_sample));
			}
			for (int j = 0; j < abs(dif_sample); j++) {
				int tId = dis[j].id; // �������С�������к����������
				sub_prob.r_y[id] = test_data.r_y[tId];
				sub_prob.y[id] = test_data.y[tId];
				copy(test_data.x[tId], sub_prob.x[id]);
				sub_cluster[id] = cluster1[tId]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c1[tId]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				k_param.k_c[i]++;
				id++;
			}
		}
		// Ƿ������ν��У���
		/*else if (dif_sub_avg > 0) { // �������С��ƽ�������С��Ƿ����
			int len = dis.size();
			int dataNum = id; // Ŀǰ����������
			for (int j = 0; j < dif_sub_avg; j++) { // ɾ������������Զ��ѵ������
				int tId = dis[len - j - 1].id; // �������С�������к����������
				for (int k = tId; k < dataNum; k++) { // ɾ�����ø��ǵķ�ʽ
					sub_prob.r_y[k] = sub_prob.r_y[k + 1];
					sub_prob.y[k] = sub_prob.y[k + 1];
					sub_cluster[k] = sub_cluster[k + 1];
					sub_c[k] = sub_c[k + 1];
					// �޸�dis�е�����ֵ
					for (int kk = 0; kk < len; kk++) {
						if (dis[kk].id > tId) dis[kk].id--;
					}
				}
				sub_count1[i]--;
				k_param.k_c[i]--;
				id--;
			}
			printf("����%d��ѵ���������࣬����%d����Ŀǰ��ʣ%d��\n", i, abs(dif_sub_avg), sub_count1[i]);
		}*/	
	}
	sub_prob.l = id; // ��������ĸ���
	cout << endl << id << " �ز�����������׼ȷ�ʣ�" << endl;
	getEachSubclassAcc();
}

// ��ɾ���ص�ʱ�򣬰��������ѵ��������Ŀ����ɾ��
void cluster_prepoocess(kmean_param * kpara) // ����ͬ�����صľ���ͽ���Ԥ��
{
	int i, j, t, n, index1 = -1, index2 = -1, acc = 0, id, no = 1;
	double min1, min2, sum0, sum, d;
	bool flag = true;

	//for (i = 0; i < kpara->k; i++) {
		// printf("cluster %d has samples: %d\n", i, kpara->size[i]); 
	//	if(count1[i] < 3)
	//		printf("cluster %d (Label:%.0f) has train samples: %d(all samples:%d)\n", i, kpara->y_c[i], count1[i], kpara->size[i]);
	//}
	while (flag) {
		printf("��ʼ��%d�μ��\n", no);
		no++;
		for (i = 0; i < kpara->k; i++)
		{
			if (kpara->pos[i] == 0) //��ǰ�����ڶ�����
			{
				for (j = 0; j < prob.l; j++) {
					if (prob.r_y[j] != kpara->y_c[i] && cluster[j] == i) // ���ڸôأ����Ǳ�ǩ�ʹ˴ز�ͬ,��ô������Ϊ������
					{
						min1 = 100000;
						for (t = 0; t < kpara->k; t++) // Ѱ�Ҹ�j�����������ز�����
						{
							if (i != t && kpara->pos[t] == 1) {
								d = distance1(kpara->x_c[t], prob.x[j]);
								if (d < min1) {
									min1 = d;
									id = t;
								}
							}
						}
						// ���ӵ���Ӧ���������
						c[j] = kpara->y_c[id];
						cluster[j] = id;
						kpara->size[id]++;
						kpara->size[i]--;
						kpara->k_c[i]--;
						kpara->k_c[id]++;
						count1[id]++;
						count1[i]--;
					}
				}
			}
			if (kpara->pos[i] == 1) // ��ǰ������������
			{
				for (j = 0; j < prob.l; j++) {
					if (prob.r_y[j] != kpara->y_c[i] && cluster[j] == i) // ���ڸôأ����Ǳ�ǩ�ʹ˴ز�ͬ,��ô������Ϊ������
					{
						int t = count1[i];
						// ɾ����Щѵ������
						kpara->size[i]--;
						kpara->k_c[i]--;
						count1[i]--;
						prob.use[j] = 0;
						k_prob.use[j] = 0;
						cluster[j] = -1;
						printf("\n��%d�У�ѵ������:%d��ɾ����������:%d-->%d", i, j, t, count1[i]);
					}
				}
			}
		}

		for (i = 0; i < kpara->k; i++) // ɾ��С�ߴ�Ķ������
		{
			// ��ʽ1����ɾ��������
			if (kpara->pos[i] == 0 && kpara->size[i] < 3) // ������ó�����-����ʾ3���ֵ
			//if (kpara->pos[i] == 0 && count1[i] < 3)
			{
				//int t = count1[i];
				kmean_delete_centroid_samples(kpara, &k_prob, i);
				//printf("��������%d-->%d\n", t, count1[i]);
			}

			// ��ʽ2����ɾ����Ҳ���ǽ�������غϲ�
			/*if (kpara->pos[i] == 0 && count1[i] < 3) {
				min1 = min2 = 1000000;
				for (j = 0; j < kpara->k; j++)
				{
					if (i != j && min1 > distance1(kpara->x_c[i], kpara->x_c[j])) // Ѱ�Ҹ��ôؾ�������Ĵ�
					{
						min1 = distance1(kpara->x_c[i], kpara->x_c[j]);
						index1 = j;
					}
					if (i != j && kpara->y_c[i] == kpara->y_c[j]) // Ѱ�Ҹ��ô�ͬ����Ҿ�������Ĵ�
					{
						if (min2 > distance1(kpara->x_c[i], kpara->x_c[j]))
						{
							min2 = distance1(kpara->x_c[i], kpara->x_c[j]);
							index2 = j;
						}
					}
				}
				if (index1 == index2) // �����������Ĵ���ͬ���ģ���ֱ�Ӻϲ�
				{
					(kpara->k)--;
					for (j = i; j < kpara->k; j++)//����ÿ���أ�ǰ��һλ////
					{
						kpara->k_c[j] = kpara->k_c[j + 1];
						kpara->y_c[j] = kpara->y_c[j + 1];
						kpara->size[j] = kpara->size[j + 1];
						copy(kpara->x_c[j + 1], kpara->x_c[j]);
						kpara->size[j] = kpara->size[j + 1];	//gei�ṹ������һ����������ÿ��Ԥ�����ֵ
						count1[j] = count1[j + 1];
						old_count[j] = old_count[j + 1];
						cluster_acc[j] = cluster_acc[j + 1];
						old_cluster_acc[j] = old_cluster_acc[j + 1];
						sse[i] = sse[j + 1]; old_sse[j] = old_sse[j + 1];
						new_split[j] = new_split[j + 1];
						count_acc[j] = count_acc[j + 1];
						old_count_acc[j] = old_count_acc[j + 1];
					}
					// ɾ���굱ǰ�Ĵ�֮���ж���ϲ����Ĵص������Ƿ�仯
					if (index1 > i) {
						index1 = index1 - 1;
						index2 = index2 - 1;
					}
					for (j = 0; j < k_prob.l; j++) //����ѵ������
					{
						if (cluster[j] == i)
						{
							cluster[j] = index1; kpara->size[index1]++; kpara->k_c[index1]++; count1[index1]++;
						}
						else if (cluster[j] > i)
						{
							cluster[j] = cluster[j] - 1; old_cluster[j] = old_cluster[j] - 1;
						}
					}
					for (j = 0; j < test_data.l; j++) // ���ڲ�������
					{
						if (cluster1[j] == i)
						{
							cluster1[j] = index1; kpara->size[index1]++;
						}
						if (cluster1[j] > i)
							cluster1[j] = cluster1[j] - 1;
					}
					printf("\n(�ϲ�)��%d--->��%d", i, index1);
				}
				else
					kmean_delete_centroid_samples(kpara, &k_prob, i);
			}*/
		}

		for (i = 0; i < kpara->k; i++) //����С�ߴ���������
		{
			if (kpara->pos[i] == 1 && kpara->size[i] < 3)
				// if (kpara->pos[i] = 1 && count1[i] < 3)
			{
				min1 = min2 = 1000000;
				for (j = 0; j < kpara->k; j++)
				{
					if (i != j && min1 > distance1(kpara->x_c[i], kpara->x_c[j])) // Ѱ�Ҹ��ôؾ�������Ĵ�
					{
						min1 = distance1(kpara->x_c[i], kpara->x_c[j]);
						index1 = j;
					}
					if (i != j && kpara->y_c[i] == kpara->y_c[j]) // Ѱ�Ҹ��ô�ͬ����Ҿ�������Ĵ�
					{
						if (min2 > distance1(kpara->x_c[i], kpara->x_c[j]))
						{
							min2 = distance1(kpara->x_c[i], kpara->x_c[j]);
							index2 = j;
						}
					}
				}
				if (index1 == index2 && index1 != -1) // �����������Ĵ���ͬ���ģ���ֱ�Ӻϲ�
				{
					(kpara->k)--;
					int t = count1[index1];
					for (j = i; j < kpara->k; j++)//����ÿ���أ�ǰ��һλ////
					{
						kpara->k_c[j] = kpara->k_c[j + 1];
						kpara->y_c[j] = kpara->y_c[j + 1];
						kpara->size[j] = kpara->size[j + 1];
						copy(kpara->x_c[j + 1], kpara->x_c[j]);
						kpara->size[j] = kpara->size[j + 1];	//gei�ṹ������һ����������ÿ��Ԥ�����ֵ	
						count1[j] = count1[j + 1];
						old_count[j] = old_count[j + 1];
						cluster_acc[j] = cluster_acc[j + 1];
						old_cluster_acc[j] = old_cluster_acc[j + 1];
						sse[i] = sse[j + 1]; old_sse[j] = old_sse[j + 1];
						new_split[j] = new_split[j + 1];
						count_acc[j] = count_acc[j + 1];
						old_count_acc[j] = old_count_acc[j + 1];
					}
					// ɾ���굱ǰ�Ĵ�֮���ж���ϲ����Ĵص������Ƿ�仯
					if (index1 > i) {
						index1 = index1 - 1;
						index2 = index2 - 1;
					}
					for (j = 0; j < k_prob.l; j++) //����ѵ������
					{
						if (cluster[j] == i)
						{
							cluster[j] = index1; kpara->size[index1]++; kpara->k_c[index1]++; count1[index1]++;
							c[j] = kpara->y_c[index1];
						}
						else if (cluster[j] > i)
						{
							cluster[j] = cluster[j] - 1; old_cluster[j] = old_cluster[j] - 1;
							c[j] = kpara->y_c[cluster[j]];
						}
					}
					for (j = 0; j < test_data.l; j++) // ���ڲ�������
					{
						if (cluster1[j] == i)
						{
							cluster1[j] = index1; kpara->size[index1]++;
							c1[j] = kpara->y_c[index1];
						}
						if (cluster1[j] > i) {
							cluster1[j] = cluster1[j] - 1; c1[j] = c1[cluster1[j]];
						}
					}
					printf("\n(�ϲ�)��%d--->��%d����������%d-->%d", i, index1, t, count1[index1]);
				} // end if(index1==index2)//�����������Ĵ���ͬ���ģ���ֱ�Ӻϲ�
			}
		}// end for(i=0;i<kpara->k;i++)//����С�ߴ���������
		kmean_update113(kpara, &k_prob); //���¸����ص�����
		//kmean_predict0(&k_param, &k_prob);
		kmean_predict_testdata0(&k_param); // �ޱ�ǩ����������Ԥ�� �������������䵽���ʵĴ���

		// ��鵱ǰ���д��Ƿ񶼺ϸ�
		flag = false;
		for (i = 0; i < k_param.k; i++) {
			printf("��%d����������%d\n", i, k_param.size[i]);
			if (k_param.size[i] < 3 && k_param.pos[i] == 0) {
				flag = true;
				break;
			}
		}
	}
	
	//////////////////////////////////���¸�������
	//kmean_update113(kpara, &k_prob);//���¸����ص�����
	//kmean_predict0(&k_param, &k_prob);//�б�ǩ����������Ԥ��
	//kmean_predict_testdata0(&k_param);//�ޱ�ǩ����������Ԥ��
	printf("\nĿǰ��ʣ�´� %d ��\n", kpara->k);
	useClusterNum = kpara->k;

	printf("�ش�����ɺ󣬸����ص������\n");
	for (i = 0; i < kpara->k; i++) // �������ڸô���Ԥ����ȷ��ѵ��������SSE
	{
		if (kpara->use[i] == 0) continue;
		int nn = 0, acc = 0;

		// ������Լ���ACC
		for (j = 0; j < test_data.l; j++)
		{
			if (cluster1[j] == i)// && edge_flag[i]==false)
			{
				nn++;
				if (c1[j] == test_data.r_y[j]) acc++;
			}
		}
		//printf("\n ���Լ���Cluster %d (Label %f) accuracy is %f,(%d/%d)", i, kpara.y_c[i], double(acc) / nn, acc, nn);
		//fprintf(fp_cluster_result, "\n ���Լ���Cluster %d (Label %.0f) accuracy is %f,(%d/%d)", i, kpara.y_c[i], double(acc) / nn, acc, nn);

		// ����ѵ������ACC
		count_acc[i] = 0;
		for (j = 0; j < k_prob.l; j++)
		{
			if (k_prob.use[j] == 0 || prob.use[j] == 0) continue;
			if (cluster[j] == i && k_prob.r_y[j] == c[j])//(cluster[j] == i && kprob->y[j]==kpara->y_c[i])// 
			{
				count_acc[i]++;
			}
		}

		if (count1[i] == 0)
			cluster_acc[i] = 0;
		else
			cluster_acc[i] = (double)count_acc[i] / count1[i];
		printf("\n ѵ������Cluster %d (Label %.0f) accuracy is %f,(%d/%d/%d)", i, kpara->y_c[i], cluster_acc[i], count_acc[i], count1[i], k_param.size[i]);
	}
}

/*     !!!!!!!!!!!! */
void reSamplingSmote()
{
	//getSamplingRatio();
	//for (int i = 0; i < k_param.k; i++)
	//	printf("\n%d", k_param.use[i]);
	int id = 0;
	// ��������������Ԥ����ȷ��ѵ������
	for (int i = 0; i < k_param.k; i++) {
		if (k_param.use[i] == 0) continue; // use = 0 : �ôر�ɾ��
		if (k_param.k_c[i] == 0 || count1[i] == 0) continue;
		sub_count1[i] = 0;
		k_param.k_c[i] = 0;
		k_param.size[i] = 0; // ���߶��Ǽ�¼�Ӵ�������������
		for (int j = 0; j < prob.l; j++) {
			if (prob.use[j] == 0) continue;
			if (cluster[j] == i && prob.r_y[j] == c[j]) { // ���ڸô���Ԥ����ȷ��ѵ������
				sub_prob.r_y[id] = prob.r_y[j];
				sub_prob.y[id] = prob.y[j];
				sub_prob.pos[id] = prob.pos[j];
				sub_prob.use[id] = prob.use[j];
				copy(prob.x[j], sub_prob.x[id]);
				sub_cluster[id] = cluster[j]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c[j]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				k_param.k_c[i]++;
				k_param.size[i]++;
				id++;
			}
		}
	}
	sub_prob.l = id; // �ݶ�����ĸ���
	cout << endl << "��������������" << sub_prob.l << endl;

	// ��Ŀǰ����׼ȷ��
	// deleteNegCluster();
	// getEachSubclassAcc();
	getAvgSubclass(k_param, sub_prob);
	int maxArea = setEachSubSample();

	// ���濪ʼ�����ز���--����avgSubclass
	// step1: ��������ѡ�����ŶȸߵĲ�����������������
	for (int i = 0; i < k_param.k; i++) {
		if (sub_count1[i] <= 0) continue; // ���ô�������Ϊ0��ɾ��
		if (k_param.use[i] == 0) continue;
		sortSample temp;
		vector<sortSample> dis;

		// ��ʽ1�����ݲ�����������ǰ���ĵľ���ɸѡ�����Ŷ�
		for (int j = 0; j < test_data.l; j++) {
			if (cluster1[j] == i) {
				temp.id = j;
				temp.d = distance1(k_param.x_c[i], test_data.x[j]);
				dis.push_back(temp);
			}
		}

		// ��ʽ2����������ָ��ɸѡ�����ŶȲ���������Ϊα��ǩ����  u1��kmean_final_predict()�����Ѿ������
		/*for (int j = 0; j < test_data.l; j++) {
			if (cluster1[j] == i) {
				temp.id = j;
				temp.d = u1[j];
				dis.push_back(temp);
			}
		}*/
		sort(dis.begin(), dis.end(), cmp); // ��С��������
		int dif_sub_avg = sub_count1[i] - avgSubclass; // �ж������ǽ��й���������Ƿ����
		int addTestNum = dis.size() * 0.1;
		//int addTestNum = dis.size();

		// ���ӿ������ĵ�10%����������10%Զ�����ĵĲ�������
		if (addTestNum > 0 && dif_sub_avg < 0 && k_param.y_c[i] == 1) { // sub_count1[i] < avgSubclass   ������
			// step1.�������Ӿ������������10%
			for (int j = 0; j < addTestNum; j++) {
				int tId = dis[j].id; // �������С�������к����������
				sub_prob.r_y[id] = 1;
				sub_prob.y[id] = 1;
				sub_prob.pos[id] = 1;
				sub_prob.use[id] = 1;
				copy(test_data.x[tId], sub_prob.x[id]);
				sub_cluster[id] = cluster1[tId]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c1[tId]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				k_param.k_c[i]++;
				k_param.size[i]++;
				id++;
			}
			// step2.�����Ӿ���������Զ��10%
			/*for (int j = dis.size() - 1; j >= (dis.size() - addTestNum); j--) {
				int tId = dis[j].id; // �������С�������к����������
				sub_prob.r_y[id] = 1;
				sub_prob.y[id] = 1;
				sub_prob.pos[id] = 1;
				sub_prob.use[id] = 1;
				// ���ڲ���������һ��Ԥ��ԣ�����Ĭ�϶���������ǩ
				//sub_prob.r_y[id] = test_data.r_y[tId];
				//sub_prob.y[id] = test_data.y[tId];
				//sub_prob.pos[id] = prob.pos[tId];
				//sub_prob.use[id] = prob.use[tId];
				copy(test_data.x[tId], sub_prob.x[id]);
				sub_cluster[id] = cluster1[tId]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c1[tId]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				k_param.k_c[i]++;
				k_param.size[i]++;
				id++;
			}*/
		}

		dif_sub_avg = sub_count1[i] - avgSubclass; // �ж����Ӳ����������Ƿ�ﵽƽ��
		// ��ʼ����SMOTE
		if (dif_sub_avg < 0 && k_param.y_c[i] == 1) {
			// step1:ȷ���������ʣ������K����ɸѡ��Щ������Ҫ����SMOTE
			int KN = 3;
			int N = abs(dif_sub_avg); // ���ѡ����������������������
			//if (i == maxArea) // �����������С�����Ҫ�ֲ�����������븺����Ҫ�ֲ�������ͬ������
			//	N = maxSubNegCluster;
			vector<int> idInCluster; // ��ǰ����������id
			vector<int> selectId; // ��ѡ�н��в���������id
			for (int j = 0; j < id; j++) { // �����Ӧ���������ģ���Ϊid������
				if (sub_cluster[j] == i)
					idInCluster.push_back(j);
			}
			//srand((unsigned)time(NULL));
			for (int j = 0; j < N; j++) {
				int sId = rand() % (idInCluster.size() - 0) + 0;
				selectId.push_back(idInCluster[sId]); //ע�����ӵ�����ǣ�idInCluster[sId]��sIdֻ����idInCluster�е�����
			}
			/*for (int j = 0; j < idInCluster.size(); j++)
				cout << idInCluster[j] << " ";
			// cout << idInCluster[j] << " " << sub_cluster[idInCluster[j]] << " " << sub_prob.r_y[idInCluster[j]] << endl;
			cout << endl;
			for (int j = 0; j < N; j++)
				cout << selectId[j] << " ";
			cout << endl;
			for (int j = 0; j < N; j++)
				cout << sub_c[selectId[j]] << " " << sub_cluster[j] << endl;
			cout << endl;*/
			// step2: ����ÿһ����Ҫ��SMOTE��������������K����
			for (int j = 0; j < N; j++) {
				int curId = selectId[j]; // curId��ʾ��׼������id
				// cout << sub_prob.r_y[curId] << endl;
				vector<sortSample> kNearst;
				// ���������������3������ȡ�����ʱԽ��
				if (idInCluster.size() > 2) {
					for (int k = 0; k < idInCluster.size(); k++) {
						if ((idInCluster[k] != curId)) { // ���������if (idInCluster[k] != curId) ���������if ((idInCluster[k] != curId) || (idInCluster.size() == 1) || ((idInCluster.size() == 2)))
							temp.id = idInCluster[k];
							temp.d = distance1(sub_prob.x[temp.id], sub_prob.x[curId]);
							kNearst.push_back(temp);
						}
					}
				}
				else {
					for (int k = 0; k < idInCluster.size(); k++) {
						// ���������if (idInCluster[k] != curId)
						temp.id = idInCluster[k];
						temp.d = distance1(sub_prob.x[temp.id], sub_prob.x[curId]);
						kNearst.push_back(temp);
					}
					// �������������٣�������Ҳ���ӽ�ȥ
					temp.id = -1;
					temp.d = distance1(k_param.x_c[i], sub_prob.x[curId]);
					kNearst.push_back(temp);
				}

				KN = KN > kNearst.size() ? kNearst.size() : KN;
				sort(kNearst.begin(), kNearst.end(), cmp);
				// step3: ���ݹ�ʽ�����µ����������ӵ����ݼ���
				int tId = rand() % (KN - 0) + 0;
				int sId = kNearst[tId].id; // ��ѡ�еĽ���������id
				if (sId != -1) {
					svm_node *newSample = Malloc(struct svm_node, max_index + 1);
					diminish(sub_prob.x[curId], sub_prob.x[sId]);  // x - x'
					copy(result, newSample);
					double factor = rand() / double(RAND_MAX);
					multiply(factor, newSample);
					add(newSample, sub_prob.x[curId]);
					copy(result, newSample);

					// ������curId�������sub_prob��������˵�ģ����Բ���ʹ��cluster[curId]  c[curId]
					sub_prob.r_y[id] = sub_prob.r_y[curId];
					sub_prob.y[id] = sub_prob.y[curId];
					sub_prob.pos[id] = sub_prob.pos[curId];
					sub_prob.use[id] = sub_prob.use[curId];
					//sub_prob.x[id] = newSample; // ���ָ�ֵ��ʽ
					copy(newSample, sub_prob.x[id]);
					sub_cluster[id] = i;
					sub_c[id] = 1;
					// sub_cluster[id] = cluster[curId]; // ��id��ѵ�������Ĵ����
					// sub_c[id] = c[curId]; // ��id��ѵ��������Ԥ���ǩ
					sub_count1[i]++;
					k_param.k_c[i]++;
					k_param.size[i]++;
					// cout << curId << " " << sub_prob.y[id] << " " << c[curId] << endl;
					id++;
				}
				else {
					svm_node *newSample = Malloc(struct svm_node, max_index + 1);
					diminish(sub_prob.x[curId], k_param.x_c[i]);  // x - x'
					copy(result, newSample);
					double factor = rand() / double(RAND_MAX);
					multiply(factor, newSample);
					add(newSample, sub_prob.x[curId]);
					copy(result, newSample);

					// ������curId�������sub_prob��������˵�ģ����Բ���ʹ��cluster[curId]  c[curId]
					sub_prob.r_y[id] = k_param.y_c[i];
					sub_prob.y[id] = k_param.y_c[i];
					sub_prob.pos[id] = 1;
					sub_prob.use[id] = 1;
					//sub_prob.x[id] = newSample;
					copy(newSample, sub_prob.x[id]);
					sub_cluster[id] = i;
					sub_c[id] = 1;
					// sub_cluster[id] = cluster[curId]; // ��id��ѵ�������Ĵ����
					// sub_c[id] = c[curId]; // ��id��ѵ��������Ԥ���ǩ
					sub_count1[i]++;
					k_param.k_c[i]++;
					k_param.size[i]++;
					id++;
				}
				kNearst.clear();
			}
			selectId.clear();
			idInCluster.clear();
		}
		// Ƿ�������֣�Ƿ������һ����Ҫ
		/*else if (dif_sub_avg > 0 && k_param.y_c[i] == -1) {
			dis.clear(); // ���֮ǰ������
			if (sub_count1[i] <= 0) continue; // ���ô�������Ϊ0��ɾ��
			if (k_param.use[i] <= 0) continue;
			for (int j = 0; j < sub_prob.l; j++) { // ����ô��ڵ�ѵ����
				if (sub_cluster[j] == i) {
					temp.id = j;
					temp.d = distance1(k_param.x_c[i], prob.x[j]);
					dis.push_back(temp);
				}
			}
			sort(dis.begin(), dis.end(), cmp); // ��С��������
			int deleteNegNum = dif_sub_avg * 0.5;
			// ���ӿ������ĵ�10%����������10%Զ�����ĵĲ�������
			if (deleteNegNum > 0 && dif_sub_avg > 0) { // sub_count1[i] < avgSubclass   ������
				// step1.����ɾ���������������10%
				for (int j = 0; j < deleteNegNum; j++) {
					int tId = dis[j].id; // �������С�������к����������
					sub_prob.pos[tId] = 0;
					sub_prob.use[tId] = 0;
					sub_count1[i]--;
					k_param.k_c[i]--;
					//id--;
				}
				// step2.�����Ӿ���������Զ��10%
				for (int j = dis.size() - 1; j >= (dis.size() - deleteNegNum); j--) {
					int tId = dis[j].id; // �������С�������к����������
					sub_prob.pos[tId] = 0;
					sub_prob.use[tId] = 0;
					sub_count1[i]--;
					k_param.k_c[i]--;
					//id--;
				}
			}
		}*/
	}
	sub_prob.l = id; // ��������ĸ���
	cout << endl << id << " �ز�����������׼ȷ�ʣ�" << endl;
	getEachSubclassAcc();
	// ��������Ϊ0�Ĵ�ɾ��
	for (int i = 0; i < k_param.k; i++) {
		if (sub_count1[i] <= 0)
			k_param.use[i] = 0;
	}
}

void createSubProb()
{
	int id = 0;
	for (int i = 0; i < k_param.k; i++) {
		sub_count1[i] = 0;
		if (k_param.use[i] == 0) continue;
		for (int j = 0; j < prob.l; j++) {
			if (cluster[j] == i) { // ���ڸôص�ѵ������
				sub_prob.r_y[id] = prob.r_y[j];
				sub_prob.y[id] = prob.y[j];
				sub_prob.pos[id] = prob.pos[j];
				copy(prob.x[j], sub_prob.x[id]);
				sub_cluster[id] = cluster[j]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c[j]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				id++;
			}
		}
	}
	sub_prob.l = id; // �ݶ�����ĸ���
	cout << endl << "��������������" << sub_prob.l << endl;
}

void createSubProbWithUnlabel()
{
	int id = 0;
	for (int i = 0; i < k_param.k; i++) {
		sub_count1[i] = 0;
		if (k_param.use[i] == 0) continue;
		for (int j = 0; j < prob.l; j++) {
			if (cluster[j] == i) { // ���ڸôص�ѵ������
				sub_prob.r_y[id] = prob.r_y[j];
				sub_prob.y[id] = prob.y[j];
				copy(prob.x[j], sub_prob.x[id]);
				sub_cluster[id] = cluster[j]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c[j]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				id++;
			}
		}
	}
	for (int i = 0; i < k_param.k; i++) {
		sub_count1[i] = 0;
		if (k_param.use[i] == 0) continue;
		for (int j = 0; j < test_data.l; j++) {
			if (cluster1[j] == i) { // ���ڸôصĲ�������
				sub_prob.r_y[id] = test_data.r_y[j];
				sub_prob.y[id] = test_data.y[j];
				copy(test_data.x[j], sub_prob.x[id]);
				sub_cluster[id] = cluster1[j]; // ��id��ѵ�������Ĵ����
				sub_c[id] = c1[j]; // ��id��ѵ��������Ԥ���ǩ
				sub_count1[i]++;
				id++;
			}
		}
	}
	sub_prob.l = id; // �ݶ�����ĸ���
	cout << endl << "��������������" << sub_prob.l << endl;
}

void getEachClusterNumAndSampleNum()
{
	unordered_map<int, double> labelCount;
	for (int i = 0; i < model->nr_class; i++) {
		for (int j = 0; j < k_param.k; j++) {
			if (k_param.y_c[j] == model->label[i]) {
				labelCount[model->label[i]]++;
			}
		}
	}
	for (auto it = labelCount.begin(); it != labelCount.end(); it++) {
		printf("��ǩΪ %d �Ĵصĸ���Ϊ��%.0f\n", it->first, it->second);
	}
	printf("\n");
	//for (int i = 0; i < k_param.k; i++) {
	//	printf("��ǩΪ%.0f�Ĵأ��������ܸ���Ϊ��%d\n", k_param.y_c[i], k_param.k_c[i]);
	//}
	unordered_map<double, int> map1;
	for (int i = 0; i < prob.l; i++) {
		map1[cluster[i]]++;
	}
	for(auto it = map1.begin(); it != map1.end(); it++)
		printf("��%.0f���أ���ǩΪ%.0f����ѵ����������Ϊ��%d\n", it->first, k_param.y_c[(int)it->first], it->second);

	printf("************************************\n");

	// ���ÿ����ʵ�ʵ�ѵ����������
	int tNum = 0, misclassified = 0;
	for (int i = 0; i < model->nr_class; i++) {
		tNum = 0;
		misclassified = 0;
		for (int j = 0; j < prob.l; j++) {
			if (prob.r_y[j] == model->label[i]) {
				tNum++;
				if (prob.y[j] != c[j]) misclassified++;
			}
		}
		printf("��ǩΪ%d���࣬��ѵ����������Ϊ��%d�����ֵ���ĿΪ��%d\n", model->label[i], tNum, misclassified);
	}
}

void setTrainPosAndUse()
{
	unordered_map<int, int> posMap;
	for (int i = 0; i < model->nr_class; i++) {
		if (model->pos[i] == 1)
			posMap[model->label[i]] = 1;
		else
			posMap[model->label[i]] = 0;
	}
	for (int i = 0; i < prob.l; i++) {
		if (posMap[(int)prob.r_y[i]] == 1) {
			prob.pos[i] = 1;
			k_prob.pos[i] = 1;
		}
		else {
			prob.pos[i] = 0;
			k_prob.pos[i] = 1;
		}	
	}
	for (int i = 0; i < prob.l; i++) { // ��ʼ����������Ҫʹ�õ�
		prob.use[i] = 1;
	}
}

void initUseCluster()
{
	for (int i = 0; i < k_param.k; i++)
		k_param.use[i] = 1;
}

void initUseAndPosCluster()
{
	// ��ʼ��k_param.use
	for (int i = 0; i < k_param.k; i++)
		k_param.use[i] = 1;
	// ��ʼ��k_param.pos
	for (int i = 0; i < k_param.k; i++) {
		for (int j = 0; j < model->nr_class; j++) {
			if (k_param.y_c[i] == model->label[j]) {
				k_param.pos[i] = model->pos[j];
				break;
			}
		}
	}
	// ��ʼ��ѵ��������prob.use
	for (int i = 0; i < prob.l; i++) {
		if (prob.r_y[i] == 1) {
			prob.pos[i] = 1;
			k_prob.pos[i] = 1;
		}
		else {
			prob.pos[i] = 0;
			k_prob.pos[i] = 0;
		}
	}
	//for (int i = 0; i < k_param.k; i++)
	//	printf("\n��%d�ı�ǩΪ:%d", i, k_param.pos[i]);
}

void mergeCluster()
{
	int usePosCent = 0;
	double avgCenterDis = 0;
	// ��������ؿ���ʹ�õĸ���
	for (int i = 0; i < k_param.k; i++) {
		if (k_param.y_c[i] == 1 && k_param.use[i] == 1) {
			useClusterNum++;
		}
	}
	// ���������֮���ƽ������
	for (int i = 0; i < k_param.k-1; i++) {
		if (k_param.use[i] == 0 || k_param.y_c[i] == -1) continue;
		for (int j = i + 1; j < k_param.k; j++) {
			if (k_param.use[j] == 0 || k_param.y_c[i] == -1) continue;
			avgCenterDis += distance1(k_param.x_c[i], k_param.x_c[j]);
		}
	}
	// �ϲ�С��ƽ������������
	avgCenterDis /= useClusterNum;
	for (int i = 0; i < k_param.k - 1; i++) {
		if (k_param.use[i] == 0 || k_param.y_c[i] == -1) continue;
		for (int j = i + 1; j < k_param.k; j++) {
			if (k_param.use[j] == 0 || k_param.y_c[j] == -1) continue;
			if (distance1(k_param.x_c[i], k_param.x_c[j]) <= avgCenterDis) {
				k_param.use[j] = 0; // ������j����
				for (int k = 0; k < prob.l; k++) { // �ϲ�ѵ������
					if (cluster[k] == j) {
						cluster[k] = i;
						c[k] = k_param.y_c[i];
						count1[i]++;
					}
				}
				for (int k = 0; k < test_data.l; k++) { // �ϲ���������
					if (cluster1[k] == j) {
						cluster1[k] = i;
						c1[k] = k_param.y_c[i];
					}
				}
				useClusterNum--;
				printf("\ncluster:%d merge into cluster:%d", i, j);
			}
		}
	}
}

void deleteNoiseUsingKNN() // ����KNNȥ��������
{
	int kRemove = 3;
	int difNum;
	for (int i = 0; i < prob.l; i++) {
		difNum = 0;
		sortSample temp;
		vector<sortSample> dis;
		for (int j = 0; j < prob.l; j++) {
			if (i == j) continue;
			temp.id = j;
			temp.d = distance1(prob.x[i], prob.x[j]);
			dis.push_back(temp);
		}
		sort(dis.begin(), dis.end(), cmp);
		for (int k = 0; k < kRemove; k++) {
			int id = dis[k].id;
			if (prob.r_y[id] != prob.y[i])
				difNum++;
		}
		if (difNum == kRemove && prob.r_y[i] == -1) {
			prob.use[i] = 0;
			k_prob.use[i] = 0;
			printf("��%d����������������ǩΪ:%.0f\n", i, prob.r_y[i]);
		}
			
	}
}

int getSampleCount()
{
	int posC = 0;
	printf("ѵ�����������ֲ���\n");
	for (int i = 0; i < model->nr_class; i++)
	{
		int t = 0;
		for (int j = 0; j < prob.l; j++)
		{
			if ((int)prob.r_y[j] == model->label[i])
				t++;
		}
		printf("label��%d--->%d\n", model->label[i], t);
		if (model->label[i] == 1) posC = t, trainPosCount = t;
		if (model->label[i] == -1) trainNegCount = t;
	}
	printf("���Լ��������ֲ���\n");
	for (int i = 0; i < model->nr_class; i++)
	{
		int t = 0;
		for (int j = 0; j < test_data.l; j++)
		{
			if ((int)test_data.r_y[j] == model->label[i])
				t++;
		}
		printf("label��%d--->%d\n", model->label[i], t);
	}
	trainIR = trainPosCount * 1.0 / trainNegCount;
	trainPosPro = trainPosCount * 1.0 / prob.l;
	trainNegPro = 1 - trainPosPro;
	return posC;
}

void getPreLabelProp(int flag) // ��ȡ����������������ǩ�Ϸֱ��Ԥ�����
{
	for (int i = 0; i < test_data.l; i++) {
		if (test_data.y[i] == 1) {
			testPosCount++;
			if(flag)
				printf("��%d����������,��ʵ��ǩΪ:%.0f,Ԥ���ǩΪ:%.0f\n", i, test_data.r_y[i], test_data.y[i]);
		}
		if (test_data.y[i] == -1)
			testNegCount++;
	}
	testIR = testPosCount * 1.0 / testNegCount;
	printf("ѵ�����������������%f    �������������������%f\n", trainIR, testIR);
}

double getSVMLabelProp(int flag) // ��ȡ����������������ǩ��ռ���������ı���
{
	int tPos = 0, tNeg = 0;
	for (int i = 0; i < test_data.l; i++) {
		if (test_data.y[i] == 1) {
			tPos++;
			if (flag)
				printf("��%d����������,��ʵ��ǩΪ:%.0f,Ԥ���ǩΪ:%.0f\n", i, test_data.r_y[i], test_data.y[i]);
		}
		if (test_data.y[i] == -1)
			tNeg++;
	}
	return tPos * 1.0 / test_data.l;
}

double getSshKmeansPreLabelProp(int flag) // ��ȡ��ල�����ڲ���������������ǩ�Ϸֱ��Ԥ�����
{
	int testPosCountKmeans = 0, testNegCountKmeans = 0;
	for (int i = 0; i < test_data.l; i++) {
		if (c1[i] == 1) {
			testPosCountKmeans++;
			if (flag)
				printf("��%d����������,��ʵ��ǩΪ:%.0f,Ԥ���ǩΪ:%.0f\n", i, test_data.r_y[i], test_data.y[i]);
		}
		if (c1[i] == -1)
			testNegCountKmeans++;
	}
	double kmeansTestPosIR = testPosCountKmeans * 1.0 / test_data.l;
	return kmeansTestPosIR;
}

void afterChangeLabelByKNN(svm_problem& prob_, int K, int id, int changeLabel) // ����Ԥ�������΢�����ս��  afterChangeLabel(5, i, -1);
{
	int posNum = 0, negNum = 0;
	sortSample temp;
	vector<sortSample> dis;
	for (int i = 0; i < prob_.l; i++) { // �ò���������ѵ�������ľ���
		temp.id = i;
		temp.d = distance1(prob_.x[i], test_data.x[id]);
		dis.push_back(temp);
	}
	sort(dis.begin(), dis.end(), cmp);
	for (int i = 0; i < K; i++) {
		int idx = dis[i].id;
		if (prob.r_y[idx] == 1)
			posNum++;
		if (prob.r_y[idx] == -1)
			negNum++;
		/* ����ʹ��sub_prob����KNN
		int label = sub_prob.r_y[idx];
		int pLabel = 0; // ԭ����ı�ǩ
		for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
			for (double subLabel : it->second) {
				if (subLabel == label) {
					pLabel = it->first;
					break;
				}
			}
			if (pLabel != 0) break;
		}
		if (pLabel == 1)
			posNum++;
		if (pLabel == -1)
			negNum++;
		*/
	}
	if (negNum > posNum && posNum <= 1) {
		test_data.y[id] = changeLabel;
		printf("��%d�����������ı�ǩ����Ϊ:%.0f(����������:%d,������:%d)\n", id, test_data.y[id], posNum, negNum);
	}
}

void SMOTE()
{
	//random_shuffle();
	int id = prob.l;
	int KN = 3;
	int sampleProp = trainNegCount - trainPosCount;
	printf("trainPosCount - trainNegCount = %d\n", sampleProp);
	vector<int> idIsPosVec;
	vector<int> selectId;
	srand((unsigned)time(NULL));
	for (int i = 0; i < prob.l; i++) {
		if (prob.r_y[i] == 1)
			idIsPosVec.push_back(i);
	}
	for (int j = 0; j < sampleProp; j++) {
		int sId = rand() % (idIsPosVec.size() - 0) + 0;
		selectId.push_back(idIsPosVec[sId]); // ע�����ӵ�����ǣ�idIsPosVec[sId]��sIdֻ����idIsPosVec�е�����
	}
	sortSample temp;
	for (int j = 0; j < sampleProp; j++) {
		int curId = selectId[j]; // curId��ʾ��׼������id
		// cout << sub_prob.r_y[curId] << endl;
		vector<sortSample> kNearst;
		for (int k = 0; k < idIsPosVec.size(); k++) {
			if ((idIsPosVec[k] != curId)) { // ���������if (idInCluster[k] != curId) ���������if ((idInCluster[k] != curId) || (idInCluster.size() == 1) || ((idInCluster.size() == 2)))
				temp.id = idIsPosVec[k];
				temp.d = distance1(prob.x[temp.id], prob.x[curId]);
				kNearst.push_back(temp);
			}
		}
		KN = KN > kNearst.size() ? kNearst.size() : KN;
		sort(kNearst.begin(), kNearst.end(), cmp);
		// step3: ���ݹ�ʽ�����µ����������ӵ����ݼ���
		int tId = rand() % (KN - 0) + 0;
		int sId = kNearst[tId].id; // ��ѡ�еĽ���������id
		svm_node *newSample = Malloc(struct svm_node, max_index + 1);
		diminish(prob.x[curId], prob.x[sId]);  // x - x'
		copy(result, newSample);
		double factor = rand() / double(RAND_MAX);
		multiply(factor, newSample);
		add(newSample, prob.x[curId]);
		copy(result, newSample);

		/*int k = 0;
		svm_node *px = newSample;
		while (px[k].index != -1) {
			std::cout << px[k].value << " ";
			k++;
		}
		std::cout << endl;*/

		// ������curId����ѵ�������е�����
		prob.r_y[id] = prob.r_y[curId];
		prob.y[id] = prob.y[curId];
		prob.pos[id] = prob.pos[curId];
		prob.use[id] = prob.use[curId];
		prob.x[id] = newSample;
		// copy(newSample, prob.x[id]);

		id++;
	}
}

void svmOnSmote()
{
	char model_file_smote[100];
	strcpy(model_file_smote, "svm_model_smote.txt");
	initParamDefault(); // ʹ�����gamma,c
	SMOTE();
	if (cross_validation) {
		do_cross_validation();
	}
	else {
		model0 = svm_train(&prob, &param);
		if (svm_save_model(model_file_smote, model0)) {
			fprintf(stderr, "can't save model to file %s\n", model_file_smote);
			exit(1);
		}
	}
	predict1("result.txt", model_file_smote);
}

void afterChangeLabelBySVM(svm_problem& prob_, const char *model_file_name_before, int id, int changeLabel)
{
	//Z = (trainNegCount - trainPosCount) * 1.0 / (trainNegCount + trainPosCount + 2);
	model0 = svm_load_model(model_file_name_before);
	double* prob_estimates = (double *)malloc(model0->nr_class * sizeof(double));
	double predict_label = svm_predict_probability(model0, test_data.x[id], prob_estimates);
	if (predict_label == changeLabel)
		test_data.y[id] = changeLabel;
	//printf("��%d�����������ı�ǩ����Ϊ:%.0f\n", id, test_data.y[id]);
}

void afterChangeLabelBySvmOnSmote(svm_problem& prob_, const char *model_file_name, int id, int changeLabel)
{
	model0 = svm_load_model(model_file_name);
	double* prob_estimates = (double *)malloc(model0->nr_class * sizeof(double));
	double predict_label = svm_predict_probability(model0, test_data.x[id], prob_estimates);
	if (predict_label == changeLabel)
		test_data.y[id] = changeLabel;
	//printf("��%d�����������ı�ǩ����Ϊ:%.0f\n", id, test_data.y[id]);
}

void getFinalAcc()
{
	int correct = 0;
	int total = test_data.l;
	for (int i = 0; i < test_data.l; i++) {
		if (test_data.r_y[i] == test_data.y[i])
			correct++;
	}
	SVM_ACC = (double)correct / total;
	printf("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct / total * 100, correct, total);
}

/*
author:
data: 2020.12.12
˼·�� �Ѵ���Ԥ����ȷ��ѵ������������Ԥ�����ľ���ȥ
�ѵ�1�������ĳЩ���е�ѵ����Ԥ����ȫ���������ݼ�glass 10%/5train.txt
�ѵ�2��ĳЩ�صĲ�����������������
�ѵ�3�� ���صĸ�������Ļ���ģ��̫���ӵ��������½�����

Դ��Ķ�λ�ã�
1.��svm.cpp��svm_group_classes�޸��˲��ִ���
2. ������������־λ
		1).��svm.cpp����svm_group_classes_getPos, �����������ı�ǩpositive
		2).��model�ṹ��������*pos����,svm_problem�ṹ������pos
		3).����model->pos����main.cpp����setTrainPos()���prob.pos
	(Ŀǰѵ�������Լ���ǩ����һ�£���ֱ�Ӹ��ݲ��Լ��ж�С���)
*/

int main(int argc, char **argv)
{
	int i, No;
	int afterDealFlag = 0;
	No = 131;
	tC = 0.5;// 0.25;//0.5;
	tGamma = 0.000488281;// 0.015625;//0.000488281;
	tweight0 = 2.0;
	tweight1 = 1;

	//for(No=2;No<=2;No++) //ʵ��ѭ����No��  
	//{
	char input_file_name[1024];
	char model_file_name_before[1024];	
	char model_file_name_after[1024];
	char cluster_file_name_before[50];
	char cluster_file_name_after[50];
	const char *error_msg;

	// default values
	predict_probability = 1; // �Ƿ�ʹ�ø���Ԥ��
	//initParam();  // �˴�ʹ�����gamma,c��weightc
	initParamDefault(); // ʹ�����gamma,c
	
	strcpy(model_file_name_before,"svm_model_before.txt");
	strcpy(model_file_name_after, "svm_model_after.txt");
	char filename_Notrain[100];
	char filename_Notest[100];
	sprintf(filename_Notrain,"/data/%dtrain.txt",No);
	sprintf(filename_Notest,"/data/%dtest.txt",No);
	sprintf(cluster_file_name_before, "%dcluster_process_before.txt", No);
	sprintf(cluster_file_name_after, "%dcluster_process_after.txt", No);
	//kmean_select_traindata(1000, "data.txt");//���е����������Ƶ�data����
	strcpy(input_file_name, filename_Notrain);//��ѵ�������Ƶ�mytrain����
	read_problem(input_file_name); //����ѵ����

	// ������������
	train_with_noise=false;
	if(train_with_noise)
		set_noise(&prob,&k_prob);

	// ȥ������
	// deleteNoiseUsingKNN();
	
	error_msg = svm_check_parameter(&prob,&param);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	load_test_data(filename_Notest);//�Ѳ��Լ����Ƶ�mytest����

	allocateSpace(); // �����ڴ�
		
	//distance_score = (double *)malloc(100 * sizeof(double));
	//////////SVM����
	if(cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model = svm_train(&prob,&param);
		if(svm_save_model(model_file_name_before,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name_before);
			exit(1);
		}
		//for (i = 0; i < model->nr_class; i++) {
		//	model->pos[i] = posYs[i];
		//}
	}	
	predict("result.txt", model_file_name_before);
	printf("1111\n");
	resultMap[0].push_back(SVM_ACC);
	setTrainPosAndUse(); // Ϊѵ����ÿ������ָʾ�����������Ǹ�����,����ʼ�� prob.use
	printf("1111\n");
	/*for (int i = 0; i < 20; i++) {
		cout << prob.r_y[i] << " : " << prob.pos[i] << endl;
	}*/
	printf("1111\n");
	tAUC_ROC = getAUC_ROC();
	//tAUC_PR = getAUC_PR();
	//double tAuc = 0;
	double tAuc = auc();
	//double tAp = ap();
	resultMap[0].push_back(tAUC_ROC);
	resultMap[0].push_back(tAuc);
	printf("2222\n");
	getPerformance(0); // ��ȡ��ʹ��SVM�����ܲ���
	printf("1111\n");

	int posLableCount = getSampleCount(); // ��ȡѵ�����������������и����������ĸ���
	allocateSpace2();
	oriClassNum = model->nr_class;

	/**************************   ��ֽⲿ��   ********************/
	weight = 0.7;//Ŀ�꺯����׼ȷ�ʵ�Ȩ��
	k_center(&k_prob, class_num);
	// i=kmean_train_new113(No,&k_param,&k_prob);  // ԭ����
	i = kmean_train_new111(No, &k_param, &k_prob); // split5
	// i = kmean_train_new_ori(No, &k_param, &k_prob);
	//if (k_param.k > 20)
	//	i = kmean_train_new_ori(No, &k_param, &k_prob);
	//kmean_predict0(&k_param, &k_prob);//�б�ǩ����������Ԥ��
	//kmean_predict_testdata0(&k_param);//�ޱ�ǩ����������Ԥ��
	initUseAndPosCluster(); // �����������ʼ����ʹ�ñ��use
	getEachClusterAcc(k_param, prob, cluster_file_name_before, No, 0);
	cluster_prepoocess(&k_param); // ����������Ը����ؽ�����������
	//kmean_predict_testdata0(&k_param);
	getEachClusterAcc(k_param, prob, cluster_file_name_after, No, 1);
	// kmean_final_predict(&k_param); // ȷ�����յĴ�֮�󣬼����ල����Բ���������Ԥ�����

	// �����ලkmeans��AUC��F1ָ��
	//tAUC_ROC = getAUC_ROC();
	double kmeans_AUC_ROC = sskmeans_auc(u3); // u1
	// getPerformance(); // ��ȡ��ʹ��SVM�����ܲ���
	setClusterConfusionMatrix(0);
	printf("\nsskmeans��AUC��%f  mF1score: %f\n", kmeans_AUC_ROC, test_f1);
	resultMap[3].push_back(kmeans_AUC_ROC);
	resultMap[3].push_back(test_f1);

	//deleteNegCluster(); // ɾ����С�Ĵ�  ������Ԥ����ȷ������֮����ɾ�����ԣ�
	// mergeCluster(); // �ϲ���С�Ĵ�
	printf("------------------------------------------------------\n");
	// iterative_update2(No);//�������ĵ�������

	// ͳ��ÿ������ѳ��Ĵصĸ���
	//getEachClusterNumAndSampleNum();

	// ����ѵ��������Ԥ��׼ȷ��
	// getEachClusterAcc(k_param, prob); // ��ȡÿ���ص�ѵ��׼ȷ��
	// ��ȡ�����ƽ����С(��������) -- ���������Ԥ����ȷ�������е�������С Ԥ�����ѵ������ֱ������
	//getAvgSubclass(k_param, prob);


	/********************* ���ú��ַ�ʽ����ѵ����  ********************/
	// �ز���--ɸѡѵ������--�����Ӹ����ŶȲ���������ѵ������
	//reSampling();
	// ������ԭʼ�ఴ�����·���
	//createSubProb(); // ����ʹ��ѵ������(�ޱ�ǩ����)
	//createSubProbWithUnlabel(); // ʹ�ô���ѵ�������Ͳ������������Ѵ�����������������ͬһ���
	reSamplingSmote(); //  SMOTE + ɸѡѵ������ + �����Ӹ����ŶȲ���������ѵ������

	// CSSC��������󣬸����ӱ�ǩ(���ر�ǩ)
	for (int i = 0; i < model->nr_class; i++) {
		int num = 1;
		int key = 50;
		for (int j = 0; j < k_param.k; j++) {
			if (k_param.use[j] == 0) continue;
			if (k_param.y_c[j] == model->label[i]) {
				k_param.subclass_y[j] = (double)(model->label[i] * key + num);
				hashMap[k_param.y_c[j]].push_back(k_param.subclass_y[j]);
				num++;
			}
		}
	}
	cout << "���������£�" << endl;
	int idx = 0;
	for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
		cout << it->first << " : ";
		for (double num : it->second) {
			cout << num << " ";
		}
		cout << endl;
		if (it->first == 1) {
			for (double num : it->second)
				posSubLabel[idx++] = num;
		}
	}

	// ����ѵ������ر�ǩ
	for (int i = 0; i < sub_prob.l; i++) {
		if (sub_prob.use[i] == 0) continue;
		int id = sub_cluster[i]; // �����������Ĵ����
		//prob.y[i] = sub_c[i] = k_param.subclass_y[id]; 
		sub_prob.y[i] = sub_prob.r_y[i] = sub_c[i] = k_param.subclass_y[id]; // ѵ�����ı�ǩ���ԸĶ�
		// printf("%d  %.0f\n", id, sub_prob.r_y[i]);
	}
	// ���Ĳ��Լ���ر�ǩ  (�ɲ�����)
	///for (int i = 0; i < test_data.l; i++) {
	///	int id = cluster1[i]; // ssc����������Ԥ���ǩ
	///	c1[i] = k_param.subclass_y[id];
		// test_data.r_y[i] = c1[i] = k_param.subclass_y[id]; // test_data.r_y[i]������������ʵ��ǩ���ܸģ��ĵĻ������ܺ�
	///}

	/*set<int> labelSet;
	for (int i = 0; i < sub_prob.l; i++) {
		labelSet.insert(sub_prob.r_y[i]);
	}
	for (auto it = labelSet.begin(); it != labelSet.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;*/

	// ע��

	// �����ʼ��model->pos
	unordered_map<int, int> tempPos;
	for (int i = 0; i < model->nr_class; i++) {
		tempPos[model->label[i]] = model->pos[i];
	}

	int useCluster = 0;
	for (int i = 0; i < k_param.k; i++) {
		if (k_param.use[i] != 0)
			useCluster++;
	}
	// SVM����
	if (cross_validation) {
		do_cross_validation();
	}
	else {
		// initParamAfterSplit(useCluster);
		initParamAfterSplit2(useCluster);
		Z = 0;
		model = svm_train(&sub_prob, &param);
		if (svm_save_model(model_file_name_after, model)) {
			fprintf(stderr, "can't save model to file %s\n", model_file_name_after);
			exit(1);
		}
	}

	printf("���ĺ��ǩΪ(����%d)��\n", model->nr_class);
	for (int i = 0; i < model->nr_class; i++) {
		cout << model->label[i] << " ";
	}
	cout << endl;
	//final_predict();

	// ���¸����ӱ�ǩ����Ԥ��
	//cout << endl << "�Ƚ�Ԥ��Ľ����";
	//predict1("result.txt", model_file_name);
	cout << endl << "ͶƱԤ��Ľ����";
	proCombination = 0; // �Ƿ�ϲ�pro_extimate����Ԥ���ǩ
	predict_probability = 1; // �ⲿ��ʹ�ø���Ԥ��
	predict2("result.txt", model_file_name_after);
	// resultMap[INDEX].push_back(SVM_ACC);

	// �ϲ�Ԥ����
	// final_predict1();
	// final_predict2();
	final_predict3();

	// ���յ�Ԥ���ǩ����
	getPreLabelProp(0);
	/*
		����������Ԥ��Ϊ������ˣ����ǽ���������תΪ����
			�����ķ�����1.KNN   2.svm
			1.����Ԥ���ǩΪ����Ĳ�������������˲�����������Χ�����Ǹ������������ǽ���תΪ����
			2.����Ԥ���ǩΪ����Ĳ�������������˲�����������Χ�������������������ǽ���תΪ����
	*/
	if (afterDealFlag) {
		svmOnSmote(); // ����ԭʼ���ݼ���SMOTE����SVMԤ��
		tAuc = auc();
		//getPerformance();
		if (testIR - trainIR > 0.1) {
			for (int i = 0; i < test_data.l; i++) {
				if (test_data.y[i] == 1) {
					//afterChangeLabelByKNN(prob, 3, i, -1);
					// afterChangeLabelBySVM(prob, model_file_name_before, i, -1);
					afterChangeLabelBySvmOnSmote(prob, model_file_name_before, i, -1);
				}
			}
			getPreLabelProp(0);
		}
		// ����ĸ����Ԥ��Ϊ������ˣ����ǽ����ָ���תΪ����
		if (trainIR > testIR) { // trainIR - testIR > 0.05
			for (int i = 0; i < test_data.l; i++) {
				if (test_data.y[i] == -1) {
					//afterChangeLabelByKNN(prob, 3, i, 1);
					// afterChangeLabelBySVM(prob, model_file_name_before, i, 1);
					afterChangeLabelBySvmOnSmote(prob, model_file_name_before, i, -1);
				}
			}
			getPreLabelProp(0);
		}
		getPreLabelProp(1);
		getFinalAcc(); // ���������Ԥ���׼ȷ��
	}
	// resultMap[INDEX].push_back(SVM_ACC);

	//cout << endl << "���ʽ��Ԥ��Ľ����";
	//proCombination = 1; // ����ʹ�ø��ʽ�ϵķ�ʽ
	//predict2("result.txt", model_file_name);

	// ���Ĳ��Լ���ر�ǩ
	/*for (int i = 0; i < test_data.l; i++) {
		int id = cluster1[i]; // ssc����������Ԥ���ǩ
		// c1[i] = k_param.subclass_y[id];
		test_data.r_y[i] = c1[i] = k_param.subclass_y[id];
	}*/
	// ����model�ı�ǩ
	int flag;
	unordered_set<int> tLabelSet;
	for (int i = 0; i < model->nr_class; i++) {
		flag = 0;
		for (auto it = hashMap.begin(); it != hashMap.end(); it++) {
			for (int k = 0; k < it->second.size(); k++) {
				if (it->second[k] == model->label[i]) {
					//cout << it->second[k] << " " << it->first << " " << model->label[i] << " ";
					model->label[i] = it->first;
					tLabelSet.insert(model->label[i]);
					//cout <<it->first << " " << model->label[i] << endl;
					flag = 1;
					break;
				}
			}
			if (flag) break;
		}
	}

	int id = 0;
	model->nr_class = oriClassNum;
	for(auto it=tLabelSet.begin(); it != tLabelSet.end(); it++) // ��ԭ��ǩ
		model->label[id++] = *it;
	for (int i = 0; i < model->nr_class; i++)
		model->pos[i] = tempPos[model->label[i]];
	//for (int i = 0; i < model->nr_class; i++)
	//	cout << model->label[i] << " : " << model->pos[i] << endl;
	/*for (int i = 0; i < 10; i++) {
		cout << test_data.y[i] << " " << test_data.r_y[i] << endl;
	}*/
	cout << endl; 
	
	// �������յ�ָ��
	printf("���ս�Ϻ�SVM��ָ��:\n");
	resultMap[2].push_back(SVM_ACC);
	double merge_AUC_ROC = getAUC_ROC();
	//AUC_PR = getAUC_PR();
	tAuc = auc();
	//tAp = ap();
	resultMap[2].push_back(merge_AUC_ROC);
	resultMap[2].push_back(tAuc);
	getPerformance(2);

	// ������ֽ���  ����������ֽ���SVM����Ϊ���ʺ�Ԥ���ǩ�Ὣtest_data.r_y_p��test_data.y����
	printf("��ֽ��SVM��ָ��:\n");
	resultMap[1].push_back(SVM_ACC);
	double deco_AUC_ROC = getAUC_ROC(1);
	tAuc = auc(1);
	resultMap[1].push_back(deco_AUC_ROC);
	resultMap[1].push_back(tAuc);
	getPerformance(1, 1);
	//printf("��ֽ��SVM��AUC-ROC:%lf F1-score:%lf\n", deco_AUC_ROC, *resultMap[1].end());

	printf("\n-------------------------------------\n");
	printf("����������Ϊ��\n\n");
	printf("���մص�����Ϊ��%d\n", k_param.k);
	printf("��ʼSVM��AUC-ROC:%lf  F1-score:%lf\n", tAUC_ROC, *(resultMap[0].end() - 1));
	printf("��ֽ��SVM��AUC-ROC:%lf  F1-score:%lf\n", deco_AUC_ROC, *(resultMap[1].end() - 1));
	printf("��Ϻ�SVM��AUC-ROC:%lf  F1-score:%lf\n", merge_AUC_ROC, *(resultMap[2].end() - 1));
	printf("sskmeans��AUC��%f  mF1score: %f\n", kmeans_AUC_ROC, *(resultMap[3].end() - 1));
	printf("�ش�����(�ز���ǰ)���������%.0f  ���������%.0f\n", resultMap[4][0], resultMap[4][1]);
	printf("�ش�����(�ز���ǰ)������������%.0f  ������������%.0f IR��%lf\n", resultMap[4][2], resultMap[4][3], resultMap[4][4]);
	
	
	// ���д���ļ�
	FILE *fp_svm_result;
	char saveFile[100] = "ans.txt";
	//sprintf(saveFile, "result%d.txt", No);
	fp_svm_result = fopen(saveFile, "a");
	//for (int i = 0; i < test_data.l; i++) {
	//	fprintf(fp_svm_result, "%0.f\t%.0f\t%lf\n", test_data.y[i], test_data.r_y[i], test_data.r_y_p[i]);
	//}
	if (No == 1)
		fprintf(fp_svm_result, "���\t��ʼSVM_ACC\tAUC-1\tAUC-2\tmF-score\t��ֽ��SVM_ACC\tAUC-1\tAUC-2\tmF-score\t��Ϻ�SVM_ACC\tAUC-1\tAUC-2\tmF-score\tSS-kmeans_AUC\tmF-score\t����������\t����������\t����ѵ��������\t����ѵ��������\tIR\n");
	fprintf(fp_svm_result, "%d\t", No);
	for (auto it = resultMap.begin(); it != resultMap.end(); it++) {
		for (double num : it->second) {
			fprintf(fp_svm_result, "%f\t", num);
		}
		fprintf(fp_svm_result, "\t\t");
		//if(it->first == 0)
		//	fprintf(fp_svm_result, "\t\t");
	}

	fprintf(fp_svm_result, "\n");
	if (No % 10 == 0)
		fprintf(fp_svm_result, "\n\n\n");
	fclose(fp_svm_result);

	system("pause");
	return 0;
}






/*cout<<endl;
for(int i=0; i<k_prob.l; i++)
{
	int j = 0, k = 0;
	svm_node *px = k_prob.x[i];
	// while(k_prob.x[i][k].index != -1) // �����ַ�ʽ�����Զ�ȡ�ڲ�������
	while(px[k].index != -1)
	{
		cout<<px[k].value<<" ";
		k++;
	}
	cout<<endl;
}

cout << endl;
	for (int i = 0; i < sub_prob.l; i++)
	{
		int j = 0, k = 0;
		svm_node *px = sub_prob.x[i];
		cout << sub_prob.r_y[i] << ":";
		// while(k_prob.x[i][k].index != -1) // �����ַ�ʽ�����Զ�ȡ�ڲ�������
		while (px[k].index != -1)
		{
			cout << k  << ":" << px[k].value << " ";
			k++;
		}
		cout << endl;
	}


for (int i = 0; i < k_param.k; i++) {
	cout << k_param.y_c[i] << " "  << k_param.subclass_y[i] << endl;
	int j = 0, k = 0;
	svm_node *px = k_param.x_c[i];
	// while(k_prob.x[i][k].index != -1) // �����ַ�ʽ�����Զ�ȡ�ڲ�������
	while (px[k].index != -1)
	{
		cout << px[k].value << " ";
		k++;
	}
	cout << endl;
	}
*/

//  ��ָ֤��Ĳ�������

// 1.��֤����SVM�Ĳ���ָ��
	/*test_data.l = 7;
	int y_true[7] = { 1,1,0,0,1,1,0 };
	double pp[7] = { 0.8,0.7,0.5,0.5,0.5,0.5,0.3 };
	for (int i = 0; i < test_data.l; i++) {
		test_data.r_y[i] = y_true[i];
		test_data.r_y_p[i] = pp[i];
	}
	AUC_ROC = getAUC_ROC();*/

	/*test_data.l = 20;
	int y_true[20] = { 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1 };
	double pp[20] = { 0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11, 0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.1, 0.23, 0.46, 0.08 };
	for (int i = 0; i < test_data.l; i++) {
		test_data.r_y[i] = y_true[i];
		test_data.r_y_p[i] = pp[i];
	}*/

// 2.��֤���յĲ���ָ��
	// double y_true[9] = { 0, 0, 0, 0, 1, 1, 1, 2, 2 };
	// double y_pred[9] = { 0, 0, 1, 2, 1, 1, 2, 1, 2 };
	/*double y_true[6] = { 0, 1, 2, 0, 1, 2 };
	double y_pred[6] = { 0, 2, 1, 0, 0, 1 };
	test_data.l = 6;
	model->nr_class = 3;
	model->label[0] = 0;
	model->label[1] = 1;
	model->label[2] = 2;
	for (int i = 0; i < test_data.l; i++) {
		test_data.r_y[i] = y_true[i];
		test_data.y[i] = y_pred[i];
	}*/

/*
//double y_true[9] = { 0, 0, 0, 0, 1, 1, 1, 2, 2 };
// double y_pred[9] = { 0, 0, 1, 2, 1, 1, 2, 1, 2 };
//double y_true[10] = { -1, -1, 1, -1, 1, 1, -1, 1, -1, 1 };
//double y_pred[10] = { 1, -1, 1, -1, -1, 1, -1, -1, -1, 1 };
double y_true[15] = { 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1 };
double y_pred[15] = { 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 };
//double y_true[15] = { 0, 0, 1, 0, 1, 1, 0, 1, 0, 1 };
//double y_pred[15] = { 1, 0, 1, 0, 0, 1, 0, 0, 0, 1 };
//double y_true[6] = { 0, 1, 2, 0, 1, 2 };
//double y_pred[6] = { 0, 2, 1, 0 ,0, 1 };
test_data.l = 15;
model->nr_class = 2;
model->label[0] = 0;
model->label[1] = 1;
//model->label[2] = 2;
for (int i = 0; i < test_data.l; i++) {
	test_data.r_y[i] = y_true[i];
	test_data.y[i] = y_pred[i];
}
getPerformance();*/