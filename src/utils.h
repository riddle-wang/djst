/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/

#ifndef _UTILS_H
#define _UTILS_H

#include <string>
#include <algorithm>
#include "dataset.h"
using namespace std;

// for sorting word probabilitys
struct sort_pred {
    bool operator()(const std::pair<int,double> &left, const std::pair<int,double> &right) {
	    return left.second > right.second;
    }
};

class model;
class Inference;

class utils {
	
public:
	utils();
		
    // parse command line arguments
    int parse_args(int argc, char ** argv, int&  model_status);
  	int parse_args_est(int argc, char ** argv, model * pmodel);
	int parse_args_inf(int argc, char ** argv, Inference * pmodel_inf);
    
    // read configuration file
	int read_config_file(string configfile);
  
    // generate the model name for the current iteration
    string generate_model_name(int epoch ,int iter);
    string generate_infer_name(int iter);

    // make directory
    int make_dir(string strPath);
    
    // sort    
    void sort(vector<double> & probs, vector<int> & words);

private:
    int model_status;
    string input_dir;
    string output_dir;
    string wordmapfile;
    string sentiment_lexicon_dir;
    string configfile;
    
    int numSentiLabs;
    int numTopics;
    int niters;
    int savestep;
    int twords;
    int updateParaStep;
    double alpha;
    double beta;
    double gamma;
    int max_epochs;
    string train_file_list;
    string positive_lexicon_file;
    string negative_lexicon_file;
    int S;
    
    // inference
    string model_wordmap;
    string model_name;
    string model_dir;
    string infer_data;
};

#endif

