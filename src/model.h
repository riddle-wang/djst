/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/


#ifndef	_MODEL_H
#define	_MODEL_H

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "dataset.h"
#include "document.h"
#include "map_type.h"
#include "utils.h"
#include "math_func.h"
#include "polya_fit_simple.h"
#include "strtokenizer.h"
using namespace std;

class model {
public:
    
    model(void);
    ~model(void);
    int init(int argc, char ** argv);
    int djst_estimate();
    
	string input_dir;
	string output_dir;
	string sentiment_lexicon_dir;
	string wordmapfile;
	string tassign_suffix;
	string pi_suffix;
	string theta_suffix;
	string phi_suffix;
	string others_suffix;
	string twords_suffix;
    string train_file_list;
    string positive_lexicon_file;
    string negative_lexicon_file;

	int numTopics;
	int numSentiLabs; 
	int niters;
	int liter;
	int twords;
	int savestep;
	int updateParaStep;
    int max_epochs;
    int S;
	
    double _alpha;
	double _beta;
	double _gamma;
    float smooth_v;

	MapWord2Id word2id;
    MapId2Word id2word;
private:
    // int init_model_parameters();
    inline int delete_model_parameters() {
        numDocs = 0;
        vocabSize = 0;
        corpusSize = 0;
        aveDocLength = 0;
        
        if (pdataset != NULL) {
            delete pdataset;
            pdataset = NULL;
        }
        
        return 0;
    }
    
    int set_gamma();
    
    // compute parameter functions
    void compute_phi_lzw();
    void compute_pi_dl();
    void compute_theta_dlz();
    
    // save model parameter funtions
    int save_model(string model_name);
    int save_model_tassign(string filename);
    int save_model_twords(string filename);
    int save_model_pi_dl(string filename);
    int save_model_theta_dlz(string filename);
    int save_model_phi_lzw(string filename);
    int save_model_others(string filename);
    
    int sampling(int m, int n, int& sentiLab, int& topic);
    int djst_initial_beta_1();
    int djst_initial_beta_2();
    int djst_initial_others();
    int update_E();
    int djst_initial_general();
    int update_Parameters();
    
    // ---------- Variales ----------
	int numDocs;
	int vocabSize;
	int corpusSize;
	int aveDocLength;
	
	ifstream fin;	
	dataset * pdataset;
	utils * putils;

	// model counts
	vector<int> nd;
	vector<vector<int> > ndl;
	vector<vector<vector<int> > > ndlz;
	vector<vector<vector<int> > > nlzw;
	vector<vector<int> > nlz;
	
	// topic and label assignments
	vector<vector<double> > p;
	vector<vector<int> > z;
	vector<vector<int> > l;
	
	// model parameters
	vector<vector<double> > pi_dl; // size: (numDocs x L)
	vector<vector<vector<double> > > theta_dlz; // size: (numDocs x L x T) 
	vector<vector<vector<double> > > phi_lzw; // size: (L x T x V)
	
	// hyperparameters 
	vector<vector<double> > alpha_lz; // \alpha_tlz size: (L x T)
	vector<double> alphaSum_l; 
	vector<vector<vector<double> > > beta_lzw; // size: (L x T x V)
	vector<vector<double> > betaSum_lz;
	vector<vector<double> > gamma_dl; // size: (numDocs x L)
	vector<double> gammaSum_d; 
	vector<vector<double> > lambda_lw; // size: (L x V) -- for encoding prior sentiment information 
		
	vector<vector<double> > opt_alpha_lz;  //optimal value, size:(L x T) -- for storing the optimal value of alpha_lz after fix point iteration
	
    vector<string> word_positive;
    vector<string> word_negative;
    vector<SigmaLZW> E_tlzw;// Size: TimeSlice * Label * Topic * Word
    vector<double> miu_t;
    vector<vector<vector<double> > > clzw;
    vector<vector<double> > clz;
    SigmaLZW sigma_tmp;
};
#endif
