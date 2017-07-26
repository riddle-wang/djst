/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/


#ifndef _INFERENCE_H
#define _INFERENCE_H

#include <sys/stat.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "constants.h"
#include "document.h"
#include "dataset.h"
#include "utils.h"
#include "strtokenizer.h"

using namespace std;


class Inference {
    
public:
    Inference(void);
    ~Inference(void);
    
    int init(int argc, char ** argv);
    int init_inf();
    int inference(); // inference for new (unseen) data based on previously trained model
    int inf_sampling(int m, int n, int& sentiLab, int& topic);
    int init_parameters();
    
    int read_newData(string filename);
    int read_model_setting(string filename);
    int load_model(string model_name);
    int prior2beta(); // for incorporating priro information
    
    // compute model parameters
    void compute_newpi();
    void compute_newtheta();
    int compute_newphi();
    
    // save new data models
    int save_model(string model_name);
    int save_model_newtassign(string filename);
    int save_model_newpi_dl(string filename);
    int save_model_newtheta_dlz(string filename);
    int save_model_newphi_lzw(string filename);
    int save_model_newothers(string filename);
    int save_model_newtwords(string filename);
    
    int numSentiLabs;
    int numTopics;
    int numDocs;      // for trained model
    int vocabSize;    // for trained model
    int newNumDocs;   // for test set
    int newVocabSize; // for test set
    
    vector<vector<vector<int> > > nlzw; // for trained model
    vector<vector<int> > nlz;  // for trained model
    MapWord2Id word2id;
    MapId2Word id2word;
    map<int, int> id2_id;
    map<int, int> _id2id;
    MapWord2Sentiment word2prior; // <string, int> => <word, polarity>
    vector<string> newWords;
    
    string model_dir;
    string model_name;
    string data_dir;
    string datasetFile;
    string result_dir;
    string sentiment_lexicon_dir;
    string wordmapfile;
    string betaFile;
    string positive_lexicon_file;
    string negative_lexicon_file;
    
    string tassign_suffix;
    string pi_suffix;
    string theta_suffix;
    string phi_suffix;
    string others_suffix;
    string twords_suffix;
    
    dataset * pmodelData;	// pointer to trained model object
    dataset * pnewData; // pointer to new/test dataset object
    utils * putils;
    
    int niters;
    int twords;
    int savestep;
    int updateParaStep;
    
    double _alpha;
    double _beta;
    double _gamma;
    
    vector<vector<double> > new_p; // for posterior
    vector<vector<int> > new_z;
    vector<vector<int> > new_l;
    vector<vector<int> > z;  // for trained model
    vector<vector<int> > l;  // for trained model
    vector<string> word_positive;
    vector<string> word_negative;
    
    
    // from NEW/test documents
    vector<int> new_nd;
    vector<vector<int> > new_ndl;
    vector<vector<vector<int> > > new_ndlz;
    vector<vector<vector<int> > > new_nlzw;
    vector<vector<int> > new_nlz;
    
    // hyperparameters
    vector<vector<double> > alpha_lz; // size: (L x T)
    vector<double> alphaSum_l;
    vector<vector<vector<double> > > beta_lzw; // size: (L x T x V)
    vector<vector<double> > betaSum_lz;
    vector<double> gamma_l; // size: (L)
    double gammaSum;
    vector<vector<double> > lambda_lw; // size: (L x V) -- for encoding prior sentiment information
    
    // model parameters
    vector<vector<double> > newpi_dl; // size: (numDocs x L)
    vector<vector<vector<double> > > newtheta_dlz; // size: (numDocs x L x T)
    vector<vector<vector<double> > > newphi_lzw; // size: (L x T x V)
};

#endif
