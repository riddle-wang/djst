/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/


#ifndef	_DATASET_H
#define	_DATASET_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include "constants.h"
#include "document.h"
#include "map_type.h"
using namespace std; 


class dataset {

public:
    dataset();
    explicit dataset(string output_dir);
    ~dataset(void);
    
    static int write_wordmap(string wordmapfile, MapWord2Id &pword2id_new_words,string pattern);
    static int read_wordmap(string wordmapfile, MapId2Word& pid2word);
    static int read_wordmap(string wordmapfile, MapWord2Id& pword2id);
    
    int read_epoch_data(ifstream& fin);
    int generate_wordmap(string input_dir);
    int read_newData(string infer_data, string model_wordmap);
    int read_senti_lexicon(string sentiLexiconFileDir);
    int analyzeCorpus(vector<string>& docs);
    int read_trainfile_list(string filename);
    void deallocate();
    void add_doc(document * doc, int idx);
    void _add_doc(document * doc, int idx);
    
    
	MapId2Word id2word;
	MapWord2Id word2id; // wordmap，dynamic grown

	document ** pdocs; // store training data vocab ID
	document ** _pdocs; // only use for inference, i.e., for storing the new/test vocab ID
    ifstream fin;
	
	string input_dir;
	string output_dir;
	string wordmapfile;
	string positive_lexicon_file;
	string negative_lexicon_file;

	int numDocs;
	int aveDocLength; // average document length
	int vocabSize;
	int corpusSize;
	
	vector<string> docs; // for buffering dataset
	vector<string> newWords;
	vector<string> train_file_list;
	vector<string> positive_lexicon;
	vector<string> negative_lexicon;
};

#endif
