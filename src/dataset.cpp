/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/

#include "dataset.h"
#include <stdlib.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include "document.h"
#include "model.h"
#include "map_type.h"
#include "strtokenizer.h"
using namespace std; 

dataset::dataset() {
	pdocs = NULL;
	_pdocs = NULL;
	output_dir = ".";
	wordmapfile = "wordmap.txt";

	numDocs = 0;
	aveDocLength = 0;
	vocabSize = 0;
	corpusSize = 0;
}

dataset::dataset(string output_dir) {
	pdocs = NULL;
	_pdocs = NULL;
	this->output_dir = output_dir;
	wordmapfile = "wordmap.txt";

	numDocs = 0; 
	aveDocLength = 0;
	vocabSize = 0; 
	corpusSize = 0;
}


dataset::~dataset(void) {
	deallocate();
}

int dataset::write_wordmap(string wordmapfile, MapWord2Id &pword2id, string pattern) {
    FILE * fout = fopen(wordmapfile.c_str(), pattern.c_str());
    if (!fout) {
        printf("Cannot open wordmapfile %s to write!\n", wordmapfile.c_str());
        return 1;
    }
    MapWord2Id::iterator it;
    for (it = pword2id.begin(); it != pword2id.end(); it++) {
        fprintf(fout, "%s %d\n", (it->first).c_str(), it->second);
    }
    fclose(fout);
    return 0;
}


int dataset::read_wordmap(string wordmapfile, MapId2Word &pid2word) {
    pid2word.clear();
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
        printf("Cannot open file %s to read!\n", wordmapfile.c_str());
        return 1;
    }
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
        fgets(buff, BUFF_SIZE_SHORT - 1, fin);
        line = buff;
        strtokenizer strtok(line, " \t\r\n");
        if (strtok.count_tokens() != 2) {
            printf("Warning! Line %d in %s contains less than 2 words!\n", i+1, wordmapfile.c_str());
            continue;
        }
        
        pid2word.insert(pair<int, string>(atoi(strtok.token(1).c_str()), strtok.token(0)));
    }
    
    fclose(fin);
    return 0;
}


int dataset::read_wordmap(string wordmapfile, MapWord2Id& pword2id) {
    pword2id.clear();
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
        printf("Cannot read file %s!\n", wordmapfile.c_str());
        return 1;
    }
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
        fgets(buff, BUFF_SIZE_SHORT - 1, fin);
        line = buff;
        strtokenizer strtok(line, " \t\r\n");
        if (strtok.count_tokens() != 2) {
            continue;
        }
        pword2id.insert(pair<string, int>(strtok.token(0), atoi(strtok.token(1).c_str())));
    }
    
    fclose(fin);
    return 0;
}


int dataset::read_epoch_data(ifstream& fin) {
	string line;
	char buff[BUFF_SIZE_LONG];
	docs.clear();
	numDocs = 0;
    
	while (fin.getline(buff, BUFF_SIZE_LONG)) {
		line = buff;
		if(!line.empty()) {
			docs.push_back(line);
			numDocs++;
		}
	}
    
	if (numDocs > 0) {
		this->analyzeCorpus(docs);
	}
	
	return 0;
}

int dataset::generate_wordmap(string input_dir) {
    string line;
    char buff[BUFF_SIZE_LONG];
    MapWord2Id::iterator word2id_it;
    MapId2Word::iterator id2word_it;
    for (int i = 0; i < train_file_list.size(); i++ ) {
        fin.open( (input_dir + train_file_list[i]).c_str(), ifstream::in );
        if (!fin) return 1;
        while (fin.getline(buff, BUFF_SIZE_LONG)) {
            line = buff;
            if (!line.empty()) {
                strtokenizer strtok(line," \t\r\n");
                int docLength = strtok.count_tokens();
                if (docLength <= 0) {
                    printf("Invalid (empty) document!\n");
                    return 1;
                }
                for (int k = 0; k < docLength - 1; k++) {
                    word2id_it = word2id.find(strtok.token(k+1).c_str());
                    if (word2id_it == word2id.end()) {
                        int word_id = (int)word2id.size()+1;
                        word2id.insert(pair<string, int>(strtok.token(k+1), word_id));
                        id2word.insert(pair<int, string>(word_id, strtok.token(k+1)));
                    }
                }
            }
        }
        fin.close();
    }
    vocabSize = (int)word2id.size();
    if (write_wordmap(output_dir + wordmapfile, word2id, "w")) {
        printf("ERROE! Can not write wordmap file %s!\n", wordmapfile.c_str());
        return 1;
    }
    return 0;
}

int dataset::read_newData(string dfile, string model_wordmap) {
   
    
    return 0;
}

int dataset::read_senti_lexicon(string sentiment_lexicon_dir) {
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    string positive_lexicon_path = sentiment_lexicon_dir + "positive.txt";
    string negative_lexicon_path = sentiment_lexicon_dir + "negative.txt";
    
    FILE * fin_pos = fopen(positive_lexicon_path.c_str(), "r");
    FILE * fin_neg = fopen(negative_lexicon_path.c_str(), "r");
    
    if (!fin_pos || !fin_neg) {
        printf("Cannot read sentiment lexicon file \n");
        return 1;
    }
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin_pos) != NULL) {
        line = buff;
        positive_lexicon.push_back(line.substr(0,line.size()-1));
    }
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin_neg) != NULL) {
        line = buff;
        negative_lexicon.push_back(line.substr(0,line.size()-1));
    }
    
    if (positive_lexicon.size() == 0 || negative_lexicon.size() == 0) {
        printf("Sentiment Lexicon is empty. \n");
        return 1;
    }
    
    fclose(fin_pos);
    fclose(fin_neg);
    
    return 0;
}

int dataset::analyzeCorpus(vector<string>& docs) {
	MapId2Word::iterator id2word_it;
	MapWord2Id::iterator word2id_it;
	string line;
	numDocs = (int)docs.size();
	corpusSize = 0;
	aveDocLength = 0;

  	// allocate memory for corpus
	if (pdocs != NULL ) {
		deallocate();
		pdocs = new document*[numDocs];
    }
    else {
		pdocs = new document*[numDocs];
	}
    
	for (int i = 0; i < (int)docs.size(); ++i) {
		line = docs.at(i);
		strtokenizer strtok(line, " \t\r\n");
		int docLength = strtok.count_tokens();
	
		if (docLength <= 0) {
			printf("Invalid (empty) document!\n");
			deallocate();
			numDocs = vocabSize = 0;
			return 1;
		}
	
		corpusSize += docLength - 1; // the first word is document id
		
		// allocate memory for the new document_i 
		document * pdoc = new document(docLength-1);
		pdoc->docID = strtok.token(0).c_str();
		
		// generate ID for the tokens in the corpus, and assign each word token with the corresponding vocabulary ID.
		for (int k = 0; k < docLength-1; k++) {
			word2id_it = word2id.find(strtok.token(k+1).c_str());
			if (word2id_it == word2id.end()) {
                printf("Can not find the word %s in wordmapfile.\n",strtok.token(k+1).c_str() );
                return 1;
			}
            else {
				pdoc->words[k] = word2id_it->second;
			}
		}
		add_doc(pdoc, i);
	}
	aveDocLength = corpusSize/numDocs;
	docs.clear();
	return 0;
}

int dataset::read_trainfile_list(string filename) {
    ifstream fin;
	fin.open(filename.c_str(), ifstream::in);
	string line;
	char buff[BUFF_SIZE_LONG];
	train_file_list.clear();
	while (fin.getline(buff, BUFF_SIZE_LONG)) {
		line = buff;
		if (!line.empty()) {
			train_file_list.push_back(line);
		}
	}
	if (train_file_list.size()>0) {
        return 0;
	}
    else {
		printf("Train file list is empty with path = %s \n",filename.c_str());
		return 1;
	}
}

void dataset::deallocate() {
	if (pdocs != NULL ) {
		delete [] pdocs;
		pdocs = NULL;
	}
	if (_pdocs) {
		for (int i = 0; i < numDocs; i++) {
			delete _pdocs[i];
		}
		delete [] _pdocs;
		_pdocs = NULL;
	}
}
    

void dataset::add_doc(document * doc, int idx) {
    if (0 <= idx && idx < numDocs)
        pdocs[idx] = doc;
}   

void dataset::_add_doc(document * doc, int idx) {
    if (0 <= idx && idx < numDocs) {
	    _pdocs[idx] = doc;
    }
}
