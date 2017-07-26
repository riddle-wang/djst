/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/

   
#ifndef	_DOCUMENT_H
#define	_DOCUMENT_H

#include <vector>
#include <iostream>
using namespace std; 



class document {

public:
	
	document() {
		words = NULL;
		priorSentiLabels = NULL;
		docID = "";
		rawstr = "";
		length = 0;	
	}
    
	// Constructor. Retrieve the length of the document and allocate memory for storing the documents
    explicit document(int length) {
		this->length = length;
		docID = "";
		rawstr = "";
		words = new int[length]; // words stores the word token ID, which is integer
		priorSentiLabels = new int[length];	
    }
    
    // Constructor. Retrieve the length of the document and store the element of words into the array
    document(int length, int * words) {
		this->length = length;
		docID = "";
		rawstr = "";
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = words[i];
		}
		priorSentiLabels = new int[length];	
    }

    document(int length, int * words, string rawstr) {
		this->length = length;
		docID = "";
		this->rawstr = rawstr;
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			 this->words[i] = words[i];
		}
		priorSentiLabels = new int[length];
    }
    

    explicit document(vector<int> & doc) {
		this->length = doc.size();
		docID = "";
		rawstr = "";
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = doc[i];
		}
		priorSentiLabels = new int[length];	
    }


	document(vector<int> & doc, string rawstr) {
		this->length = doc.size();
		docID = "";
		this->rawstr = rawstr;
		this->words = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = doc[i];
		}
		priorSentiLabels = new int[length];
	}

    document(vector<int> & doc, vector<int> &priorSentiLab, string rawstr) {
		this->length = doc.size();
		docID = "";
		this->rawstr = rawstr;
		this->words = new int[length];
		this->priorSentiLabels = new int[length];
		for (int i = 0; i < length; i++) {
			this->words[i] = doc[i];
			this->priorSentiLabels[i] = priorSentiLab[i];
		}
    }
    
    ~document() {
		if (NULL != words) {
			delete [] words;
			words = NULL;
		}
			
		if (priorSentiLabels != NULL) {
			delete [] priorSentiLabels;
			priorSentiLabels = NULL;
		}
    }
    
    int * words;
    int * priorSentiLabels;
    string docID;
    string rawstr;
    int length;
};

#endif
