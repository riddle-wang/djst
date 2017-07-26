/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/

   
#ifndef	_MAP_TYPE_H
#define	_MAP_TYPE_H
#include <map>
#include <iostream>
#include <vector>
using namespace std;

typedef map<string, int> MapWord2Id;

typedef map<int, string> MapId2Word;

typedef map<string, int> MapWord2Sentiment;

typedef vector<vector<vector<double> > > SigmaLZW;

#endif
