/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/

#include "utils.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include "strtokenizer.h"
#include "model.h"
#include "inference.h"
#include "dataset.h"

using namespace std;

#undef WINDOWS
#ifdef _WIN32
    #define WINDOWS
#endif
#ifdef __WIN32__
    #define WINDOWS
#endif

#ifdef WINDOWS
	#include <direct.h>  // For _mkdir().
	#include <io.h>      // For access().
#else 
	#include <unistd.h>  // For access().
#endif


utils::utils() {
	model_status = MODEL_STATUS_UNKNOWN;
	input_dir = "";
	output_dir = "";
	configfile = "";

	wordmapfile = "";
	sentiment_lexicon_dir = "";
    configfile = "";
    numSentiLabs = 0;
	numTopics = 0;
    niters = 0;
    savestep = 0;
    twords = 0;
	updateParaStep = -1; 

	alpha = -1.0;
	beta = -1.0;
    gamma = -1.0;

    max_epochs = 0;
    train_file_list = "";
    
    // inference
    model_wordmap = "";
    model_name = "";
    model_dir = "";
    infer_data = "";
}


int utils::parse_args(int argc, char ** argv, int&  model_status) {
    int i = 1;
    while (i < argc) {
		string arg = argv[i];
		if (arg == "-est") {
			model_status = MODEL_STATUS_EST;
			break;
		}
		else if (arg == "-estc") {
			model_status = MODEL_STATUS_ESTC;
			break;
		}
		else if (arg == "-inf") {
			model_status = MODEL_STATUS_INF;
			break;
		}
		i++;
	}

    this->model_status = model_status;
    printf("model_status = %d\n", this->model_status);
	return (model_status);
}



int utils::parse_args_est(int argc, char ** argv, model * pmodel) {

    int i = 1;
    while (i < argc) {
	    string arg = argv[i];
		if (arg == "-config") {
			configfile = argv[++i];
			break;
		}
		i++;
	}

	if (configfile != "") {
		if (read_config_file(configfile)) {
			return 1;
		}
	}
	
	if (wordmapfile != "") pmodel->wordmapfile = wordmapfile;
	if (sentiment_lexicon_dir != "") pmodel->sentiment_lexicon_dir = sentiment_lexicon_dir;
	if (train_file_list != "") pmodel->train_file_list = train_file_list;
	if (positive_lexicon_file != "" ) pmodel->positive_lexicon_file = positive_lexicon_file;
	if (negative_lexicon_file != "" ) pmodel->negative_lexicon_file = negative_lexicon_file;
	if (numSentiLabs > 0) pmodel->numSentiLabs = numSentiLabs;
	if (numTopics > 0) pmodel->numTopics = numTopics;
	if (niters > 0) pmodel->niters = niters;
	if (savestep > 0) pmodel->savestep = savestep;
	if (twords > 0) pmodel->twords = twords;
	if (max_epochs > 0) pmodel->max_epochs = max_epochs;
	if ( S > 0 ) pmodel->S = S;
	if (alpha > 0.0) pmodel->_alpha = alpha;
	if (beta > 0.0) pmodel->_beta = beta;
	if (gamma > 0.0) pmodel->_gamma = gamma;
    
    pmodel->updateParaStep = updateParaStep; // -1: no parameter optimization

	if (input_dir != "")	{
		if (input_dir[input_dir.size() - 1] != '/') {
			input_dir += "/";
		}
		pmodel->input_dir = input_dir;
	}
	else {
		printf("Please specify input data dir!\n");
		return 1;
	}
	
	if (output_dir != ""){
	    if (make_dir(output_dir)) return 1;
	    if (output_dir[output_dir.size() - 1] != '/') {
		    output_dir += "/";
	    }
		pmodel->output_dir = output_dir;
	}
	else {
		printf("Please specify output dir!\n");
		return 1;
	}

    return 0;
}
   

int utils::parse_args_inf(int argc, char ** argv, Inference * pmodel_inf) {

	int i = 1; 
	while (i < argc) {
		string arg = argv[i];
	    printf("arg=%s\n", arg.c_str());
		if (arg == "-config") {
			configfile = argv[++i];
			break;
		}
		i++;
	}
	if (configfile != "") {
		if (read_config_file(configfile)) return 1;
	}
    
    if (model_wordmap != "") pmodel_inf->wordmapfile = model_wordmap;
		
    if (sentiment_lexicon_dir != "") pmodel_inf->sentiment_lexicon_dir = sentiment_lexicon_dir;
	
	if (model_dir != "")	{
		if (model_dir[model_dir.size() - 1] != '/') model_dir += "/";
		pmodel_inf->model_dir = model_dir;
	}
    if (positive_lexicon_file != "" ) pmodel_inf->positive_lexicon_file = positive_lexicon_file;
    if (negative_lexicon_file != "" ) pmodel_inf->negative_lexicon_file = negative_lexicon_file;
	if (input_dir != "") {
		if (input_dir[input_dir.size() - 1] != '/') input_dir += "/";
		pmodel_inf->data_dir = input_dir;
	}
	else {
		printf("Please specify input data dir!\n");
		return 1;
	}
	
	if (output_dir != "") {
		if (make_dir(output_dir)) return 1;
		if (output_dir[output_dir.size() - 1] != '/') output_dir += "/";
		pmodel_inf->result_dir = output_dir;
	}
	else {
		printf("Please specify output dir!\n");
		return 1;
	}
	
    if (model_name != "") {
		pmodel_inf->model_name = model_name;
    }
	else {
		printf("Please specify the trained dJST model name!\n");
		return 1;
	}
    if (infer_data != "") pmodel_inf->datasetFile = infer_data;
	if (niters > 0) pmodel_inf->niters = niters;
	if (twords > 0) pmodel_inf->twords = twords;
	if (savestep > 0) pmodel_inf->savestep = savestep;
	if (updateParaStep > 0) pmodel_inf->updateParaStep = updateParaStep;
	if (alpha > 0.0) pmodel_inf->_alpha = alpha;
	if (beta > 0.0) pmodel_inf->_beta = beta;
	if (gamma > 0.0) pmodel_inf->_gamma = gamma;
    return 0;
}
   

int utils::read_config_file(string filename) {
	char buff[BUFF_SIZE_SHORT];
    string line;
	FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
		printf("Cannot read file %s\n", filename.c_str());
		return 1;
    }
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin)) {
        line = buff;
        strtokenizer strtok(line, "= \t\r\n");
        int count = strtok.count_tokens();
        
        if (count != 2) continue;
        string optstr = strtok.token(0);
        string optval = strtok.token(1);
        
        if (optstr == "nsentiLabs") {
            numSentiLabs = atoi(optval.c_str());
        }
        else if (optstr == "ntopics") {
            numTopics = atoi(optval.c_str());
        }
        else if (optstr == "niters"){
            niters = atoi(optval.c_str());
        }
        else if (optstr == "savestep") {
            savestep = atoi(optval.c_str());
        }
        else if (optstr == "updateParaStep") {
            updateParaStep = atoi(optval.c_str());
        }
        else if (optstr == "twords") {
            twords = atoi(optval.c_str());
        }
        else if (optstr == "input_dir") {
            input_dir = optval;
        }
        else if (optstr == "model_dir") {
            model_dir = optval;
        }
        else if (optstr == "output_dir") {
            output_dir = optval;
        }
        else if (optstr == "sentiment_lexicon_dir") {
            sentiment_lexicon_dir = optval;
        }
        else if (optstr == "model_wordmap") {
            model_wordmap = optval;
        }
        else if (optstr == "alpha") {
            alpha = atof(optval.c_str());
        }
        else if (optstr == "beta") {
            beta = atof(optval.c_str());
        }
        else if (optstr == "gamma") {
            gamma = atof(optval.c_str());
        }
        else if (optstr == "model") {
            model_name = optval;
        }
        else if (optstr == "max_epochs") {
            max_epochs = atoi(optval.c_str());
        }
        else if (optstr == "train_file_list" ) {
            train_file_list = optval;
        }
        else if (optstr == "positive_lexicon") {
            positive_lexicon_file = optval;
        }
        else if (optstr == "negative_lexicon") {
            negative_lexicon_file = optval;
        }
        else if (optstr == "S") {
            S = atoi(optval.c_str());
        }
        else if (optstr == "datasetFile") {
            infer_data = optval;
        }
    }
    fclose(fin);
    return 0;
}


string utils::generate_model_name(int epoch, int iter)  {

	string model_name;
	std::stringstream out;
	char buff[BUFF_SIZE_SHORT];
	
	// sprintf(buff, "%05d", iter);
	sprintf(buff, "t%d-iter%d", epoch ,iter);

	if (iter >= 0){
		model_name = buff;
	}
	else{
		sprintf(buff, "t%d-final", epoch);
		model_name = buff;
	}
	
	return model_name;
}

string utils::generate_infer_name(int iter)  {
    
    string model_name;
    std::stringstream out;
    char buff[BUFF_SIZE_SHORT];
    
    // sprintf(buff, "%05d", iter);
    sprintf(buff, "inf_iter%d",iter);
    
    if (iter >= 0){
        model_name = buff;
    }
    else{
        sprintf(buff, "inf-final");
        model_name = buff;
    }
    
    return model_name;
}

#ifdef WINDOWS
int utils::make_dir(string strPath) {
    if (_access(strPath.c_str(), 0) == 0) {
		return 0;
    }
    else if (_mkdir(strPath.c_str()) == 0) {
		return 0;
    }
	else {
		printf("Throw exception in creating directory %s !\n",strPath.c_str());
		return 1;
	} 
}
#else
int utils::make_dir(string strPath) {
    if (access(strPath.c_str(), 0) == 0) {
		return 0;
    }
    else if (mkdir(strPath.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
		return 0;
    }
	else {
        printf("Throw exception in creating directory: %s\n",strPath.c_str());
		return 1; 
	}
}
#endif
