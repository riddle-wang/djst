/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/


#include "model.h"
using namespace std;


model::model(void) {
    input_dir = "";
    output_dir = "";
    sentiment_lexicon_dir = "";
	wordmapfile = "wordmap.txt";
	tassign_suffix = ".tassign";
	pi_suffix = ".pi";
	theta_suffix = ".theta";
	phi_suffix = ".phi";
	others_suffix = ".others";
	twords_suffix = ".twords";
    train_file_list = "";
    positive_lexicon_file = "";
    negative_lexicon_file = "";
	
	numTopics = 50;
	numSentiLabs = 3;
    niters = 1000;
    liter = 0;
    twords = 5;
    savestep = 200;
    updateParaStep = 40;
    max_epochs = 0;
    S = 1;
    _alpha  = -1.0;
    _beta = -1.0;
    _gamma = -1.0;
    smooth_v = 0.0;
    
	vocabSize = 0;
	numDocs = 0;
	corpusSize = 0;
	aveDocLength = 0;
	
	putils = new utils();
}


model::~model(void) {
	if (putils) delete putils;
}


int model::init(int argc, char ** argv) {

    if (putils->parse_args_est(argc, argv, this)) {
        return 1;
    }
    
    printf("\n---------- DJST initial info ----------\n\n");
    printf("input_dir = %s\n",input_dir.c_str());
    printf("output_dir = %s\n",output_dir.c_str());
    printf("sentiment_lexicon_dir = %s\n",sentiment_lexicon_dir.c_str());
    printf("wordmapfile = %s\n",wordmapfile.c_str());
    
    printf("numTopics = %d\n",numTopics);
    printf("numSentiLabs = %d\n",numSentiLabs);
    printf("niters = %d\n",niters);
    printf("savestep = %d\n",savestep);
    printf("twords = %d\n",twords);
    printf("updateParaStep = %d\n",updateParaStep);
    
    printf("_alpha = %f\n",_alpha);
    printf("_beta = %f\n",_beta);
    printf("_gamma = %f\n",_gamma);
    
    printf("max_epochs = %d\n",max_epochs);
    printf("train_file_list = %s\n",train_file_list.c_str());
    printf("positive_lexicon_file = %s\n",positive_lexicon_file.c_str());
    printf("negative_lexicon_file = %s\n",negative_lexicon_file.c_str());
    printf("S = %d\n",S);
	return 0;
}

int model::djst_estimate(){
    pdataset = new dataset(output_dir);
    int sentiLab, topic;
    if (sentiment_lexicon_dir != "") {
        if (pdataset->read_senti_lexicon((sentiment_lexicon_dir).c_str())) {
            printf("Error! Cannot read sentiment_lexicon_dir %s\n", (sentiment_lexicon_dir).c_str());
            delete pdataset;
            return 1;
        }
        this->word_positive = pdataset->positive_lexicon;
        this->word_negative = pdataset->negative_lexicon;
    }
    
    pdataset->read_trainfile_list(this->input_dir + this->train_file_list);
    pdataset->generate_wordmap(this->input_dir);
    
    djst_initial_general();
    for (int t = 1; t <= max_epochs; t++ ) {
        if ( max_epochs != pdataset->train_file_list.size()) {
            printf("Lack of sufficient epoches data. \n");
            return 1;
        }
        fin.open((this->input_dir +pdataset->train_file_list[t-1]).c_str(), ifstream::in);
        if (!fin) return 1;
        if (pdataset->read_epoch_data(fin)) {
            printf("Throw exception in function read_epoch_data()! \n");
            delete pdataset;
            return 1;
        }
        fin.close();
        
        numDocs = pdataset->numDocs;
        corpusSize = pdataset->corpusSize;
        aveDocLength = pdataset->aveDocLength;
        
        if (t == 1) {
            djst_initial_beta_1();
        }
        else {
            djst_initial_beta_2();
        }
        djst_initial_others();
        for (liter = 1; liter <= niters; liter++) {
            printf("Epochs %d, Iteration %d ...\n", t, liter);
            for (int m = 0; m < numDocs; m++) {
                for (int n = 0; n < pdataset->pdocs[m]->length; n++) {
                    sampling(m, n, sentiLab, topic);
                    l[m][n] = sentiLab;
                    z[m][n] = topic;
                }
            }
            // Update alpha, miu, beta
            if (updateParaStep > 0 && liter % updateParaStep == 0) {
                printf("Update parameters at Epochs %d iteration %d ...\n", t ,liter);
                this->update_Parameters();
            }
            // Update pi, theta, phi
            if (savestep > 0 && liter % savestep == 0) {
                if (liter == niters) break;
                printf("Saving the model at Epochs %d iteration %d ...\n", t ,liter);
                compute_pi_dl();
                compute_theta_dlz();
                compute_phi_lzw();
                save_model(putils->generate_model_name(t,liter));
            }
        }//End_for_liter
        compute_pi_dl();
        compute_theta_dlz();
        compute_phi_lzw();
        save_model(putils->generate_model_name(t,-1));
        update_E();
    }//end_for_t_epoches
    return 0;
}//end_method_djst_estimate

// Document level sentiment semmetric prior
int model::set_gamma() {

	if (_gamma <= 0 ) {
		_gamma = (double)aveDocLength * 0.05 / (double)numSentiLabs;
	}

	gamma_dl.resize(numDocs);
	gammaSum_d.resize(numDocs);

	for (int d = 0; d < numDocs; d++) {
		gamma_dl[d].resize(numSentiLabs);
		gammaSum_d[d] = 0.0;
		for (int l = 0; l < numSentiLabs; l++) {
			gamma_dl[d][l] = _gamma;
			gammaSum_d[d] += gamma_dl[d][l];
		}
	}

	return 0;
}

void model::compute_phi_lzw() {
	for (int l = 0; l < numSentiLabs; l++) {
	    for (int z = 0; z < numTopics; z++) {
			for (int r = 0; r < vocabSize; r++) {
				phi_lzw[l][z][r] = (nlzw[l][z][r] + 1 + beta_lzw[l][z][r]) / (nlz[l][z] + this->vocabSize + betaSum_lz[l][z]);
			}
		}
	}
}


void model::compute_pi_dl() {
	for (int m = 0; m < numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++) {
		    pi_dl[m][l] = (ndl[m][l] + gamma_dl[m][l]) / (nd[m] + gammaSum_d[m]);
		}
	}
}

void model::compute_theta_dlz() {
	for (int m = 0; m < numDocs; m++) {
	    for (int l = 0; l < numSentiLabs; l++)  {
			for (int z = 0; z < numTopics; z++) {
			    theta_dlz[m][l][z] = (ndlz[m][l][z] + 1 + alpha_lz[l][z]) / (ndl[m][l] + this->numTopics + alphaSum_l[l]);
			}
		}
	}
}


int model::save_model(string model_name) {
	if (save_model_tassign(output_dir + model_name + tassign_suffix)) return 1;
	if (save_model_twords(output_dir + model_name + twords_suffix)) return 1;
	if (save_model_pi_dl(output_dir + model_name + pi_suffix)) return 1;
	if (save_model_theta_dlz(output_dir + model_name + theta_suffix)) return 1;
	if (save_model_phi_lzw(output_dir + model_name + phi_suffix)) return 1;
	if (save_model_others(output_dir + model_name + others_suffix)) return 1;
	return 0;
}


int model::save_model_tassign(string filename) {

    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }

	for (int m = 0; m < pdataset->numDocs; m++) {
        if (pdataset->pdocs == NULL ) {
            printf("pdataset->pdocs is null.\n");
            return 1;
        }
		fprintf(fout, "%s \n", pdataset->pdocs[m]->docID.c_str());
		for (int n = 0; n < pdataset->pdocs[m]->length; n++) {
	        fprintf(fout, "%d:%d:%d ", pdataset->pdocs[m]->words[n], l[m][n], z[m][n]); //  wordID:sentiLab:topic
	    }
	    fprintf(fout, "\n");
    }

    fclose(fout);
	return 0;
}


int model::save_model_twords(string filename) 
{
    FILE * fout = fopen(filename.c_str(), "w");
    if (fout == NULL) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }
    if (phi_lzw.empty()) {
        return 1;
    }
    if (id2word.empty()) {
        return 1;
    }
    
    if (twords > vocabSize) {
	    twords = vocabSize; // print out entire vocab list
    }
    
    MapId2Word::iterator it;
    if (pdataset->pdocs == NULL ) {
        printf("pdataset->pdocs is null.\n");
        return 1;
    }
    
    for (int l = 0; l < numSentiLabs; l++) {
        for (int k = 0; k < numTopics; k++) { 
	        vector<pair<int, double> > words_probs;
	        for (int w = 0; w < vocabSize; w++) { 
	            words_probs.push_back(pair<int, double>(w,phi_lzw[l][k][w]));
	        }
		    std::sort(words_probs.begin(), words_probs.end(), sort_pred());
            
            fprintf(fout, "Label%d_Topic%d\n", l, k);
            
	        for (int i = 0; i < twords; i++) { 
		        it = id2word.find(words_probs[i].first);
                if (it != id2word.end()) {
			        fprintf(fout, "%s\t%15f\n", (it->second).c_str(), words_probs[i].second);
                }
	        }
	    }
    }
    
    fclose(fout);
    
    return 0;
}

int model::save_model_pi_dl(string filename) {

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
    }

	for (int m = 0; m < numDocs; m++) {
		fprintf(fout, "d_%d ", m);
		for (int l = 0; l < numSentiLabs; l++) {
			fprintf(fout, "%f ", pi_dl[m][l]);
		}
		fprintf(fout, "\n");
    }
   
    fclose(fout);       
	return 0;
}


int model::save_model_theta_dlz(string filename) {

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
		printf("Cannot save file %s!\n", filename.c_str());
		return 1;
    }
    
    for (int m = 0; m < numDocs; m++) {
        fprintf(fout, "Document::%d\n", m);
	    for (int l = 0; l < numSentiLabs; l++) {
	        for (int z = 0; z < numTopics; z++) {
		        fprintf(fout, "%f ", theta_dlz[m][l][z]);
	        }
		    fprintf(fout, "\n");
		 }
    }

    fclose(fout);
    return 0;
}


int model::save_model_phi_lzw(string filename) {

	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }
    
	for (int l = 0; l < numSentiLabs; l++) {  
	    for (int z = 0; z < numTopics; z++) { 
		    fprintf(fout, "Label::%d::Topic::%d\n", l, z);
     	    for (int r = 0; r < vocabSize; r++) {
			    fprintf(fout, "%.15f ", phi_lzw[l][z][r]);
     	    }
            fprintf(fout, "\n");
	   }
    }
    
    fclose(fout);
	return 0;
}



int model::save_model_others(string filename) {
	FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	    printf("Cannot save file %s!\n", filename.c_str());
	    return 1;
    }
    
	fprintf(fout, "data_dir=%s\n", this->input_dir.c_str());
	fprintf(fout, "result_dir=%s\n", this->output_dir.c_str());
	fprintf(fout, "sentiment_lexicon_dir=%s\n", this->sentiment_lexicon_dir.c_str());

	fprintf(fout, "\n-------------------- Corpus statistics -----------------------\n");
    fprintf(fout, "numDocs=%d\n", numDocs);
    fprintf(fout, "corpusSize=%d\n", corpusSize);
	fprintf(fout, "aveDocLength=%d\n", aveDocLength);
    fprintf(fout, "vocabSize=%d\n", vocabSize);

    fprintf(fout, "\n---------------------- Model settings -----------------------\n");
	fprintf(fout, "numSentiLabs=%d\n", numSentiLabs);
	fprintf(fout, "numTopics=%d\n", numTopics);
	fprintf(fout, "liter=%d\n", liter);
	fprintf(fout, "savestep=%d\n", savestep);
	fprintf(fout, "updateParaStep=%d\n", updateParaStep);

	fprintf(fout, "_alpha=%f\n", _alpha);
	fprintf(fout, "_beta=%f\n", _beta);
	fprintf(fout, "_gamma=%f\n", _gamma);
    
    if (fout != NULL ) {
        fclose(fout);
    }
    else{
        printf("METHOD save_model_others fout is null \n");
    }
    return 0;
}

int model::sampling(int m, int n, int& sentiLab, int& topic) {
	sentiLab = l[m][n];
	topic = z[m][n];
	int w = pdataset->pdocs[m]->words[n]; // the ID/index of the current word token in vocabulary 
	double u;
	nd[m]--;
	ndl[m][sentiLab]--;
	ndlz[m][sentiLab][topic]--;
	nlzw[sentiLab][topic][pdataset->pdocs[m]->words[n]]--;
	nlz[sentiLab][topic]--;

	for (int l = 0; l < numSentiLabs; l++) {
		for (int k = 0; k < numTopics; k++) {
			p[l][k] = (nlzw[l][k][w] + beta_lzw[l][k][w]) / (nlz[l][k] + betaSum_lz[l][k]) *
		   		(ndlz[m][l][k] + alpha_lz[l][k]) / (ndl[m][l] + alphaSum_l[l]) *
				(ndl[m][l] + gamma_dl[m][l]) / (nd[m] + gammaSum_d[m]);
		}
	}

	for (int l = 0; l < numSentiLabs; l++)  {
		for (int k = 0; k < numTopics; k++) {
			if (k==0) {
			    if (l==0) continue;
		        else p[l][k] += p[l-1][numTopics-1]; // accumulate the sum of the previous array
			}
			else p[l][k] += p[l][k-1];
		}
	}

	// probability normalization
	u = ((double)rand() / RAND_MAX) * p[numSentiLabs-1][numTopics-1];

	// sample sentiment label l, where l \in [0, S-1]
	bool loopBreak=false;
	for (sentiLab = 0; sentiLab < numSentiLabs; sentiLab++) {   
		for (topic = 0; topic < numTopics; topic++) { 
		    if (p[sentiLab][topic] > u) {
		        loopBreak = true;
		        break;
		    }
		}
		if (loopBreak == true) {
			break;
		}
	}
    
	if (sentiLab == numSentiLabs) sentiLab = numSentiLabs - 1; // to avoid over array boundary
	if (topic == numTopics) topic = numTopics - 1;
	
	// add estimated 'z' and 'l' to count variables
	nd[m]++;
	ndl[m][sentiLab]++;
	ndlz[m][sentiLab][topic]++;
	nlzw[sentiLab][topic][pdataset->pdocs[m]->words[n]]++;
	nlz[sentiLab][topic]++;
	
    return 0;
}


int model::djst_initial_beta_1(){
	MapWord2Id::iterator word2id_it;
	lambda_lw.resize(numSentiLabs);
    
	for (int l = 0; l < numSentiLabs; l++) {
	    lambda_lw[l].resize(vocabSize);
		for (int r = 0; r < vocabSize; r++) {
			lambda_lw[l][r] = 1;
		}
	}
    // Default sentiment: neural
	for (int r = 0; r < vocabSize; r++ ) {
		lambda_lw[0][r] = 0.9;//neural
		lambda_lw[1][r] = 0.05;
		lambda_lw[2][r] = 0.05;
	}
    
	for (int i = 0; i < word_positive.size(); i++ ) {
		word2id_it	 = word2id.find(word_positive[i]);
		if ( word2id_it != word2id.end()) {
			lambda_lw[0][word2id_it->second] = 0.05;
			lambda_lw[1][word2id_it->second] = 0.9;// positive
			lambda_lw[2][word2id_it->second] = 0.05;
		}
	}
    
	for (int i = 0; i < word_negative.size(); i++ ) {
		word2id_it = word2id.find(word_negative[i]);
		if (word2id_it != word2id.end()) {
			lambda_lw[0][word2id_it->second] = 0.05;
			lambda_lw[1][word2id_it->second] = 0.05;
			lambda_lw[2][word2id_it->second] = 0.9;// negative
		}
	}
    
	beta_lzw.resize(numSentiLabs);
	betaSum_lz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		beta_lzw[l].resize(numTopics);
		betaSum_lz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			betaSum_lz[l][z] = 0.0;
			beta_lzw[l][z].resize(vocabSize);
			for (int r = 0; r < vocabSize; r++) {
                beta_lzw[l][z][r] = lambda_lw[l][r] * 0.01;
                betaSum_lz[l][z] += beta_lzw[l][z][r];
			}
		} 		
	}
	return 0;
}

int model::djst_initial_beta_2(){
    
    for (int l = 0; l < numSentiLabs; l++) {
        for (int z = 0; z < numTopics; z++) {
            betaSum_lz[l][z] = 0.0;
        }
    }
	for (int t = 0; t < E_tlzw.size(); t++) {
		for (int l = 0; l < numSentiLabs; l++ ) {
			for (int z = 0; z < numTopics; z++ ) {
				for (int w = 0; w < vocabSize; w++ ) {
                    beta_lzw[l][z][w] = E_tlzw[t][l][z][w] * miu_t[t];
                    if ( beta_lzw[l][z][w]  == 0 ) beta_lzw[l][z][w] = 0.01;
				    betaSum_lz[l][z] += beta_lzw[l][z][w];
				}
			}
		}
	}
	return 0;
}

int model::djst_initial_others(){
	nd.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		nd[m]  = 0;
	}

	ndl.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		ndl[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
		    ndl[m][l] = 0;
		}
	}

	ndlz.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		ndlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			ndlz[m][l].resize(numTopics);
            for (int z = 0; z < numTopics; z++) {
				ndlz[m][l][z] = 0;
            }
		}
	}

	nlzw.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlzw[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
			nlzw[l][z].resize(vocabSize);
            for (int r = 0; r < vocabSize; r++) {
			    nlzw[l][z][r] = 0;
            }
		}
	}
	
	nlz.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		nlz[l].resize(numTopics);
		for (int z = 0; z < numTopics; z++) {
		    nlz[l][z] = 0;
		}
	}

	p.resize(numSentiLabs);
	for (int l = 0; l < numSentiLabs; l++) {
		p[l].resize(numTopics);
	}
	
	pi_dl.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		pi_dl[m].resize(numSentiLabs);
	}

	theta_dlz.resize(numDocs);
	for (int m = 0; m < numDocs; m++) {
		theta_dlz[m].resize(numSentiLabs);
		for (int l = 0; l < numSentiLabs; l++) {
			theta_dlz[m][l].resize(numTopics);
		}
	}
    
    for (int l = 0; l < numSentiLabs; l++) {
        alphaSum_l[l] = 0.0;
        for (int z = 0; z < numTopics; z++) {
            alpha_lz[l][z] = _alpha;
            alphaSum_l[l] += alpha_lz[l][z];
        }
    }
	
	this->set_gamma();

	for (int l = 0 ; l < numSentiLabs; l++ ) {
		for ( int z = 0; z < numTopics; z++ ) {
			clz[l][z] = 0.0;
			for(int r = 0; r < vocabSize; r++ ) {
				clzw[l][z][r] = 0.0;
			}
		}
	}
    
	srand(time(0));
    
	int sentiLab, topic;
	z.resize(numDocs);
	l.resize(numDocs);

	for (int m = 0; m < numDocs; m++) {
		int docLength = pdataset->pdocs[m]->length;
		z[m].resize(docLength);
		l[m].resize(docLength);
        for (int t = 0; t < docLength; t++) {
			sentiLab = (int)(((double)rand() / RAND_MAX) * numSentiLabs);
    	    l[m][t] = sentiLab;
			topic = (int)(((double)rand() / RAND_MAX) * numTopics);
			z[m][t] = topic;
			nd[m]++;
			ndl[m][sentiLab]++;
			ndlz[m][sentiLab][topic]++;
			nlzw[sentiLab][topic][pdataset->pdocs[m]->words[t]]++;
			nlz[sentiLab][topic]++;
        }
    }
	return 0;
}

int model::update_E(){
    
    for (int l = 0; l < numSentiLabs; l++)  {
        for (int z = 0; z < numTopics; z++) {
            clz[l][z] = 0.0;
            for(int r = 0; r < vocabSize; r++) {
                clzw[l][z][r] = (nlzw[l][z][r] +1 )* phi_lzw[l][z][r];
                clz[l][z] += clzw[l][z][r];
            }
        }
    }
    for (int l = 0; l < numSentiLabs; l++ ) {
        for (int z = 0; z < numTopics; z++ ) {
            for (int r = 0; r < vocabSize; r++ ) {
                sigma_tmp[l][z][r] = clzw[l][z][r] / clz[l][z];
            }
        }
    }
    
    E_tlzw.push_back(sigma_tmp);
    if (E_tlzw.size() > S ) E_tlzw.erase(E_tlzw.begin());
    return 0;
}

int model::djst_initial_general(){
    this->word2id = pdataset->word2id;
    this->id2word = pdataset->id2word;
    this->vocabSize = pdataset->vocabSize;
    this->smooth_v = 1.0/(float)this->vocabSize;
    
    miu_t.resize(S);
    float sum_miu = 0.0;
    for (int i = 1; i <= S; i++ ) {
        miu_t[i-1] = (1.0/(float)S) * i;
        sum_miu += miu_t[i-1];
    }
    for (int i = 0; i < S; i++ ) {
        miu_t[i] /= sum_miu;
    }
    
    sigma_tmp.resize(numSentiLabs); // L * T * V
    for (int l = 0; l < numSentiLabs; l++ ) {
        sigma_tmp[l].resize(numTopics);
        for (int z = 0; z < numTopics; z++ ) {
            sigma_tmp[l][z].resize(vocabSize);
        }
    }
    
    phi_lzw.resize(numSentiLabs);
    for (int l = 0; l < numSentiLabs; l++) {
        phi_lzw[l].resize(numTopics);
        for (int z = 0; z < numTopics; z++) {
            phi_lzw[l][z].resize(vocabSize);
        }
    }
    
    alpha_lz.resize(numSentiLabs);
    for (int l = 0; l < numSentiLabs; l++) {
        alpha_lz[l].resize(numTopics);
    }
    alphaSum_l.resize(numSentiLabs);
    
    if (_alpha <= 0) {
        _alpha = (double)aveDocLength * 0.05 / (double)(numSentiLabs * numTopics);
    }
    
    opt_alpha_lz.resize(numSentiLabs);
    for (int l = 0; l < numSentiLabs; l++) {
        opt_alpha_lz[l].resize(numTopics);
    }
    
    clzw.resize(numSentiLabs);
    clz.resize(numSentiLabs);
    for (int l = 0 ; l < numSentiLabs; l++ ) {
        clzw[l].resize(numTopics);
        clz[l].resize(numTopics);
        for ( int z = 0; z < numTopics; z++ ) {
            clzw[l][z].resize(vocabSize);
            clz[l][z] = 0.0;
            for (int r = 0; r < vocabSize; r++ ) {
                clzw[l][z][r] = 0.0;
            }
        }
    }
    
    return 0;
}

int model::update_Parameters() {
    printf("update_parameters.\n");
	int ** data; // temp valuable for exporting 3-dimentional array to 2-dimentional
	double * alpha_temp;
	data = new int*[numTopics];
	for (int k = 0; k < numTopics; k++) {
		data[k] = new int[numDocs];
		for (int m = 0; m < numDocs; m++) {
			data[k][m] = 0;
		}
	}
	alpha_temp = new double[numTopics];
	for (int k = 0; k < numTopics; k++) {
		alpha_temp[k] = 0.0;
	}
	// update alpha
	for (int j = 0; j < numSentiLabs; j++) {
		for (int k = 0; k < numTopics; k++) {
			for (int m = 0; m < numDocs; m++) {
				data[k][m] = ndlz[m][j][k] + 1; // ntldsum[j][k][m];
			}
		}

		for (int k = 0; k < numTopics; k++) {
			alpha_temp[k] =  alpha_lz[j][k]; //alpha[j][k];
		}

		polya_fit_simple(data, alpha_temp, numTopics, numDocs);

		alphaSum_l[j] = 0.0;
		for (int k = 0; k < numTopics; k++) {
			alpha_lz[j][k] = alpha_temp[k];
			alphaSum_l[j] += alpha_lz[j][k];
		}
	}// for_j
    
	// update beta
    for (int l = 0; l < numSentiLabs; l++ ) {
        for (int z = 0; z < numTopics; z++ ) {
            betaSum_lz[l][z] = 0.0;
        }
    }
    for (int t = 0; t < E_tlzw.size(); t++) {
        for (int l = 0; l < numSentiLabs; l++ ) {
            for (int z = 0; z < numTopics; z++ ) {
                for (int w = 0; w < vocabSize; w++ ) {
                    beta_lzw[l][z][w] = (E_tlzw[t][l][z][w] + smooth_v) * miu_t[t];
                    betaSum_lz[l][z] += beta_lzw[l][z][w];
                }
            }
        }
    }
	return 0;
}
