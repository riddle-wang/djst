/**********************************************************************
 Dynamic Joint Sentiment-Topic (DJST) Model
 ***********************************************************************
 
 Written by: Ruidong Wang, wantrd@yeah.net
 Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
 This file is part of DJST implementation.
 ***********************************************************************/

   
#ifndef	_MATH_FUNC_H
#define	_MATH_FUNC_H


//*************************  asa032.h   ************************************//
double alngam ( double xvalue, int *ifault );
double gamain ( double x, double p, int *ifault );
void gamma_inc_values ( int *n_data, double *a, double *x, double *fx );
double r8_abs ( double x );
void timestamp ( void );


//*************************  asa103.cpp   ************************************//
double digama ( double x, int *ifault );
void psi_values ( int *n_data, double *x, double *fx );
//void timestamp ( void );


//*************************  asa121.cpp   ************************************//
//void timestamp ( void );
double trigam ( double x, int *ifault );
void trigamma_values ( int *n_data, double *x, double *fx );


#endif
