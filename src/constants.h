/**********************************************************************
		        Dynamic Joint Sentiment-Topic (DJST) Model
***********************************************************************

Written by: Ruidong Wang, wantrd@yeah.net
Part of code is from http://gibbslda.sourceforge.net/. and https://github.com/linron84/JST
This file is part of DJST implementation.
***********************************************************************/


#ifndef _CONSTANTS_H
#define _CONSTANTS_H

#define	BUFF_SIZE_LONG	1000000
#define	BUFF_SIZE_SHORT	512

#define	MODEL_STATUS_UNKNOWN	0
#define	MODEL_STATUS_EST	1
#define	MODEL_STATUS_ESTC	2
#define	MODEL_STATUS_INF	3

#define	MODE_NONE	0
#define	MODE_SLIDING	1
#define	MODE_SKIP	2
#define	MODE_MULTISCALE	3

#define	MAX_ITERATION	100000

#endif

