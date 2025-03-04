/*******************************************************************************
* 2pt_box/load_conf.h: this file is part of the FCFC program.

* FCFC: Fast Correlation Function Calculator.

* Github repository:
        https://github.com/cheng-zhao/FCFC

* Copyright (c) 2020 -- 2021 Cheng Zhao <zhaocheng03@gmail.com>
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

*******************************************************************************/

#ifndef __LOAD_CONF_H__
#define __LOAD_CONF_H__

#include <stdbool.h>

/*============================================================================*\
                   Data structure for storing configurations
\*============================================================================*/

typedef struct {
  char *fconf;          /* Name of the configuration file. */
  char **input;         /* CATALOG              */
  int ninput;           /* Number of input catalogues. */
  bool ascii;           /* Indicate whether there are ASCII catalogues. */
  bool hdf5;            /* Indicate whether there are HDF5 catalogues. */
  char *label;          /* CATALOG_LABEL        */
  int *ftype;           /* CATALOG_TYPE         */
  long *skip;           /* ASCII_SKIP           */
  char *comment;        /* ASCII_COMMENT        */
  char **fmtr;          /* ASCII_FORMATTER      */
  char **group;          /* HDF5_GROUP           */
  char **pos;           /* POSITION             */
  char **sel;           /* SELECTION            */
  double bsize;         /* BOX_SIZE             */

  int bintype;          /* BINNING_SCHEME       */
  char **pc;            /* PAIR_COUNT           */
  bool *comp_pc;        /* Indicate whether to evaluate the pair counts. */
  int npc;              /* Number of pair counts to be computed. */
  char **pcout;         /* PAIR_COUNT_FILE      */
  char **cf;            /* CF_ESTIMATOR         */
  int ncf;              /* Number of correlation functions to be computed. */
  char **cfout;         /* CF_OUTPUT_FILE       */
  int *poles;           /* MULTIPOLE            */
  int npole;            /* Number of multipoles to be evaluated. */
  char **mpout;         /* MULTIPOLE_FILE       */
  bool wp;              /* PROJECTED_CF         */
  char **wpout;         /* PROJECTED_FILE       */

  char *fsbin;          /* SEP_BIN_FILE         */
  double smin;          /* SEP_BIN_MIN          */
  double smax;          /* SEP_BIN_MAX          */
  double ds;            /* SEP_BIN_SIZE         */
  int nsbin;            /* Number of separation bins */
  int nmu;              /* MU_BIN_NUM           */
  char *fpbin;          /* PI_BIN_FILE          */
  double pmin;          /* PI_BIN_MIN           */
  double pmax;          /* PI_BIN_MAX           */
  double dpi;           /* PI_BIN_SIZE          */
  int npbin;            /* Number of pi bins    */
  int dprec;            /* SQ_DIST_PREC         */

  int ostyle;           /* OUTPUT_STYLE         */
  int ovwrite;          /* OVERWRITE            */
  bool verbose;         /* VERBOSE              */
} CONF;


/*============================================================================*\
                      Interface for loading configurations
\*============================================================================*/

/******************************************************************************
Function `load_conf`:
  Read, check, and print configurations.
Arguments:
  * `argc`:     number of arguments passed via command line;
  * `argv`:     array of command line arguments.
Return:
  The structure for storing configurations.
******************************************************************************/
CONF *load_conf(const int argc, char *const *argv);

/******************************************************************************
Function `conf_destroy`:
  Release memory allocated for the configurations.
Arguments:
  * `conf`:     the structure for storing configurations.
******************************************************************************/
void conf_destroy(CONF *conf);

#endif
