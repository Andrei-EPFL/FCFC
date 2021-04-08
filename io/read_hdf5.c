#include "read_file.h"
#include "libast.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include "hdf5.h"


static int read_col(hid_t group_id, char const *pos, real **data, hsize_t *num) {
    const int ndims = 1;//H5Sget_simple_extent_ndims(data_space);
    hsize_t dims[ndims];
    
    hid_t dataset_id = 0, data_space = 0;  /* identifiers */

    /* Open an existing dataset. */
    if (!(dataset_id = H5Dopen(group_id, pos, H5P_DEFAULT))) {
        P_ERR("failed to open the dataset of column %s\n", pos);
        //CLEAN_PTR;
        return 1;
    }
    /* Get the dataspace of the dataset_id */
    if (!(data_space = H5Dget_space(dataset_id))) {
        P_ERR("Failed to get the data_space from the dataset of column %s\n", pos);
        //CLEAN_PTR;
        return 1;
    }

    if (H5Sget_simple_extent_dims(data_space, dims, NULL) < 0) {
        P_ERR("There are no dimensions in the dataset\n");
        //CLEAN_PTR;
        return 1;
    }
    *num = dims[0];

    if (!(*data = malloc(sizeof(real) * dims[0]))) {
        P_ERR("failed to allocate memory the column\n");
        return 1;
    }
    
    /* Read the dataset. */
    if (H5Dread(dataset_id, H5T_REAL, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data) < 0) {
        P_ERR("failed to read from the dataset of column %s\n", pos);
        return 1;
    }

    /* Close the dataset. */
    if (H5Dclose(dataset_id) < 0) {
        P_ERR("failed to close the dataset of column %s\n", pos);
        return 1;
    }

    return 0;
}

int read_hdf5_data(const char *fname, const char *groupname, char *const *pos, const char *wt, const char *sel,
    DATA **data, size_t *num, const int verb) {

    hid_t file_id = 0, group_id = 0;    
    
    DATA *tmp;
    real *datax = NULL, *datay = NULL, *dataz = NULL;
    hsize_t dimx, dimy, dimz;
    size_t index = 0;
    
    /* Open an existing file. */
    if (!(file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT))) {
        P_ERR("Failed to open the HDF5 file %s\n", fname);
        //CLEAN_PTR;
        return FCFC_ERR_FILE;
    }

    /* Open an existing group. */
    if (!(group_id = H5Gopen(file_id, groupname, H5P_DEFAULT))) {
        P_ERR("failed to open the group %s\n", groupname);
        //CLEAN_PTR;
        return FCFC_ERR_FILE;
    }
    
    /* Read the columns */
    if (read_col(group_id, pos[0], &datax, &dimx)) {
        P_ERR("failed to read the column %s\n", pos[0]);
        return 1;
    }

    if (read_col(group_id, pos[1], &datay, &dimy)) {
        P_ERR("failed to read the column %s\n", pos[1]);
        return 1;
    }
    
    if (read_col(group_id, pos[2], &dataz, &dimz)) {
        P_ERR("failed to read the column %s\n", pos[2]);
        return 1;
    }

    /* Check dimensions of the columns */
    if ((dimx != dimy) || (dimy != dimz) || (dimz != dimx)) {
        P_ERR("the sizes of the columns are not compatible\n");
        return 1;
    }

    /* Allocate memory for data, a tmp variable */
    if (!(tmp = malloc(dimx * sizeof(DATA)))) {
        P_ERR("failed to allocate memory for the data\n");
        //CLEAN_PTR; 
        return FCFC_ERR_MEMORY;
    }
    *num = dimx;

#ifdef OMP
#pragma omp parallel for
#endif
    for (index = 0; index < dimx; index ++) {
        tmp[index].x[0] = datax[index];
        tmp[index].x[1] = datay[index];
        tmp[index].x[2] = dataz[index];
    }



#ifdef FCFC_DATA_WEIGHT
    real *weight = NULL;
    hsize_t dimw;
    if (wt) {
        if (read_col(group_id, wt, &weight, &dimw)) {
            P_ERR("failed to read the column %s\n", pos[2]);
            return 1;
        }
    }
    if (weight && (dimx == dimw)) {
#ifdef OMP
#pragma omp parallel for
#endif
        for (index = 0; index < dimx; index ++) {
            tmp[index].w = weight[index];
        }
    }
    else {
#ifdef OMP
#pragma omp parallel for
#endif
        for (index = 0; index < dimx; index ++) {
            tmp[index].w = 1;
        }        
    }
#endif

    *data = tmp;
    
    /* Close the group. */
    if (H5Gclose(group_id) < 0) {
        P_ERR("failed to close the group of the HDF5 file\n");
    }

    /* Close the file. */
    if (H5Fclose(file_id) < 0) {
        P_ERR("failed to close the HDF5 file\n");
    }
    return 0;
}