/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

 /*   cudnn : Neural Networks Library

 */

#if !defined(CUDNN_H_)
#define CUDNN_H_

#include "driver_types.h"
#include <cuda_runtime.h>

#ifndef CUDNNWINAPI
#ifdef _WIN32
#define CUDNNWINAPI __stdcall
#else
#define CUDNNWINAPI 
#endif
#endif

#if defined (__cplusplus)
extern "C" {
#endif

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

/*
 * CUDNN return codes
 */
typedef enum
{
    CUDNN_STATUS_SUCCESS          = 0,
    CUDNN_STATUS_NOT_INITIALIZED  = 1,
    CUDNN_STATUS_ALLOC_FAILED     = 2,
    CUDNN_STATUS_BAD_PARAM        = 3,
    CUDNN_STATUS_INTERNAL_ERROR   = 4,
    CUDNN_STATUS_INVALID_VALUE    = 5,
    CUDNN_STATUS_ARCH_MISMATCH    = 6,
    CUDNN_STATUS_MAPPING_ERROR    = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED    = 9,
    CUDNN_STATUS_LICENSE_ERROR    = 10
} cudnnStatus_t;


cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);


/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct cudnnTensor4dStruct*      cudnnTensor4dDescriptor_t;
typedef struct cudnnConvolutionStruct*   cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*       cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*        cudnnFilterDescriptor_t;


/*
* CUDNN data type
*/
typedef enum
{
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1
} cudnnDataType_t;

/* Create an instance of Tensor4d descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateTensor4dDescriptor( cudnnTensor4dDescriptor_t *tensorDesc );

typedef enum
{
    CUDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
} cudnnTensorFormat_t;

cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(   cudnnTensor4dDescriptor_t tensorDesc,
                                                        cudnnTensorFormat_t  format,                      
                                                        cudnnDataType_t dataType, // image data type
                                                        int n,        // number of inputs (batch size)
                                                        int c,        // number of input feature maps
                                                        int h,        // height of input section
                                                        int w         // width of input section
                                                    );


cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx( cudnnTensor4dDescriptor_t tensorDesc,                          
                                                        cudnnDataType_t dataType, // image data type
                                                        int n,        // number of inputs (batch size)
                                                        int c,        // number of input feature maps
                                                        int h,        // height of input section
                                                        int w,        // width of input section
                                                        int nStride,
                                                        int cStride,
                                                        int hStride,
                                                        int wStride
                                                      );

cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(   const cudnnTensor4dDescriptor_t tensorDesc,
                                                        cudnnDataType_t *dataType, // image data type
                                                        int *n,        // number of inputs (batch size)
                                                        int *c,        // number of input feature maps
                                                        int *h,        // height of input section
                                                        int *w,        // width of input section
                                                        int *nStride,
                                                        int *cStride,
                                                        int *hStride,
                                                        int *wStride
                                                    );

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1


   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

   Example 1) is the one currently implemented
*/

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyTensor4dDescriptor( cudnnTensor4dDescriptor_t tensorDesc );


/* Tensor layout conversion helper */
cudnnStatus_t CUDNNWINAPI cudnnTransformTensor4d( cudnnHandle_t handle,
                                                  cudnnTensor4dDescriptor_t  srcDesc,
                                                  const void                *srcData,
                                                  cudnnTensor4dDescriptor_t  destDesc,
                                                  void                      *destData
                                                );

typedef enum
{
   CUDNN_ADD_IMAGE   = 0,       /* add one image to every feature maps of each input */
   CUDNN_ADD_SAME_HW = 0,        

   CUDNN_ADD_FEATURE_MAP = 1,   /* add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest */
   CUDNN_ADD_SAME_CHW    = 1,

   CUDNN_ADD_SAME_C      = 2,   /* add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input */

   CUDNN_ADD_FULL_TENSOR = 3    /* add 2 tensors with same n,c,h,w */
} cudnnAddMode_t;           
                                         
/* Tensor Bias addition */
cudnnStatus_t CUDNNWINAPI cudnnAddTensor4d( cudnnHandle_t handle,            
                                            cudnnAddMode_t mode,
                                            const void *alpha,                                         
                                            cudnnTensor4dDescriptor_t  biasDesc,
                                            const void                *biasData,
                                            cudnnTensor4dDescriptor_t  srcDestDesc,
                                            void                      *srcDestData
                                          );

/* Set all data points of a tensor to a given value */
cudnnStatus_t CUDNNWINAPI cudnnSetTensor4d( cudnnHandle_t handle,
                                            cudnnTensor4dDescriptor_t  tensorDesc,
                                            void                      *tensorData,
                                            const void                *value
                                          );
                                      
/*
 *  convolution mode
 */
typedef enum
{
    CUDNN_CONVOLUTION       = 0,         
    CUDNN_CROSS_CORRELATION = 1       
} cudnnConvolutionMode_t;

/*
 *  convolution path
 */
typedef enum
{
    CUDNN_CONVOLUTION_FWD         = 0,        /* Tensor Convolution function */
    CUDNN_CONVOLUTION_WEIGHT_GRAD = 1,        /* Weight Gradient update function */
    CUDNN_CONVOLUTION_DATA_GRAD   = 2         /* Data Gradient update function */
} cudnnConvolutionPath_t;

/* Create an instance of FilterStruct */
cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor( cudnnFilterDescriptor_t *filterDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetFilterDescriptor(    cudnnFilterDescriptor_t filterDesc,                                                       
                                                       cudnnDataType_t dataType, // image data type
                                                       int k,        // number of output feature maps
                                                       int c,        // number of input feature maps
                                                       int h,        // height of each input filter
                                                       int w         // width of  each input fitler
                                                  );

cudnnStatus_t CUDNNWINAPI cudnnGetFilterDescriptor(    const cudnnFilterDescriptor_t filterDesc,                                                     
                                                       cudnnDataType_t *dataType, // image data type
                                                       int *k,        // number of output feature maps
                                                       int *c,        // number of input feature maps
                                                       int *h,        // height of each input filter
                                                       int *w         // width of  each input fitler
                                                  );


cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor( cudnnFilterDescriptor_t filterDesc );

/* Create an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor( cudnnConvolutionDescriptor_t *convDesc );

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionDescriptor(    cudnnConvolutionDescriptor_t convDesc,
                                                            cudnnTensor4dDescriptor_t inputTensorDesc,
                                                            cudnnFilterDescriptor_t filterDesc,
                                                            int pad_h,    // zero-padding height
                                                            int pad_w,    // zero-padding width
                                                            int u,        // vertical filter stride
                                                            int v,        // horizontal filter stride
                                                            int upscalex, // upscale the input in x-direction
                                                            int upscaley, // upscale the input in y-direction
                                                            cudnnConvolutionMode_t mode
                                                       );
                                               

cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionDescriptorEx(  cudnnConvolutionDescriptor_t convDesc,
                                                            int n,
                                                            int c,
                                                            int h,
                                                            int w,
                                                            int k,
                                                            int r,
                                                            int s,
                                                            int pad_h,    // zero-padding height
                                                            int pad_w,    // zero-padding width
                                                            int u,        // vertical filter stride
                                                            int v,        // horizontal filter stride
                                                            int upscalex, // upscale the input in x-direction
                                                            int upscaley, // upscale the input in y-direction
                                                            cudnnConvolutionMode_t mode
                                                         );

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnGetOutputTensor4dDim( const cudnnConvolutionDescriptor_t convDesc,
                                                     cudnnConvolutionPath_t path,
                                                     int *n,
                                                     int *c,
                                                     int *h,
                                                     int *w
                                                   );

/* Destroy an instance of convolution descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor( cudnnConvolutionDescriptor_t convDesc );

/*
 *  accumulate the result of the operation into output buffer (or overwrite) ?
 */
typedef enum
{
    CUDNN_RESULT_ACCUMULATE      = 0,           /* Evaluate O += I * F */
    CUDNN_RESULT_NO_ACCUMULATE   = 1            /* Evaluate O = I * F  */
} cudnnAccumulateResult_t;

/* Function to perform the forward multiconvolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(        cudnnHandle_t handle,
                                                          cudnnTensor4dDescriptor_t     srcDesc,
                                                          const void                   *srcData,
                                                          cudnnFilterDescriptor_t       filterDesc,
                                                          const void                   *filterData,
                                                          cudnnConvolutionDescriptor_t  convDesc,
                                                          cudnnTensor4dDescriptor_t     destDesc,
                                                          void                         *destData,
                                                          cudnnAccumulateResult_t       accumulate
                                                 );

/* Functions to perform the backward multiconvolution */
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(   cudnnHandle_t handle,
                                                          cudnnTensor4dDescriptor_t     srcDesc,
                                                          const void                   *srcData,
                                                          cudnnTensor4dDescriptor_t     destDesc,
                                                          void                         *destData,
                                                          cudnnAccumulateResult_t       accumulate
                                                      );

cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter( cudnnHandle_t handle,
                                                          cudnnTensor4dDescriptor_t     srcDesc,
                                                          const void                   *srcData,
                                                          cudnnTensor4dDescriptor_t     diffDesc,
                                                          const void                   *diffData, 
                                                          cudnnConvolutionDescriptor_t  convDesc,
                                                          cudnnFilterDescriptor_t       gradDesc,
                                                          void                         *gradData,
                                                          cudnnAccumulateResult_t       accumulate
                                                        );
                                                        
cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData( cudnnHandle_t handle,
                                                         cudnnFilterDescriptor_t       filterDesc,
                                                         const void                    *filterData,
                                                         cudnnTensor4dDescriptor_t     diffDesc,
                                                         const void                    *diffData,
                                                         cudnnConvolutionDescriptor_t  convDesc,
                                                         cudnnTensor4dDescriptor_t     gradDesc,
                                                         void                          *gradData,
                                                         cudnnAccumulateResult_t       accumulate
                                                       );                                                        
                                        
/*
 *  softmax algorithm
 */
typedef enum
{
    CUDNN_SOFTMAX_FAST     = 0,        /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1         /* subtract max from every point to avoid overflow */
} cudnnSoftmaxAlgorithm_t;

typedef enum
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;


/* Function to perform forward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(  cudnnHandle_t handle,
                                                cudnnSoftmaxAlgorithm_t    algorithm,
                                                cudnnSoftmaxMode_t         mode,
                                                cudnnTensor4dDescriptor_t  srcDesc,
                                                const void                *srcData,
                                                cudnnTensor4dDescriptor_t  destDesc,
                                                void                      *destData
                                             );

/* Function to perform backward softmax */
cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward( cudnnHandle_t handle,
                                                cudnnSoftmaxAlgorithm_t    algorithm,
                                                cudnnSoftmaxMode_t         mode,
                                                cudnnTensor4dDescriptor_t  srcDesc,
                                                const void                *srcData,
                                                cudnnTensor4dDescriptor_t  srcDiffDesc,
                                                const void                *srcDiffData,
                                                cudnnTensor4dDescriptor_t  destDiffDesc,
                                                void                      *destDiffData
                                              );

/*
 *  pooling mode
 */
typedef enum
{
    CUDNN_POOLING_MAX     = 0,
    CUDNN_POOLING_AVERAGE = 1
} cudnnPoolingMode_t;

/* Create an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor( cudnnPoolingDescriptor_t *poolingDesc);

cudnnStatus_t CUDNNWINAPI cudnnSetPoolingDescriptor(    cudnnPoolingDescriptor_t poolingDesc,
                                                        cudnnPoolingMode_t mode,
                                                        int windowHeight,
                                                        int windowWidth,
                                                        int verticalStride,
                                                        int horizontalStride
                                                   );
                                              
cudnnStatus_t CUDNNWINAPI cudnnGetPoolingDescriptor(    const cudnnPoolingDescriptor_t poolingDesc,
                                                        cudnnPoolingMode_t *mode,
                                                        int *windowHeight,
                                                        int *windowWidth,
                                                        int *verticalStride,
                                                        int *horizontalStride
                                                   );                                   

/* Destroy an instance of pooling descriptor */
cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor( cudnnPoolingDescriptor_t poolingDesc );

/* Function to perform forward pooling */
cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(  cudnnHandle_t handle,
                                                cudnnPoolingDescriptor_t   poolingDesc,
                                                cudnnTensor4dDescriptor_t  srcDesc,
                                                const void                *srcData,
                                                cudnnTensor4dDescriptor_t  destDesc,
                                                void                      *destData
                                             );

 /* Function to perform backward pooling */
cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward( cudnnHandle_t handle,
                                                cudnnPoolingDescriptor_t     poolingDesc,
                                                cudnnTensor4dDescriptor_t    srcDesc,
                                                const void                  *srcData,
                                                cudnnTensor4dDescriptor_t    srcDiffDesc,
                                                const void                  *srcDiffData,
                                                cudnnTensor4dDescriptor_t    destDesc,
                                                const void                  *destData,
                                                cudnnTensor4dDescriptor_t    destDiffDesc,
                                                void                        *destDiffData
                                              );

/* 
 * activation mode
 */
typedef enum
{
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU    = 1,
    CUDNN_ACTIVATION_TANH    = 2
} cudnnActivationMode_t;


/* Function to perform forward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationForward( cudnnHandle_t handle,
                                                  cudnnActivationMode_t mode,
                                                  cudnnTensor4dDescriptor_t  srcDesc,
                                                  const void                *srcData,
                                                  cudnnTensor4dDescriptor_t  destDesc,
                                                  void                      *destData
                                                );
/* Function to perform backward activation  */
cudnnStatus_t CUDNNWINAPI cudnnActivationBackward( cudnnHandle_t handle,
                                                   cudnnActivationMode_t mode,
                                                   cudnnTensor4dDescriptor_t  srcDesc,
                                                   const void                *srcData,
                                                   cudnnTensor4dDescriptor_t  srcDiffDesc,
                                                   const void                *srcDiffData,
                                                   cudnnTensor4dDescriptor_t  destDesc,
                                                   const void                *destData,
                                                   cudnnTensor4dDescriptor_t  destDiffDesc,
                                                   void                      *destDiffData
                                                 );
#if defined (__cplusplus)
}
#endif

#endif /* CUDNN_H_ */
