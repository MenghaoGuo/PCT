import math
import numpy as np
import jittor as jt
from jittor import nn
from jittor.contrib import concat 

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = jt.array(np.arange(B, dtype=np.int32)).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape
    # print ('before matmul size', src.size(), dst.size())
    dist = -2 * nn.matmul(src, dst.permute(0, 2, 1))
    dist += jt.sum(src ** 2, -1).view(B, N, 1)
    dist += jt.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        self.relu = nn.ReLU()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def execute(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N] 
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            idx, dists = jt.argsort(dists, dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = jt.sum(dist_recip, dim=2, keepdims=True)
            weight = dist_recip / norm
            interpolated_points = jt.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = concat ([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        # l = len(self.mlp_convs)
        for i, conv in self.mlp_convs.layers.items():
            # conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points))) 
        return new_points.permute(0, 2, 1)


def optimal_block(batch_size):
    return 2 ** int(math.log(batch_size))


class FurthestPointSampler(nn.Module):
    cuda_src='''
        __device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                                int idx1, int idx2) {
            const float v1 = dists[idx1], v2 = dists[idx2];
            const int i1 = dists_i[idx1], i2 = dists_i[idx2];
            dists[idx1] = max(v1, v2);
            dists_i[idx1] = v2 > v1 ? i2 : i1;
        }

        __global__ void furthest_point_sampling_kernel (
            int b, int n, int m, int block_size,
            const float *__restrict__ dataset,
            float *__restrict__ temp, 
            int *__restrict__ idxs) {

            if (m <= 0) return;

            extern __shared__ int dists_i[];
            float *dists =  (float *) &dists_i[block_size];

            int batch_index = blockIdx.x;
            dataset += batch_index * n * 3;
            temp += batch_index * n;
            idxs += batch_index * m;

            int tid = threadIdx.x;
            const int stride = block_size;

            int old = 0;
            if (threadIdx.x == 0) idxs[0] = old;

            // initialize temp with INF
            for (int k = tid; k < n; k += stride)
                temp[k] = 1e10;

            __syncthreads();
            for (int j = 1; j < m; j++) {
                int besti = 0;
                float best = -1;
                float x1 = dataset[old * 3 + 0];
                float y1 = dataset[old * 3 + 1];
                float z1 = dataset[old * 3 + 2];
                for (int k = tid; k < n; k += stride) {
                    float x2, y2, z2;
                    x2 = dataset[k * 3 + 0];
                    y2 = dataset[k * 3 + 1];
                    z2 = dataset[k * 3 + 2];
                    float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
                    if (mag <= 1e-3) continue;

                    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

                    float d2 = min(d, temp[k]);
                    temp[k] = d2;
                    besti = d2 > best ? k : besti;
                    best = d2 > best ? d2 : best;
                }
                dists[tid] = best;
                dists_i[tid] = besti;
                __syncthreads();

                if (block_size >= 512) {
                    if (tid < 256) {
                        __update(dists, dists_i, tid, tid + 256);
                    }
                    __syncthreads();
                }
                if (block_size >= 256) {
                    if (tid < 128) {
                        __update(dists, dists_i, tid, tid + 128);
                    }
                    __syncthreads();
                }
                if (block_size >= 128) {
                    if (tid < 64) {
                        __update(dists, dists_i, tid, tid + 64);
                    }
                    __syncthreads();
                }
                if (block_size >= 64) {
                    if (tid < 32) {
                        __update(dists, dists_i, tid, tid + 32);
                    }
                    __syncthreads();
                }
                if (block_size >= 32) {
                    if (tid < 16) {
                        __update(dists, dists_i, tid, tid + 16);
                    }
                    __syncthreads();
                }
                if (block_size >= 16) {
                    if (tid < 8) {
                        __update(dists, dists_i, tid, tid + 8);
                    }
                    __syncthreads();
                }
                if (block_size >= 8) {
                    if (tid < 4) {
                        __update(dists, dists_i, tid, tid + 4);
                    }
                    __syncthreads();
                }
                if (block_size >= 4) {
                    if (tid < 2) {
                        __update(dists, dists_i, tid, tid + 2);
                    }
                    __syncthreads();
                }
                if (block_size >= 2) {
                    if (tid < 1) {
                        __update(dists, dists_i, tid, tid + 1);
                    }
                    __syncthreads();
                }

                old = dists_i[0];
                if (tid == 0) idxs[j] = old;
            }
        }

        int block_size = #block_size;

        float *temp;
        cudaMallocManaged(&temp, in0_shape0 * in0_shape1 * sizeof(float));

        furthest_point_sampling_kernel<<<in0_shape0, block_size, 2*block_size*sizeof(int)>>>(
            in0_shape0,
            in0_shape1,
            out_shape1,
            block_size,
            in0_p,
            temp,
            out_p
        );
        cudaDeviceSynchronize();
        cudaFree(temp);
    '''
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def execute(self, x):
        '''
        Parameters
        ----------
        x: jt.Var, (B, N, 3)

        Returns
        -------
        y: jt.Var, (B, n_samples, 3)
        '''
        batch_size, n_points, n_coords = x.shape

        assert self.n_samples <= n_points
        assert n_coords == 3
        assert x.dtype == 'float32'

        block_size = optimal_block(batch_size)

        cuda_src = self.cuda_src.replace('#block_size', str(block_size))

        idxs_shape = [batch_size, self.n_samples]
        idxs = jt.code(idxs_shape, 'int32', [x,], cuda_src=cuda_src)
        
        y = x.reindex([batch_size, self.n_samples, 3], [
            'i0',               # Batchid
            '@e0(i0, i1)',      # Nid
            'i2'
        ], extras=[idxs])

        return y, idxs

class BallQueryGrouper(nn.Module):
    cuda_src = '''
        __global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                                int nsample,
                                                const float *__restrict__ new_xyz,
                                                const float *__restrict__ xyz,
                                                int *__restrict__ idx,
                                                int *__restrict__ cnt) {
            int batch_index = blockIdx.x;
            xyz += batch_index * n * 3;
            new_xyz += batch_index * m * 3;
            idx += m * nsample * batch_index;
            cnt += batch_index * m;

            int index = threadIdx.x;
            int stride = blockDim.x;

            float radius2 = radius * radius;
            for (int j = index; j < m; j += stride) {
                float new_x = new_xyz[j * 3 + 0];
                float new_y = new_xyz[j * 3 + 1];
                float new_z = new_xyz[j * 3 + 2];
                cnt[j] = 0;

                for (int k = 0; k < n && cnt[j] < nsample; ++k) {
                    float x = xyz[k * 3 + 0];
                    float y = xyz[k * 3 + 1];
                    float z = xyz[k * 3 + 2];
                    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                                (new_z - z) * (new_z - z);

                    if (d2 < radius2) {
                        if (cnt[j] == 0) {
                            for (int l = 0; l < nsample; ++l)
                                idx[j * nsample + l] = k;
                        }
                        idx[j * nsample + cnt[j]] = k;
                        ++cnt[j];
                    }
                }
            }
        }

        int block_size = #block_size;

        query_ball_point_kernel<<<in0_shape0, block_size>>>(
            in0_shape0, in1_shape1, in0_shape1, #radius, #nsample,
            in0_p, in1_p, out0_p, out1_p
        );
    '''
    def __init__(self, radius, n_samples, use_xyz):
        super().__init__()
        self.radius = radius
        self.n_samples = n_samples
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        '''
        Parameters
        ----------
        xyz: jt.Var, (B, N, 3)
        features: jt.Var, (B, N, C)

        Returns
        -------
        new_feature: jt.Var, (B, N, n_samples, C)
        '''
        batch_size_x, n_input, n_coords = new_xyz.shape
        assert n_coords == 3

        batch_size_p, n_points, n_coords = pointset.shape
        assert n_coords == 3
        assert batch_size_x == batch_size_p

        if feature is not None:
            batch_size_f, n_points_f, n_feature = feature.shape
            assert batch_size_x == batch_size_f
            assert n_points == n_points_f

        block_size = optimal_block(batch_size_x)

        cuda_src = self.cuda_src.replace('#block_size', str(block_size)) \
                                .replace('#radius', str(self.radius)) \
                                .replace('#nsample', str(self.n_samples))

        idxs_shape = [batch_size_x, n_input, self.n_samples]
        cnts_shape = [batch_size_x, n_input]
        idxs, cnts = jt.code(
            [idxs_shape, cnts_shape],
            ['int32', 'int32'],
            [new_xyz, pointset],
            cuda_src=cuda_src
        )

        pc_shape = [batch_size_x, n_input, self.n_samples, 3]
        new_pointset = pointset.reindex(pc_shape, [
            'i0',
            '@e0(i0, i1, i2)',
            'i3',
        ], extras=[idxs])

        if feature is not None:
            feature_shape = [batch_size_x, n_input, self.n_samples, n_feature]
            new_feature = feature.reindex(feature_shape, [
                'i0',               # Batchid
                '@e0(i0, i1, i2)',  # Nid
                'i3',               # Featureid
            ], extras=[idxs])
        else:
            new_feature = None

        if self.use_xyz:
            local_xyz = new_pointset - new_xyz.unsqueeze(dim=2)
            if new_feature is not None:
                new_feature = jt.contrib.concat([local_xyz, new_feature], dim=-1)
            else:
                new_feature = local_xyz

        return new_feature


class GroupAll(nn.Module):
    def __init__(self, use_xyz):
        super().__init__()
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        if self.use_xyz:
            new_feature = jt.contrib.concat([pointset, feature], dim=-1)
        new_feature = new_feature.unsqueeze(dim=1) # [B, 1, N, C]
        return new_feature


class KNN(nn.Module):
    def __init__(self, k):
        self.k = k
        self.cuda_inc= """
        #undef out
        #include "helper_cuda.h" 

        __global__ void compute_distances(float * ref,
                                        int     ref_width,
                                        int     ref_pitch,
                                        float * query,
                                        int     query_width,
                                        int     query_pitch,
                                        int     height,
                                        float * dist) {

            // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
            const int BLOCK_DIM = 16;
            __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
            __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

            // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
            __shared__ int begin_A;
            __shared__ int begin_B;
            __shared__ int step_A;
            __shared__ int step_B;
            __shared__ int end_A;

            // Thread index
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int batch_id = blockIdx.z;

            // Initializarion of the SSD for the current thread
            float ssd = 0.f;

            // Loop parameters
            begin_A = BLOCK_DIM * blockIdx.y;
            begin_B = BLOCK_DIM * blockIdx.x;
            step_A  = BLOCK_DIM * ref_pitch;
            step_B  = BLOCK_DIM * query_pitch;
            end_A   = begin_A + (height-1) * ref_pitch;

            // Conditions
            int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
            int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
            int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

            // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
            for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

                // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
                if (a/ref_pitch + ty < height) {
                    shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx + batch_id * height * ref_pitch] : 0;
                    shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx + batch_id * height * query_pitch] : 0;
                }
                else {
                    shared_A[ty][tx] = 0;
                    shared_B[ty][tx] = 0;
                }

                // Synchronize to make sure the matrices are loaded
                __syncthreads();

                // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
                if (cond2 && cond1) {
                    for (int k = 0; k < BLOCK_DIM; ++k){
                        float tmp = shared_A[k][ty] - shared_B[k][tx];
                        ssd += tmp*tmp;
                    }
                }

                // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
                __syncthreads();
            }

            // Write the block sub-matrix to device memory; each thread writes one element
            if (cond2 && cond1) {
                dist[ (begin_A + ty) * query_pitch + begin_B + tx + batch_id * ref_pitch * query_pitch ] = ssd;
            }
        }

        __global__ void modified_insertion_sort(float * dist,
                                                int     ref_pitch,
                                                int *   index,
                                                int     index_pitch,
                                                int     width,
                                                int     height,
                                                int     k){

            // Column position
            unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
            int batch_id = blockIdx.z ;
            

            // Do nothing if we are out of bounds
            if (xIndex < width) {

                // Pointer shift
                float * p_dist  = dist  + xIndex + batch_id * ref_pitch * index_pitch;
                int *   p_index = index + xIndex + batch_id * index_pitch * k;

                // Initialise the first index
                p_index[0] = 0;

                // Go through all points
                for (int i=1; i<height; ++i) {

                    // Store current distance and associated index
                    float curr_dist = p_dist[i*index_pitch];
                    int   curr_index  = i;

                    // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
                    if (i >= k && curr_dist >= p_dist[(k-1)*index_pitch]) {
                        continue;
                    }

                    // Shift values (and indexes) higher that the current distance to the right
                    int j = min(i, k-1);
                    while (j > 0 && p_dist[(j-1)*index_pitch] > curr_dist) {
                        p_dist[j*index_pitch]   = p_dist[(j-1)*index_pitch];
                        p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                        --j;
                    }

                    // Write the current distance and index at their position
                    p_dist[j*index_pitch]   = curr_dist;
                    p_index[j*index_pitch] = curr_index; 
                }
            }
        }

            __global__ void compute_sqrt(float * dist, int width, int pitch, int k){
                unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
                int batch_id = blockIdx.z;
                if (xIndex<width && yIndex<k)
                    dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
            }

           inline static bool knn_cuda_global(
               int batch_size, 
               float * ref,
                    int           ref_nb,
               float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     int *         knn_index, 
                     float *  tmp_dist ){

            // Constants
            const int BLOCK_DIM = 16;

            const unsigned int size_of_float = sizeof(float);
            const unsigned int size_of_int   = sizeof(int);

            // Return variables
            cudaError_t err0, err1, err2, err3;

            // Allocate global memory
            float * ref_dev   = ref;
            float * query_dev = query;
            float * dist_dev  = tmp_dist;
            int   * index_dev = knn_index;

            // Deduce pitch values
            size_t ref_pitch   = ref_nb; 
            size_t query_pitch = query_nb;
            size_t dist_pitch  = query_nb; 
            size_t index_pitch = query_nb; 

            // Compute the squared Euclidean distances
            dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
            dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, batch_size);
            if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
            if (ref_nb   % BLOCK_DIM != 0) grid0.y += 1;


            // printf("%d", cudaDeviceSynchronize()); 
            // checkCudaErrors(cudaDeviceSynchronize());
            // printf(" before compute_distances \\n");

            compute_distances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev);
            // checkCudaErrors(cudaDeviceSynchronize());

            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" after compute_distances \\n");

            // Sort the distances with their respective indexes
            dim3 block1(256, 1, 1);
            dim3 grid1(query_nb / 256, 1, batch_size);
            if (query_nb % 256 != 0) grid1.x += 1;
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" before modified_insertion_sort \\n");
            // checkCudaErrors(cudaDeviceSynchronize());

            modified_insertion_sort<<<grid1, block1>>>(dist_dev, ref_pitch, index_dev, index_pitch, query_nb, ref_nb, k);

            // checkCudaErrors(cudaDeviceSynchronize());
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" after modified_insertion_sort \\n");
            
            // Compute the square root of the k smallest distances
            //dim3 block2(16, 16, 1);
            //dim3 grid2(query_nb / 16, k / 16, batch_size);
            //if (query_nb % 16 != 0) grid2.x += 1;
            //if (k % 16 != 0)        grid2.y += 1;
            //compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);	


            // Copy k smallest distances / indexes from the device to the host
            // TODO: batch 2d copy dist
            // cudaMemcpy2DAsync(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch*size_of_float,  query_nb * size_of_float, k, cudaMemcpyDefault);

            return true;
        }


        """
        self.cuda_src = '''
            const int k = out0_shape1;
            const int query_nb = in1_shape2; 
            const int ref_nb = in0_shape2;
            const int dim = in0_shape1;
            const int batch_size = in0_shape0;
            knn_cuda_global(batch_size, in0_p, ref_nb, in1_p, query_nb, dim, k, out0_p, in2_p);
        '''

    def execute(self, x_q, x_r): # n_points, c_dim
        batch_size, c_dim, q_points = x_q.shape 
        batch_size, c_dim, r_points = x_r.shape 
        out_idx_shapes = [batch_size, self.k, q_points]
        tmp_dist = jt.empty((batch_size, r_points, q_points), "float32")
        idxs,  = jt.code(
            [out_idx_shapes],
            ['int32'],
            [x_r, x_q, tmp_dist], # in0 r point in1 q point 
            cuda_src=self.cuda_src,
            cuda_header=self.cuda_inc,
        )
        return idxs



def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim<0:
        dim+=input.ndim

    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.transpose(transpose_dims)
    index,values = jt.argsort(input,dim=0,descending=largest)
    indices = index[:k]
    values = values[:k]
    indices = indices.transpose(transpose_dims)
    values = values.transpose(transpose_dims)
    return [values,indices]


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # print (src.size(), dst.size())
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * jt.matmul(src, dst.permute(0, 2, 1))
    dist += jt.sum(src ** 2, -1).view(B, N, 1)
    dist += jt.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.arange(B, dtype='l')
    batch_indices = jt.array(batch_indices).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    # print ('new xyz size, xyz size =',new_xyz.size(), xyz.size())
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def knn (x, k):
    inner = -2 * jt.nn.bmm(x.transpose(0, 2, 1), x)
    xx = jt.sum(x ** 2, dim = 1, keepdims=True)
    distance = -xx - inner - xx.transpose(0, 2, 1)
    idx = topk(distance ,k=k, dim=-1)[1]
    return idx

