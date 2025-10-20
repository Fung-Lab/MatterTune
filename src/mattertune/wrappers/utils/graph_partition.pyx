#cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_2_2_API_VERSION

"""
Cython accelerated functions for partitioning
numpy <= 2.2.2
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport floor
from libc.string cimport memset
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list BFS_extension(int num_nodes,
                             np.ndarray edge_indices,
                             list partitions,
                             int mp_steps):

    cdef int n_edges = edge_indices.shape[1]
    cdef np.ndarray[np.int32_t, ndim=2] edges = edge_indices

    cdef int i, j, k, src, dst
    cdef int* counts = <int*> malloc(num_nodes * sizeof(int))
    if counts == NULL:
        raise MemoryError()
    memset(counts, 0, num_nodes * sizeof(int))
    for i in range(n_edges):
        src = edges[0, i]
        counts[src] += 1

    cdef int** neighbors = <int**> malloc(num_nodes * sizeof(int*))
    cdef int* adj_len = <int*> malloc(num_nodes * sizeof(int))
    if neighbors == NULL or adj_len == NULL:
        free(counts)
        raise MemoryError()
    for i in range(num_nodes):
        adj_len[i] = counts[i]
        if counts[i] > 0:
            neighbors[i] = <int*> malloc(adj_len[i] * sizeof(int))
            if neighbors[i] == NULL:
                raise MemoryError()
        else:
            neighbors[i] = NULL
        counts[i] = 0

    for i in range(n_edges):
        src = edges[0, i]
        dst = edges[1, i]
        neighbors[src][counts[src]] = dst
        counts[src] += 1

    free(counts)

    cdef list result = []
    cdef int* queue = <int*> malloc(num_nodes * sizeof(int))
    cdef char* visited = <char*> malloc(num_nodes * sizeof(char))
    cdef int* ext_nodes = <int*> malloc(num_nodes * sizeof(int))
    if queue == NULL or visited == NULL or ext_nodes == NULL:
        for i in range(num_nodes):
            if neighbors[i] != NULL:
                free(neighbors[i])
        free(neighbors)
        free(adj_len)
        raise MemoryError()

    cdef int partition_index, node, neighbor, level_count, depth
    cdef int head, tail, ext_count, current

    cdef list part

    for partition_index in range(len(partitions)):
        part = partitions[partition_index]
        memset(visited, 0, num_nodes * sizeof(char))
        head = 0
        tail = 0
        ext_count = 0

        for node_obj in part:
            node = <int> node_obj
            if visited[node] == 0:
                visited[node] = 1
                queue[tail] = node
                tail += 1
                ext_nodes[ext_count] = node
                ext_count += 1

        depth = 0
        while head < tail and depth < mp_steps:
            level_count = tail - head
            for i in range(level_count):
                current = queue[head]
                head += 1
                for k in range(adj_len[current]):
                    neighbor = neighbors[current][k]
                    if neighbor < 0 or neighbor >= num_nodes:
                        continue
                    if visited[neighbor] == 0:
                        visited[neighbor] = 1
                        queue[tail] = neighbor
                        tail += 1
                        ext_nodes[ext_count] = neighbor
                        ext_count += 1
            depth += 1

        result.append([ext_nodes[i] for i in range(ext_count)])
    free(queue)
    free(visited)
    free(ext_nodes)
    for i in range(num_nodes):
        if neighbors[i] != NULL:
            free(neighbors[i])
    free(neighbors)
    free(adj_len)

    return result


cdef struct PartitionArray:
    int* data 
    int count
    int capacity

cpdef list grid_partition(int num_nodes,
                          np.ndarray[double, ndim=2] scaled_positions,
                          tuple granularity):
    cdef int gx = granularity[0]
    cdef int gy = granularity[1]
    cdef int gz = granularity[2]

    cdef int partitions_count = gx * gy * gz
    cdef int i, j, partition_index, ix, iy, iz
    cdef double x, y, z, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, val

    cdef double[:, :] pos = scaled_positions

    for i in range(num_nodes):
        for j in range(3):
            pos[i, j] = pos[i, j] - floor(pos[i, j])
    
    x_min = pos[0, 0]; y_min = pos[0, 1]; z_min = pos[0, 2]
    x_max = pos[0, 0]; y_max = pos[0, 1]; z_max = pos[0, 2]
    for i in range(1, num_nodes):
        val = pos[i, 0]
        if val < x_min:
            x_min = val
        elif val > x_max:
            x_max = val
        val = pos[i, 1]
        if val < y_min:
            y_min = val
        elif val > y_max:
            y_max = val
        val = pos[i, 2]
        if val < z_min:
            z_min = val
        elif val > z_max:
            z_max = val
    
    dx = (x_max - x_min) / gx if gx > 0 else 1.0
    dy = (y_max - y_min) / gy if gy > 0 else 1.0
    dz = (z_max - z_min) / gz if gz > 0 else 1.0

    cdef PartitionArray* parts = <PartitionArray*> malloc(partitions_count * sizeof(PartitionArray))
    if parts == NULL:
        raise MemoryError()
    cdef int initial_capacity = 16
    cdef int k
    for i in range(partitions_count):
        parts[i].count = 0
        parts[i].capacity = initial_capacity
        parts[i].data = <int*> malloc(initial_capacity * sizeof(int))
        if parts[i].data == NULL:
            for k in range(i):
                free(parts[k].data)
            free(parts)
            raise MemoryError()
    
    cdef PartitionArray* part_ptr
    for i in range(num_nodes):
        x = pos[i, 0]; y = pos[i, 1]; z = pos[i, 2]
        
        if dx == 0:
            ix = 0
        else:
            ix = <int>((x - x_min) / dx)
        if gx > 0 and ix >= gx:
            ix = ix % gx
        
        if dy == 0:
            iy = 0
        else:
            iy = <int>((y - y_min) / dy)
        if gy > 0 and iy >= gy:
            iy = iy % gy
        
        if dz == 0:
            iz = 0
        else:
            iz = <int>((z - z_min) / dz)
        if gz > 0 and iz >= gz:
            iz = iz % gz
        
        partition_index = ix * (gy * gz) + iy * gz + iz
        
        part_ptr = &parts[partition_index]
        if part_ptr.count >= part_ptr.capacity:
            part_ptr.capacity *= 2
            part_ptr.data = <int*> realloc(part_ptr.data, part_ptr.capacity * sizeof(int))
            if part_ptr.data == NULL:
                for k in range(partitions_count):
                    if parts[k].data != NULL:
                        free(parts[k].data)
                free(parts)
                raise MemoryError()
        part_ptr.data[part_ptr.count] = i
        part_ptr.count += 1
    
    cdef list py_partitions = []
    cdef int count
    for i in range(partitions_count):
        count = parts[i].count
        lst = [0] * count
        for k in range(count):
            lst[k] = parts[i].data[k]
        py_partitions.append(lst)
    
    for i in range(partitions_count):
        free(parts[i].data)
    free(parts)
    
    return py_partitions