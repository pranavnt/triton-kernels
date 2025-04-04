"""Matrix Transpose Optimization - V1: Row-wise Partitioning

- Each thread block handles a row of the matrix
- Performance: 3.65 ms
- Poor performance due to uncoalesced memory accesses
"""