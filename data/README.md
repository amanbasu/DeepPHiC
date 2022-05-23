# Data structure

The sequence, read, and distance features are stored as numpy arrays serialized using h5py. The keys for the h5 files are as follows:

1. 'seq1_neg' - negative sequence features of 1st region.
2. 'seq2_neg' - negative sequence features of 2nd region.
3. 'seq1_pos' - positive sequence features of 1st region.
4. 'seq2_pos' - positive sequence features of 2nd region.
5. 'read1_neg' - negative read features of 1st region.
6. 'read2_neg' - negative read features of 2nd region.
7. 'read1_pos' - positive read features of 1st region.
8. 'read2_pos' - positive read features of 2nd region.
9. 'dist1_neg' - negative distance features of 1st region.
10. 'dist2_neg' - negative distance features of 2nd region.
11. 'dist1_pos' - positive distance features of 1st region.
12. 'dist2_pos' - positive distance features of 2nd region.
