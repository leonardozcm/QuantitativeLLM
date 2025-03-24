# QuantitativeLLM
We optimize most of the operators used in LLM from scratch from the perspective of quantitative analysis of computer architecture. The programming language, framework, and even the final performance are not important. The most important thing is the thinking process of quantitative analysis.

## Experiment Device Parameters

### RTX2070s
```
Device count 1
Device name NVIDIA GeForce RTX 2070 SUPER
Amount of global memory: 7.60376 GB
Amount of total memory: 7.60376 GB avail memory: 7.50232 GB
Global memory bus width in bits:   256 bit
Compute capability:     7.5
Amount of constant memory:      64 KB
Maximum grid size:  2147483647 65535 65535
maximum block size:     1024 1024 64
Number of SMs:      40
Number of warpSize:      32
L2 Cache size:                             4096 KB
maximum l2 persisting lines capacity       0 B
Device supports caching globals in L1(Y/N) 1
Device supports caching locals in L1(Y/N)  1
Maximum amount of shared memory per block: 48 KB
Maximum amount of shared memory per SM:    64 KB
Maximum number of registers per block:     64 K
Maximum number of registers per SM:        64 K
Maximum number of threads per block:       1024
Maximum number of threads per SM:          1024
Each Turing SM includes 4 warp-scheduler units.
```
Hierarchical Memory MircoBenchmark:
```
smem latency 22 cycles
shared memory bandwidth per SM (measured): 111.734879 byte/cycle
shared memory bandwidth per SM (theoretical): 128 byte/cycle

L1 cache latency 32 cycles
L2 cache latency 214 cycles
DRAM latency 471 cycles

smem cache bandwidth 9139.200195 GB/s
L2 cache bandwidth 1811.228600GB/s
DRAM bandwidth:
 4MB (r+w)
read 339.508581GB/s
write 349.381258GB/s
copy 347.432216GB/s
---------------------------
8MB (r+w)
read 292.987497GB/s
write 336.411597GB/s
copy 326.229843GB/s
---------------------------
16MB (r+w)
read 302.639292GB/s
write 342.583800GB/s
copy 323.416465GB/s
---------------------------
32MB (r+w)
read 307.698924GB/s
write 348.205236GB/s
copy 325.219455GB/s
---------------------------
64MB (r+w)
read 311.405838GB/s
write 369.402789GB/s
copy 364.961634GB/s
---------------------------
128MB (r+w)
read 333.970297GB/s
write 394.742961GB/s
copy 360.514151GB/s
---------------------------
256MB (r+w)
read 365.683390GB/s
write 394.932527GB/s
copy 360.524464GB/s
---------------------------
512MB (r+w)
read 381.225616GB/s
write 395.261208GB/s
copy 360.617465GB/s
---------------------------
1024MB (r+w)
read 389.571450GB/s
write 395.442778GB/s
copy 360.661205GB/s
---------------------------
```
And NCU shows that its peak performance is 7.838 TFLOPS.