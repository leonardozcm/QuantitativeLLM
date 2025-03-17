# SGEMM
(!!!本文的latex公式请在本地markdown编辑器上查看！！！)

本文作为李少侠的这篇[[施工中] CUDA GEMM 理论性能分析与 kernel 优化](https://zhuanlan.zhihu.com/p/441146275)与Pzzzzz的这篇[传统 CUDA GEMM 不完全指北](https://zhuanlan.zhihu.com/p/584236348)的补充，脉络上P老师这篇是对李老师的这篇文章的一些细节的补足和扩展，同时P老师也提出了自己的一些疑问。本文的书写目的是在两位老师的基础上进一步完善一些个人可能更在意的点，例如M_BLOCK,M_THREAD等如何从数学上近似求出取值范围，share memory中的内存排布设计原理以及如何结合NCU工具来辅助确定下一步优化的重点。一些前人阐述过的过程这里会指出引用位置。另外本文的书写顺序是按照我的实现顺序记录的，也算是从零开始实现的教程。

# 项目目前进度

本文测试矩阵乘规格为 A(M 4096 * K 4096) * B(K 4096 * N 4096) = C (M 4096 * N 4096),需要注意的是，我选用的A是row major的内存排布，即K是最低维，而B是column major，即K是最低维，C是row major。至于为什么这样选择，是因为后续计划写Q40的矩阵乘法，Q40的weights是column major，可以借鉴这个案例。
目前推进到做好了从Global Memory到smem，smem到reg到tiling，规避了所有的bank conflict，还没做的是double buffer。

<div style="text-align: center;">
  <img src="../../images/image-1.png" alt="sgemm roofline analysis, left is what we are going to implement, right comes from cublass" />
</div>
<center><p>sgemm roofline analysis, left is what we are going to implement, right comes from cublass</p></center>

## 量化计算

### 准备工作
我的机器设备是RTX2070s，为了开展我们的理论计算，我们列出了如下规格：
![alt text](../../images/image-3.png)
<center><p>NCU给出的硬件的最大支持</p></center>
以及一些来自李老师提供的测量[mircobenchmark](https://github.com/Yinghan-Li/YHs_Sample/tree/master/cuda/microbenchmark)的工具:

|                      |                      |
| -------------------- | -------------------- |
| Number of SMs        | 40                   |
| Number of warpSize   | 32                   |
| L2 Cache size:       | 4096 KB              |
| smem latency         | 22 cycles            |
| shared memory bandwidth per SM (measured)     | 111.734879 byte/cycle            |
| shared memory bandwidth per SM (theoretical)   | 128 byte/cycle            |
| L1 cache latency     | 32 cycles            |
| L2 cache latency     | 214 cycles           |
| DRAM latency         | 471 cycles           |
| FMA latency          | 4 cycles           |
| smem cache bandwidth | 9139.200195 GB/s     |
| L2 cache bandwidth   | 1811.228600GB/s      |
| DRAM bandwidth       | read 311.405838GB/s  |
|                      | write 369.402789GB/s |
| Peak TFLOPS          | 7.838 TFLOPS         |

### 基本流程与变量定义
在cuda编程的世界里面访存是第一公民，做矩阵乘法优化的核心手段就是通过L2 cache, smem和register三层cache来缓存近期会重复利用的内存，从而减少对Global Memory的访存次数。我们定义 M_BLOCK, N_BLOCK, K_BLOCK, M_THREAD, N_THREAD，他们的含义为:

|                      |                      |
| -------------------- | -------------------- |
| M_BLOCK        | M范围上每个BLOCK负责的区域大小                   |
| N_BLOCK   | N范围上每个BLOCK负责的区域大小                   |
| K_BLOCK       | K范围上每个BLOCK每次迭代的步长大小              |
| M_THREAD        | M范围上每个THREAD负责的区域大小            |
| N_THREAD     | N范围上每个THREAD负责的区域大小            |

通俗一点来说，每个BLOCK负责计算 M_BLOCK * N_BLOCK 大小的子矩阵，每个THREAD负责计算 M_THREAD * N_THREAD 的子矩阵，每次一个BLOCK缓存 (M_BLOCK + N_BLOCK) * K_BLOCK 大小的矩阵。这里借来P老师的图(有侵权请联系我):

![pipeline](https://pica.zhimg.com/v2-08d3770ed51bbed3dc86203032e01b3a_1440w.jpg)

那么这里迎来了本文将解决的第一个问题，我们如何确定这五个变量的值，或者在实现前无法直接地得出具体的值，至少我们可以缩小变量值的搜索范围也是好的。我们将通过例举变量的不等式和假设来完成这一过程。

### 约束关系

#### GLOBAL MEMORY的计算访存比

我们以一个BLOCK为单位来研究一个BLOCK内的计算访存比，首先一个block每次迭代的计算量为 $ M_{BLOCK}*K_{BLOCK}*N_{BLOCK}*2 $ ，访存量为 $(M_{BLOCK}+N_{BLOCK})*K_{BLOCK}*sizeof(fp32)$ ，那么其计算访存比为 $\frac{M_{BLOCK}*N_{BLOCK}}{(M_{BLOCK}+N_{BLOCK})*sizeof(fp32)}=\frac{1}{4*(\frac{1}{M_{BLOCK}}+\frac{1}{N_{BLOCK}})}$ 。GEMM作为典型的计算boundary的算子，我们期待它的行为是计算时间和访存时间基本覆盖，为了供应计算的需求可以充分利用各级带宽。首先我们保守一点，对于global memory的带宽311.405 GB/s，假设计算单元和带宽跑满时计算强度 $I=\frac{7.838 TFLOPS}{311.405 GB/s}=25.77$ ，也即是计算访存比大于25.77时就有希望可以跑满带宽，这还是在不考虑L2 cache的存在的情况下，每次load都会miss cache的结果。如果我们考虑L2 cache的情况，M，K，N足够大到可以忽略第一次load进L2 cache时候的cache miss，同时L2 cache无限大，我们可以求出对L2的计算强度bar为 $I=\frac{7.838 TFLOPS}{1811.22 GB/s}=4.43$ 。当然，假设L2 cache足够大和MKN足够大的情况很难成立，这里的计算得出的公式是为了每个BLOCK的计算访存比兜底，并尽可能地最大化计算访存比。额外的，我们再假定L2 cache的命中率为$L2_{hitrate}$时，L2和DRAM的等效带宽为 $BW_{avg}=311.405*(1-L2_{hitrate})+1811.22*L2_{hitrate} GB/s$ 最大化利用带宽的计算强度为 $I_{avg}=\frac{7.838 TFLOPS}{BW_{avg}}$

综上，所以有:

最低要求的计算强度: 
```math

\frac{1}{4*(\frac{1}{M_{BLOCK}}+\frac{1}{N_{BLOCK}})} \geq I_{avg} \tag{1}

```

最好情况下全命中L2 cache的计算强度:

```math
\frac{1}{4*(\frac{1}{M_{BLOCK}}+\frac{1}{N_{BLOCK}})} \geq 25.77 \tag{2}
```

#### 物理线程的安排

我们假设一个block的组成为block<<<tx, ty>>>, 根据映射关系有：

```math
M_{blk}=t_x*M_{thd}\tag{3}
```

```math
N_{blk}=t_y*N_{thd}\tag{4}
```

这里简写了 $M_{blk}=M_{block}$ , $M_{thd}=M_{thread}$ ，后文将同样承袭这样的写法。

#### 寄存器，SMEM和thread的占用率

这里是基本的求占用率问题，我们再定义每个block可以最多同时运行的block数量为 $Num_{block}$ , 每个thread占用的寄存器为 $regs_{thd}$ :

*register 占用*

```math
Num_{block}*\frac{regs_{thd}*t_x*t_y}{65536} \leq 1 \tag{5}
```

*smem 占用*

```math
Num_{block}*\frac{(M_{blk}+N_{blk})*K_{blk}*sizeof(fp32)}{32768} \leq 1 \tag{6}
```

*thread 占用*

```math
Num_{block}*\frac{t_x*t_y}{1024} \leq 1 \tag{7}
```

#### Break Time

目前为止我们好像列出来几个算式又引入了几个新的变量，我们知道正常情况下N个变量对应N个算式才能求解，看起来我们在越走越远。所以我们需要引入新的约束条件才能继续我们的工作。下面我们将从smem级别的访存计算指令调度延时的角度继续看看有什么是我们可以做的。

#### smem级别的访存计算指令调度

这一节李老师的文章的1.3节后半部已经阐述地很清晰了，基本的原理是warp向LSU发送访存请求本身需要一个周期，而拿到数据这件事本身也是有延时的，那么在这段访存的时间内我们是不是可以调度FMA来做一些计算的工作来充分利用warp scheduler发射指令，保证每个周期平均发射的指令数，让各个单元不要闲下来。题外话，这也是为什么Occupancy是一个重要的考量指标但是有时候不能完全盯着Occupancy看的原因，占用率高的潜台词是scheduler有更多的机会可以发射ready状态的warp指令来隐藏延时，如果有增加指令平均SM发射指令数的方法但是Occupancy会降低，那也是值得做的优化。

那么从数学上来量化，我们要做的事情是让每个thread计算指令的发射周期数加上FMA的延时的总周期数和访存的指令发射周期数加上LSU延时的总周期数之比，尽可能地大。我的RTX2070s隶属于turing架构，这个架构里面每个SM有4个warp scheduler，每个scheduler只有一个Dispatch unit与16个full-throughput单元，因此应该一个warp(32 threads)的指令需要dispatch两次，即[两个cycles内发射完](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#instruction-scheduling)。

> 向量内积 v.s. 向量外积
>
> 我们需要考量的另一个问题是，一次迭代中我们分别从A和B中拿到了M_thread和N_thread个数据，两个向量我们是应该选择向量内积还是向量外积求和呢？
>
> 其实这里P老师文章的Thread级别的优化这一章已经讲得很清楚了，这里强调一点两种方式访存模式的根本区别在于：
> 
> *每轮迭代中内积需要对相同的B tile重复多读M次，而外积用寄存器空间换取了时间，一次迭代相同的B tile只需要读1次。*
>
> 在满带宽的情况下，一个SM有16*4=64个FMA单元，故一个周期内可完成64个FMA指令（考虑一个周期的时间切片）。smem的出口理论带宽128 B/cycle，实际只有111.7 B/cycle左右。从一个warp的角度考虑（因为warp是最小执行单位），一个周期可以产生32次FFMA指令，每次需要2个float长度的数据，也就是8字节，故总共需要256B的smem的访存量，这么多数据的访存需要 256B /128(B/cycle)=2 cycles来从smem的出口出去。而64个FMA指令需要的数据量需要512B，需要4个cycles来读取，那么这个过程的计算访存比为1/4。这是我们不愿意看到的，因为理想情况下应该是计算可以覆盖访存，现在大部分的时间FMA都在饥饿。


每次迭代中，每个thread获取M_thread+N_thread个数据，向量外积的计算量为M_thread*N_thread，单次访存延时22 cycles，单次FFMA延时4 cycles。根据little's law，我们可以求出访存指令调度到拿到全部数据的延时为$2*(M_{thd}+N_{thd})+22$ cycles, 计算的指令调度到计算结束的延时为 $2*M_{thd}*N_{thd}+4$ ，这里的乘2的原因是上面提到的一个warp的指令需要两个周期发射。我们需要做的事情是使：

```math
\frac{2*M_{thd}*N_{thd}+4}{2*(M_{thd}+N_{thd})+22} \geq 1 \tag{8}
```

当然这只是基本要求，左式的值越大越好。当然，这里我们忽略了李老师提到的访存带宽带来的比例系数Alpha的问题，但是这里我们的目的是使左式的最大，所以在有限范围内求出最大的取值方法就够了。

#### 估算$regs_{thd}$

是的，我们还需要至少一个不等式才能开始我们的理论计算。我们在这里估算 $regs_{thd}$ 的大致值，A_reg和B_reg各需要M_THREAD和N_THREAD个寄存器，矩阵外积需要M_THREAD*N_THREAD个寄存器，流程控制预留32个寄存器，那么有

```math
regs_{thd} = M_{thd}+N_{thd}+M_{thd}*N_{thd}+32 \tag{9}
```

以上应该是我们能用上的所有算式了，其实还漏了一个保证L2 cache hit rate的不等式，但是在我们这种大矩阵乘语境下其实L2的hit rate很好满足，不构成一个急迫的需求。另外一个没有提及的计算是一个warp内thread mapping的一点规则，感兴趣的朋友可以去看李老师的1.3节。

### 实际计算
