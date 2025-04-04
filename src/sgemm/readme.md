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
> 我们来看看向量内积的情况下会发生什么：在满带宽的情况下，一个SM有16*4=64个FMA单元，故一个周期内可完成64个FMA指令（考虑一个周期的时间切片）。smem的出口理论带宽128 B/cycle，实际只有111.7 B/cycle左右。从一个warp的角度考虑（因为warp是最小执行单位），一个周期可以产生32次FFMA指令，每次需要2个float长度的数据，也就是8字节，故总共需要256B的smem的访存量，这么多数据的访存需要 256B /128(B/cycle)=2 cycles来从smem的出口出去。而64个FMA指令需要的数据量需要512B，需要4个cycles来读取，那么这个过程的计算访存比为1/4。这是我们不愿意看到的，因为理想情况下应该是计算可以覆盖访存，现在大部分的时间FMA都在饥饿。


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

好了，基本上我们确定参数取值范围需要的式子都在这里了，下一步就是开始我们的计算。

类比我们求有向无环图的出入节点，第一步我们从设计变量较少的算式开始，一步一步地带入其他式子。这一段参考李老师的1.3.2节（也可以参考上一节的引用部分），我们从式（8）开始分析，但$M_{thd}$和$N_{thd}$之和，即占用的寄存器数量$t_{sum}$维持在同一水平时，则有$N_{thd}=\frac{t_{sum}}{M_{thd}}$，带入式（8）左边，有：

```math
f(M_{thd}) = \frac{t_{sum}+2}{\frac{t_{sum}}{M_{thd}}+M_{thd}+11} \tag{10}
```

通过均值不等式，下面的分母在$M_{thd}=\sqrt{t_{sum}}$时取到最小值，即整个式子取到最大值。此时$M_{thd}==N_{thd}$。从这里我们可以知道，每个thread负责的矩阵范围接近正方形的时候，计算访存比最大。

规则1：这个结论可以推广到Block的范围，即$M_{thd}==N_{thd}$时候整个Block的计算访存比最大，最容易隐藏延时。这个结论非常重要，因为在这里我们要简化两个变量，后面的计算中我们就令$M_{thd}==N_{thd}$与$M_{thd}==N_{thd}$，由式（3）与式（4），我们得知$t_x==t_y$。

规则2：另外由结论1，M_thread和N_thread的绝对值越大，这个计算访存比的值也越大，因此我们应该尽量选择大的M_thread和N_thread。

![alt text](../../images/image-4.png)

然后我们从式（7）开始，可得：

```math
t_x * t_y = t_x^2 \leq \frac{1024}{Num_{block}} \tag{10}
```
当NUM_BLOCK=1时，tx<=32, 

|   NUM_BLOCK     |            tx         |
| -------------------- | -------------------- |
| 1        |    tx<=32              |
| 2        |    tx<22              |
| 3        |    tx<18              |
| 4        |    tx<=16              |

同时考虑到另外一个事实，对于计算瓶颈的算子，一个BLOCK不应该只有一个warp，否则在通过warp级别的instructions的调度隐藏延时方面会有问题，所以我们期待：

```math
t_x*t_y>=WARPSIZE(32) \tag{11}
```

同时我们也可以简化式（1）：
```math
M_{thd}*t_x \geq 8*I \tag{12}
```
这里取L2 cache 50% 命中率的等效带宽(1811+311.4)/2=1061GB/s, 对应I=7.5624。

综合（10）（11）有：

```math
5.65 \leq t_x \leq 32 \tag{13}
```

为了最大适应二进制体系的计算架构，我们从2的整数次方中取值，*tx可能的取值有8，16，32三个*。

再由式（5）有

```math
t_x^2 * (M_{thd}^2+2*M_{thd}+32) <= \frac{65536}{Num_{block}} \tag{14}
```

*tx=32*

当$t_x=32，Num_{block}=1$时，我们求得$M_{thd}<4.7$，我们取最大值4，带入式（8）求得左边等于36/38<1，隐藏延时很艰难，而且4这个值也很难隐藏合并访存（8个fp32合并一个sector）。我们这里排除掉tx=32这个解。

*tx=16*

当$t_x=16$时，由式（12）求得 

```math
M_{thd} \geq 3.78
```

当$Num_{block}=1$时，我们由式（14）求得$M_{thd}<14$，我们尽可能取最大值得8，带入（8）有左式等于2.44。
当$Num_{block}=2$时，我们由式（14）求得$M_{thd}<8.85$，我们尽可能取最大值得4，这个解同上舍去。

*tx=8*

当$t_x=8$时，由式（12）求得 

```math
M_{thd} \geq 7.5624
```

当$Num_{block}=1$时，我们由式（14）求得$M_{thd}<=30.511$，中间可以取值8、16，对应（8）的值有2.44和6。
当$Num_{block}=2$时，我们由式（14）求得$M_{thd}<=20.93$，中间可以取值8、16，对应（8）的值有2.44和6。
当$Num_{block}=3$时，我们由式（14）求得$M_{thd}<=17.61$，中间可以取值8、16，对应（8）的值有2.44和6。
当$Num_{block}=4$时，我们由式（14）求得$M_{thd}<=14$，中间可以取值8，对应（8）的值有2.44。

综上，我们现在手上的备选项有{M_THREAD, tx}={8,16}/{8,8}/{16,8},但是这三个选项来测试也是有优先级的，依据就是我们计算的（8）式的值从大到小。

-------------

我们现在还需确定K_BLOCK的值，这里我们需要用到式（6）

#### {M_THREAD, tx}={8,16}

```math
K_{blk} \leq \frac{32}{Num_{block}}
```

对应的$Num_{block}$取值有2和3,由于我为了解决可能发生的bank conflict需要padding，而且一般gemm的限制主要来自寄存器的压力而非smem，也就是说smem一般用不满，所以这里预留一些空间，不取等号。

$Num_{block}=2$

```math
K_{blk} < 16
```

$Num_{block}=3$

```math
K_{blk} < 10.67
```

因为K不影响计算强度，而我们又希望迭代次数越少越好，减少控制成本，所以这里K_BLOCK取8。


#### {M_THREAD, tx}={8,8}

```math
K_{blk} \leq \frac{64}{Num_{block}}
```

$Num_{block}=2$

```math
K_{blk} < 32
```

$Num_{block}=3$

```math
K_{blk} < 21.33
```

这里K_BLOCK取16。

#### {M_THREAD, tx}={16,8}

```math
K_{blk} \leq \frac{32}{Num_{block}}
```

$Num_{block}=1$

```math
K_{blk} < 64
```

这里K_BLOCK取32。

### 小结

*备选项*

|   M_BLOCK  |   N_BLOCK   |   K_BLOCK  |  M_THREAD    | N_THREAD    | tx   | ty   | 计算访存指令比 |
| -----------| ----------- | ---------- | ------------ | ----------- | ---- | ---- | ----------- |
|   128 |  128  | 8 | 8 | 8| 16 | 16 | 6 |
|   64 |  64  | 16 | 8 | 8| 8 | 8 | 2.44 |
|   128 | 128 | 32 |16 | 16 | 8 | 8| 2.44 |


OK，我们经历了一段不算轻松的旅程，好在我们通过合理假设和理论推算，从庞大的备选参数中排除了理论上就不合理的部分，最后只剩下了三组参数作为求解空间，前期的工作大大减少了我们的编程工作量，同时也可以帮我们定位到中间实现可能的性能瓶颈，磨刀不误砍柴工，量化计算的工作还是有必要的。

## smem内存排布

我们选取备选项（1）来实现初版的SGEMM，一个block由256个thread组成，即8个warp。考虑加载A和B的tile到smem的过程，我们发现从warp的角度加载DRAM到smem的映射关系还没有确定，但我们知道每个tile我们要加载128*8=1024个fp32，也即每个thread要加载4个fp32。同时P老师的Warp Tiling一节解释了warp_x和warp_y取8 * 4或者4 * 8时一个warp内的计算访存比最大，所以我们的warp的mapping如下：
![alt text](../../images/image-5.png)

整张图对应一个block负责的C_tile(128*128)，也就是说每个warp负责 32 * 64大小的一块区域。个人认为这样排和竖着排理论上应该没有区别，因为我们关注的过程有三个：A_tile、B_tile加载到smem，A_reg、B_reg从smem加载到寄存器，以及最后的向量外积。三个阶段的实现相互独立，在Tiling Size确定的情况下，在warps之间排布顺序这个粒度并没有什么影响计算需要注意的地方。所以接下来我们还剩三个排布问题急需解决：1. A_tile在smem里面的排布 2.B_tile在smem里面的排布 3.在warp内部32个thread的排布情况。我们也明确一下我们的最理想情况下的目的：合并访存是大前提，同时尽可能地解决Bank Conflict，几个thread同时访问同一个数据的情况下整个warp可以触发广播机制。

### A_tile的排布问题

很容易想到的是，我们在M和N的维度上做向量外积，往前推一步，一次迭代中，从smem到register获取的每个值都会被其他warp或者warp中的其他线程重复访问多次，再往前推一步，从DRAM加载到smem每个值只需要访问一次，那么我们是不是应该至少保证从smem加载到register这一步就合并访存（float4访问），从而减少MIO throttle的压力。为了合并访存，我们需要对A_tile在smem存成转置后的形态，即我们定义A_tile在smem里面的样子是[K_Block, M_Block]。
![alt text](../../images/image-6.png)

OK, 大致确定了A_tile和A_reg的存取问题，按照顺序我们就可以来看看我们怎么在向smem存A_tile的时候完成转置这一操作。

我们先看怎么安排256个线程8个warp加载一个128 * 8大小的矩阵。我们很自然地可以首先想到两个方案：

![alt text](../../images/image-7.png)

两个方案的区别在于一个warp负责的区域，方案A是16 * 8，方案B时32 * 4，两者的访存量是一致的。在这里我们说A更好，因为L2 cache读取细粒度是32 bytes，所以内存对齐的访问最好是32 bytes的倍数的，B方案每个warp的thread最多同时访问一行里面的4个fp32，也即16B，无法保证每次L2访存都是32B对齐的，这样就会造成访问量一半的浪费。

再看一个warp内部的排布情况。我们已经明确了每个thread访问一个float4，类似warp的排布方式我们有两种常见的排布：

![alt text](../../images/image-8.png)

因为对DRAM的访问是以warp为一个整体思考的，同时因为存进smem时要一个元素一个元素地转置，我们会以一个float的粒度（STS.32）来转置，也是以一个warp为整体来思考，那么A和B的方式就没有什么区别了。这里我们选用了方案A。以下是我们用float4转置代码的示意：

```cpp
// transpose and store to A slm blk
A_blk_tile[warp_idx][0*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<float4*>(A_thread_reg)->x;
A_blk_tile[warp_idx][1*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<float4*>(A_thread_reg)->y;
A_blk_tile[warp_idx][2*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<float4*>(A_thread_reg)->z;
A_blk_tile[warp_idx][3*WARP_SIZE+(warp_lane>>1)+A_thread_ld_offset_in_blk_x*16] = reinterpret_cast<float4*>(A_thread_reg)->w;

```

在存smem的时候我们发现一个问题，那就是t0和t1，t2和t3等等两两之间会发生严重的bank conflict如下：

![alt text](../../images/image-9.png)

这里有两种解决办法，一种是将128扩大到128+4，这样到奇数线程store到第4行的时候，t0和t1对应的bank刚好相差4*4=16，刚好和偶数线程store的bank错位开。

![alt text](../../images/image-11.png)

虽然这样会造成轻微的浪费，对于gemm这个场景来说smem是限制occupancy的主要因素，所以这点浪费不会影响性能。

另外一种方式是，将奇数线程的16个fp32存到偶数线程16个fp32的后面：

![alt text](../../images/image-12.png)

那么M在[16，31]范围内的值就被存到了第二行，这样做的好处是可以节约smem，缺点是取值时索引变换会变得很麻烦。本文采取了第二种方式。

### A_reg的load问题

从warp的排布我们可以算出，每个warp的32个线程要从A_tile读32个fp32，一个A_reg的大小为8个fp32，这32个fp32便分成4组，每组8个fp32。那么一个A_reg要由32/4=8个thread去读，我们需要确定一种映射关系使32个线程平均映射到4个组里面。为了使访问同一个组的线程可以最快拿到数据，我们希望组内的8个线程可以尽可能依赖广播机制在一个wavefront内拿到这组数据。

这里我们回忆一下float4(128bit)访存的wavefront机制：

对于128bit的访存一个warp最少也需要两个wave front，我们考虑合并访存也是以half warp作为单位来计算(参考[CUDA Shared Memory 在向量化指令下的访存机制](https://code.hitori.moe/post/cuda-shared-memory-access-mechanism-with-vectorized-instructions/#128-%E4%BD%8D%E5%AE%BD%E7%9A%84%E8%AE%BF%E5%AD%98%E6%8C%87%E4%BB%A4))：

> 对于 64 位宽的访存指令而言，除非触发广播机制，否则一个 Warp 中有多少个活跃的 Half-Warp 就需要多少个 Memory Transaction，一个 Half-Warp 活跃的定义是这个 Half-Warp 内有任意一个线程活跃。触发广播机制只需满足以下条件中的至少一个：
> 
> 对于 Warp 内所有活跃的第 i 号线程，第 i xor 1 号线程不活跃或者访存地址和其一致；
> 
> 对于 Warp 内所有活跃的第 i 号线程，第 i xor 2 号线程不活跃或者访存地址和其一致；
> 
> 如果触发了广播机制，那么两个 Half-Warp 内的 Memory Transaction 可以合并成一个。

> 128 位宽的访存指令和 64 位宽的访存指令是类似的，不同的是需要以 Half-Warp 为单位来计算，对于每个 Half-Warp 而言，除非触发广播机制，这个 Half-Warp 中有多少个活跃的 Quarter-Warp 就需要多少个 Memory Transaction，一个 Quarter-Warp 活跃的定义是这个 Quarter-Warp 内有任意一个线程活跃。类似地，如果触发广播机制那么两个 Quarter-Warp 中的 Transaction 就可以被合并成一个。 触发广播机制的条件和 64 位宽的访存指令是一样的（注意广播机制是以整个 Warp 为单位考虑）。这也就意味着假设一个 Warp 中 32 个线程都活跃，即使它们的访存地址都一样，也需要 2 个 Memory Transaction。

令addr(t_i)为线程t_i访问的float4的第一个元素的地址，则为了触发广播机制，以t0为例，我们希望有

```math
addr(t_0)=add(t_1) / addr(t_0)=add(t_2)
```

一个half warp(t0~t15)需访问16个fp32，即两组，在上一节的方案中这16个float32刚好是连续的。于是我们有三种方案：

![alt text](../../images/image-15.png)

在这些方案中，每一个Quarter Warp都触发了广播机制的第一条或第二条或者同时两条，一个halfwarp的访存触发广播机制，一个warp的访问只需要最少的2个wavefront（推荐结合上面链接的文章理解推导一下）。

但是有一点是可以确定的，那就是一个half warp(t0~t15)内的线程最好访问连续的16个fp32。原因是我们在load A_tile时交叉排布导致的，元素16～31和0～16处于不同行的同一个bank内，假设第一个half warp里面有一个quarter的线程访问了16~31，除非我们人为地干预使两个quarter交叉访问（也没有必要人为增加工作的复杂度），否则很容易发生bank conflict。所以无论是哪一种方式，half warp的thread一定是在组0和组1之内或者组2和组3之间:

![alt text](../../images/image-16.png)

这个结论很重要，在确定B_reg的排布问题时我们会利用这个结论排除一些不成立的选项。


### B_tile的排布问题和B_reg的load问题

看起来我们对B_tile做的事情和A_tile类似，也是需要转置一下，但是和A不同的是B_reg的分组映射关系和其是不同的，这会影响到转置策略的可用性。我们先考虑使用和A_tile同样的方案，将奇数线程的16个fp32存到偶数线程16个fp32的后面。

从warp的排布我们可以算出，每个warp的32个线程要从B_tile读64个fp32，一个B_reg的大小为8个fp32，这64个fp32便分成8组，每组8个fp32。那么一个B_reg要由32/8=4个thread去读，我们需要确定一种映射关系使32个线程平均映射到8个组里面。为了使访问同一个组的线程可以最快拿到数据，我们希望组内的4个线程可以尽可能依赖广播机制在一个wavefront内拿到这组数据，整个warp用最少的两个wavefront就可以拿到数据。

我们排出这八组在bank里面的相对位置

![alt text](../../images/image-17.png)

我们再看一下上一节的这张图：

![alt text](../../images/image-16.png)

从这两张图我们可以推理出来，t0~t15的这halfwarp必须覆盖到第一章图里面所有的reg才可以使整个计算成立。也就是说，halfwarp里面每两个thread对应一个reg这样读。在不做padding的情况下这可能在没有bank conflict的情况下实现吗？答案是不可能的。

![alt text](../../images/image-18.png)

我们把问题具象化一下，我们有16个球，为了触发广播机制，16个球两两绑定，相当于只有8组球，现在我的问题是我要把这8组球放入这16个格子，要求在每个竖向的格子只有一组球（避免bank conflict）。由抽屉原理，这是不可能的。

所以我们只能选择方案一，即B_tile必须通过padding来规避store以及load时候的Bank Conflict。

那么store to smem就不是一个问题了，现在我们再来讨论B_reg的load问题。

![alt text](../../images/image-19.png)

考虑一个half warp，我们可以平铺直叙地这样去映射我们的线程：

![alt text](../../images/image-21.png)

这样也有问题，因为这样t0,t1和t8,t9会发生Bank Conflict，可以尝试交换组内的线程排布，但是这样总是有Bank Conflict存在。NCU也显示我们仍然存在Bank Conflict：

![alt text](../../images/image-22.png)

解决办法也很简单，我们让t0,t1和t8,t9错位去加载，t0,t1加载前4个bank的数据，t8,t9加载后4个bank的数据：

![alt text](../../images/image-23.png)

第二次load的时候再交换回来就好，如此一来half warp内相邻线程触发广播机制，两个quarter warp之间没有Bank Conflict，我们理论上可以在2个wavefront就去发送一个warp的访存指令。

![alt text](../../images/image-25.png)

确实没有Bank Conflict，wavefront也减少了三分之一。

最后我们来看看我们warp内32个线程的排布是什么样的：

![alt text](../../images/image-26.png)

没错，这就是李老师的Z字排布广播速度快的由来，A和B都可以只用两个wavefront完成一个warp的register load。

## Double Buffer

做到这里我发现离cublass的实现还有一段距离，通过NCU的source code analysis，我发现有约15%的时间是在等待DRAM的数据到寄存器，这段时间计算单元完全没有利用起来。

![alt text](../../images/image-27.png)

我们想到，可以在上一次迭代进行计算前发出这一轮迭代的数据的请求命令，这样上一轮的计算时间就可以掩盖这一轮访存的时间。另外我发现，cublass的矩阵乘和我采用了同样的block size和grid size，却用了几乎两倍于我实现的smem，所以我有理由怀疑Double buffer就是最后一块拼图。

![alt text](../../images/image-28.png)
