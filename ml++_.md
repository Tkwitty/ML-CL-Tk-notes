**知识归纳**



[TOC]







#### 知识盲点：



​	（ML实战书、西瓜书、蓝皮书、深度书/、AI导论）

​	蒙特卡洛方法与 MCMC

​	隐马尔可夫模型 HMM

​	EM & 混合高斯模型



------



​	·矩阵分解

​		一种降维方式，提取数据关键信息，发现隐藏的数据特征

​		包括 SVD分解(b)，QR分解，LU分解，LDLT分解 ...

​		PCA：样本集X的协方差矩阵的最大k个特征值，其k个特征向量矩阵M，则X降维得 XM

​		SVD：任意样本集A，可分解为 A=U∑V，求解奇异值对应的奇异矩阵∑'

​		QR：针对非奇异满秩矩阵，可直接分解为 A=QR

​		LU：分解为一个下三角与一个上三角矩阵的乘积；行列式不为0，LU分解总是存在

​			1.对矩阵A通过初等行变换将其变为一个上三角矩阵U，对角线上是主元；

​			2.将原始矩阵A变为下三角矩阵L，对角线上是1；



​	·贝叶斯网络 (信念网络/因果网络)Bayesian Networks --- 推理过程

​		DBN【动态贝叶斯网络(DynamicBN)---深度信念网络DBN(Deep Belief Network】的区别

​	·HMM· 维特比算法：

​		一种动态规划算法，用于寻找最有可能产生的观测序列的隐含状态序列，即求解图的概率最大路径

​	·马尔科夫随机场 MRF --- 条件随机场 CRF

​		马尔科夫随机场 Markov random field 

​			即概率无向图模型 Probabilistic undirected graphical model，是可以由无向图表示的联合概率分布

​		条件随机场 Conditional random field

​			是给定随机变量X时，Y的马尔科夫随机场。一般研究 线性链条件随机场。



​	·关联分析-Apriori，FP-growth算法

​	·GBDT，lightGBM

​	·聚类：层次聚类，均值漂移，模糊聚类，SOM自组织网络

​	·KL散度、伪逆法、势函数...

​	·卡尔曼滤波，粒子滤波

​	·马尔科夫决策链，贝尔曼方程---强化学习



​	·径向基函数/高斯核函数：

​		Radical Basis Function 径向基函数/高斯核函数 可以用于 

​		对任意维度的离散/连续数据点集进行插值与运算从而进行曲线拟合

​	

​	·最大熵模型：

​		考虑所有已知约束条件的情况下，对于未知不做任何假设，以均匀概率分布(熵最大)来估计模型参数

​		在条件熵最大化函数中，利用拉格朗日乘数法，利用极大似然函数，求解未知参数最优解

​		优·对于经典分类模型准确率高；缺·对于算法其迭代过程计算量巨大，实际应用较难



​	·协同过滤

​		基于用户/物品的样本矩阵，计算各向量间的相似度，根据相似度门槛过滤出推荐参考

​		再计算推荐度【相当评分预测】



​	·知识图谱、社交图谱、用户画像，基于**图谱的搜索，推荐策略



------



​	--配分函数，泛函 - 变分推断

​	--度量学习，表示学习-字典学习

​	--计算学习理论，假设空间，VC维：

​		研究分析机器学习的理论基础，为学习算法提供理论保证，指导算法设计

​		泛化误差：x服从D分布情况下，学习的模型h预测y出错的概率。

​		经验误差：x服从D分布情况下，对于某一样本集，学习的模型预测y的错误率。

​		常用不等式：

​			·Jensen不等式：任意凸函数f(x),有 f(E(x)) <= E(f(x)	[凸映射小于期望]

​			·Hoeffding不等式：霍夫丁不等式

​			·McDiarmid不等式：

​		PAC 学习：概率近似正确学习理论， Probably Approximately Correct

​			PAC辨识，PAC可学习，PAC学习算法，样本复杂度

​			若PAC学习中 假设空间H与概念类C完全相同，即H=C称为 恰PAC可学习

​			|H|有限时称为“有限假设空间”，否则称为“无限假设空间”

​		

​			增长函数：假设空间H 对 m个实例所能赋予标记的最大可能结果数

​			对分：假设空间H 对数据集D的每种可能标记结果称为对D的一种对分（dichotomy）

​			打散：假设空间H 能实现数据集D上全部实例的对分【能被学习模型正确划分】

​		VC维：能被 假设空间H 打散的最大 数据集D 大小

​			2D线性分类器的VC维为3，即2D线性分类器最多可以打散大小为3的样本集



------



技术需求：

	机器学习 天池竞赛、kaggle竞赛

	hadoop、spark、MLlib、大数据分布式存储

	分布式深度学习、GPU/CUDA并行计算



------



------



------



#### 回归方法总结：	---线性/非线性、多项式、logistic



​	Ridge 岭回归：回归问题中权重参数w=(XtX)^-1·Xty

​		如果数据特征n比样本数m多，n>m，则权重参数无法计算，因为XtX不是满秩矩阵，不可逆

​		为此引入领回归，w=(XtX+λI)^-1·Xty， λ为岭系数，I为单位阵

​		==》其实就是在最小二乘损失函数上增加了L2正则化项的 回归

​	LASSO 回归：同领回归，其实就是在最小二乘损失函数上增加了L1正则化项的 回归

​	GLASSO：group lasso

​	Elastic Net 弹性网：同xx回归，在最小二乘损失函数上增加的正则化项为：

​		λ∑{αθ^2+(1-α)|θ|}

​		结合了领回归和Lasso回归的全体权重参数的取值限制：∑L?(θ) <= t

​	最小角回归：



#### 激活函数总结：	---将神经元输出值映射到[0-1]之间，以便后续神经元的值传递



​	Sigmoid

​	tanh

​	Relu	

​	Leaky Relu

​	Softmax

​	ELU

​	MaxOut

​	



#### 核函数总结:



​	线性核函数

​	多项式核函数

​	径向基/高斯核 函数



#### Normalization-数据的伸缩不变性：	---【独立，同分布  化】



​	Batch Norm;

​	Layer Norm;

​	Weight Norm;

​	Cosine Norm



#### 损失函数总结：	



> 根据 y-pred & y-truth 的偏差（损失）最小化目标 来更新模型的权重参数

> 目标函数 J ，通常表示为 损失项 & 正则项 的和：

> $$

> J\left( w \right) =\sum_i{L\left( m_i\left( w \right) \right) +\lambda R\left( w \right)}

> \\

> y^{\left( i \right)}\text{为真实值，}f_w\left( x^{\left( i \right)} \right) =W^Tx^{\left( i \right)}\text{为预测值，定义}

> \\

> m_i=y^{\left( i \right)}\cdot f_w\left( x^{\left( i \right)} \right) 

> $$

> 记 m = yf(x)



​	0-1损失函数：

$$

L_{0-1}\left( m \right) =\begin{cases}

	\text{0   }if\,\,m\geqslant 0\\

	\text{1   }if\,\,m<0\\

\end{cases}

$$

​		在二分类任务中，输出值y的 正负号对应 类别，即预测正确时L值为1，否则为0, 等价于

$$

L_{0-1}\left( m \right) =\frac{1}{2}\left( 1-sign\left( m \right) \right) \text{，其中}sign\left( x \right) =\begin{cases}

	\text{1   ,}x>0\\

	\text{0   ,}x=0\\

	-\text{1   ,}x<0\\

\end{cases}

$$

​	Log对数损失函数：

$$

L=\log \left( 1+\exp \left( -m \right) \right)

$$

​	**对数似然损失函数**：【Logistic回归】

$$

\text{记}sigmoid\text{函数：}\sigma \left( x \right) =\frac{1}{1+\text{e}^{-x}}

\\

\text{似然函数：}\mathscr{L}\left( w \right) =\prod_{i=1}^n{\sigma \left( Wx^{\left( i \right)} \right) ^{y^{\left( i \right)}}\left[ 1-\sigma \left( Wx^{\left( i \right)} \right) \right] ^{\left( 1-y^{\left( i \right)} \right)}}

\\

\text{将求似然最大值转换为求最小值；为求偏导使用对数；}

\\

L_{\text{对数似然}}=-\log \mathscr{L}\left( w \right) 

\\

L_{\text{对数似然}}=-\sum_{i=1}^n{y^{\left( i \right)}\log \left( \sigma \left( Wx^{\left( i \right)} \right) \right)}+\left( 1-y^{\left( i \right)} \right) \log \left( 1-\sigma \left( Wx^{\left( i \right)} \right) \right)

$$

​	**均方误差 损失函数**：【MSE、最小二乘法】

$$

L=\frac{1}{N}\sum_{i=1}^N{\left( y^{\left( i \right)}-f\left( x^{\left( i \right)} \right) \right) ^2}

$$

​	平均绝对值误差 损失函数：【MAE】

$$

L=\frac{1}{N}\sum_{i=1}^N{\left| y^{\left( i \right)}-f\left( x^{\left( i \right)} \right) \right|}

$$

​	平均绝对百分误差 损失函数：【MAPE】

$$

L=\frac{1}{N}\sum_{i=1}^N{\left| \frac{y^{\left( i \right)}-f\left( x^{\left( i \right)} \right)}{\left| y^{\left( i \right)} \right|} \right|}

$$

​	均方对数误差 损失函数：【MSLE】

$$

L=\frac{1}{N}\sum_{i=1}^N{\left( \log \left( y^{\left( i \right)} \right) -\log \left( f\left( x^{\left( i \right)} \right) \right) \right) ^2}

	指数损失函数 

	泊松损失 

	余弦相似度损失函数 CosSim

	余弦对数损失函数 LogCosh

$$

​	指数 损失函数 ：【Adaboost】

$$

L=\exp \left( -y^{\left( i \right)}\cdot f\left( x^{\left( i \right)} \right) \right)

$$

​	泊松 损失函数：【Poisson】

$$

L=\frac{1}{N}\sum_{i=1}^N{\left( f\left( x^{\left( i \right)} \right) -y^{\left( i \right)}\cdot \log \left( f\left( x^{\left( i \right)} \right) +\xi \right) \right)}

$$

​	余弦相似度：【CosSim】

$$

\text{向量点积公式：}a\cdot b=\lVert a \rVert \lVert b \rVert \cos \theta 

\\

\text{设}a,b\text{为}n\text{为向量}a\left( A_1,A_2,...,A_n \right) ,b\left( B_1,B_2,...,B_n \right) \text{，则}

\\

CosSim\left( a,b \right) =\frac{a\cdot b}{\lVert a \rVert \lVert b \rVert}=\frac{\sum_{i=1}^n{A_i\times B_i}}{\sqrt{\sum_{i=1}^n{A_{i}^{2}}}\times \sqrt{\sum_{i=1}^n{B_{i}^{2}}}}

$$

​	余弦对数 损失函数：【LogCosh】

$$

\text{记双曲余弦函数：}\cosh \left( x \right) =\frac{e^x+e^{-x}}{2}

\\

L=\sum_{i=1}^n{\log \left( \cosh \left( f\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) \right)}

$$

​	Hinge 损失函数：【合页损失函数、SVM】

$$

L=\frac{1}{N}\sum_{i=1}^N{\left( \max \left\{ 1-y^{\left( i \right)}f\left( x^{\left( i \right)} \right) ,0 \right\} \right)}

$$

​	Huber 损失函数：



​		相比于平方损失来说Huber损失对于异常值不敏感，可以使用超参数δ来调节这一误差的阈值。

​		当δ趋向于无穷时则退化为了MSE，当δ趋向于0时它就退化成了MAE

​		是一个连续可微的分段函数，表达式如下：

$$

L=\begin{cases}

	\frac{1}{2}\left( y^{\left( i \right)}-f\left( x^{\left( i \right)} \right) \right) ^2\,\,  for\,\,\left| y^{\left( i \right)}-f\left( x^{\left( i \right)} \right) \right|\leqslant \delta ,\\

	\delta \cdot \left| y^{\left( i \right)}-f\left( x^{\left( i \right)} \right) \right|-\frac{1}{2}\delta ^2\,\,  otherwise.\\

\end{cases}

$$

​	**熵**：

$$

\text{信息量：定义一个事件}x\text{的信息量 }I\left( x \right) =-\log p\left( x \right) 

\\

\text{信息熵：信息量的期望，描述系统内部的混乱程度 ，定义如下}

\\

H\left( x \right) =E\left[ I\left( x \right) \right] =-\sum_{x\in X}{p\left( x \right) \log p\left( x \right)}

\\

\text{相对熵：}KL\text{散度，描述两个随机分布的距离，定义如下}

\\

D_{KL}\left( p||q \right) =E_p\left[ \log \frac{p\left( x \right)}{q\left( x \right)} \right] =\sum_{x\in X}{\left[ p\left( x \right) \log p\left( x \right) -p\left( x \right) \log q\left( x \right) \right]}

\\

\text{设}p\left( x \right) ,q\left( x \right) \text{分别为真实分布概率和估计分布概率}

\\

\text{交叉熵：有两个分布}p\left( x \right) ,q\left( x \right) \text{，定义}CrossEntropy\text{如下}

\\

H\left( p,q \right) =E_p\left[ -\log q \right] =-\sum_{x\in X}{p\left( x \right) \log q\left( x \right)}

\\

H\left( p,q \right) =H\left( p \right) +D_{KL}\left( p||q \right) 

\\

$$

​	**交叉熵 损失函数**：

​		交叉熵描述两个随机分布间的距离，使得预测值与真实值的概率分布距离最小化

$$

L=-\frac{1}{N}\sum_{i=1}^N{\left[ y^{\left( i \right)}\log f\left( x^{\left( i \right)} \right) +\left( 1-y^{\left( i \right)} \right) \log \left( 1-f\left( x^{\left( i \right)} \right) \right) \right]}

	Smooth L1损失函数：

$$



​	Smooth L1损失函数：

$$

Faster\,\,RCNN\text{中}RPN\text{回归框的损失计算表达式如下：}

\\

L\left( \left\{ p_i \right\} ,\left\{ u_i \right\} \right) =\frac{1}{N_{cls}}\sum_i{L_{cls}\left( p_i,p_{i}^{*} \right)}+\lambda \frac{1}{N_{reg}}\sum_i{p_{i}^{*}L_{reg}\left( t_i,t_{i}^{*} \right)}

\\

p_i\text{为}anchor\text{预测为目标的概率,}p_{i}^{*}=\begin{cases}

	\text{0 }negtive\,\,label\\

	\text{1 }positive\,\,label\\

\end{cases}

\\

t_i=\left\{ t_x,t_y,t_w,t_h \right\} \text{表示预测的}bounding\,\,box\text{坐标}

\\

t_{i}^{*}\text{是与}positive\,\,anchor\text{对应的真实的}bounding\,\,box\text{坐标}

\\

\text{分类的对数损失：}L_{cls}\left( p,p^* \right) =-\log \left[ p^*p+\left( 1-p^* \right) \left( 1-p \right) \right] 

\\

\text{回归的}smooth-l\text{1损失：}L_{reg}\left( t,t^* \right) =R\left( t-t^* \right) 

\\

R\text{则是}smooth\,\,L\text{1函数，设}x=t-t^*,\text{则}

\\

smooth_{L1}\left( x \right) =\begin{cases}

	0.5x^2\,\,  ,if\,\,\left| x \right|<1\\

	\left| x \right|-\text{0.5   ,}otherwise\\

\end{cases}

$$







#### 凸优化方法：



​	梯度下降法：全量-，批量-，随机-，动量-，加速-

​	牛顿法， 拟牛顿法， 高斯牛顿法， LM法（列文伯格法）

​	共轭梯度法， Adagrad-like， Adadelta， RMSprop

​	Adam：Momentum + Adaprop

​	Adamax， Nadam， NadaMax



​	启发式：模拟退火方法、遗传算法、蚁群算法，粒子群算法

​	拉格朗日乘数法



#### 欠/过拟合解决方案总结：



​		---拟合的模型（分类/回归超平面）过于简单/复杂

​	数据增强

​	正则化方法总结【模型的复杂性惩罚项】:L1正则化/L2正则化

​	Dropout：临时随机丢弃当前层的部分神经元&连接

​	Drop Connect：将所有权重的一个随机子集设置为零

​	早停法：确定迭代次数

​		迭代次数太少-易欠拟合[方差较小，偏差较大]；

​		而迭代次数太多-易过拟合[方差较大，偏差较小]；



#### 降维方法总结：



![1558255670832](C:\Users\Tinkle_GW\AppData\Roaming\Typora\typora-user-images\1558255670832.png)



​	主成分分析 (PCA)/主成分回归 (PCR)

​	独立成分析 (ICA)

​	偏最小二乘回归（Partial Least Squares Regression (PLSR)）

​	Sammon 映射（Sammon Mapping）

​	多维尺度变换（Multidimensional Scaling (MDS)）

​	投影寻踪（Projection Pursuit）



​	局部线性嵌入(Locally Linear Embedding (LLE))

​	SNE、**t-SNE**、MDS、ISOMAP（等距特征映射）

​	**自编码器**



​	线性判别分析（Linear Discriminant Analysis (LDA)）

​	混合判别分析（Mixture Discriminant Analysis (MDA)）

​	二次判别分析（Quadratic Discriminant Analysis (QDA)）

​	灵活判别分析（Flexible Discriminant Analysis (FDA)）



#### 最优化理论：



​	min_f(x)/max_f(x)	s.t.hx=0, gx<0

​	·x为R内的n维向量，属于决策变量或问题的解

​	·s.t. subject to 意为受限于、满足条件

​	·fx称为 目标/代价函数，hx为等式约束，gx为不等式约束

​	argmin_fx/argmaxfx	表示无约束的最优化数学模型，求解当函数fx达到最值时，x的取值



​	线性规划：

​		当fx，hx，gx都是 线性函数时

​	非线性规划：

​		当fx，hx，gx任意一个是非线性函数时

​	二次规划：

​		fx为二次函数，hx，gx均为线性函数时



​	多目标规划：

​		fx是向量函数时

​	凸集：

​		实数域R的向量空间中，若某一集合S的任意两点的连线 都在S内部，则称集合S为 凸集

​	凸函数：

​		凸函数是凸集中元素的数学特征

​		定义某个向量空间的凸子集C有实值函数f，若C上的任意两点x1，x2，a[0,1]都有

$$

f\left( ax_1+\left( 1-a \right) x_2 \right) \leqslant f\left( ax_1 \right) +f\left( \left( 1-a \right) x_2 \right)

$$

​	凸优化性质：

​		凸优化任一局部的最优解都是它的整体最优解

​		凸优化的任一局部极值也是全局极值



#### 采样：



​	随机采样：按照一定的概率分布随机抽取样本，通常有有限区间均匀分布，高斯分布等 

​	接受-拒绝采样：

​	分权采样（重要性采样）：

​	MCMC - MH算法 采样：

​	MCMC - Gibbs采样：

​	变分推断：



#### 模型评估方法：



​	PR曲线

​	ROC曲线

​	AUC指标



​	假设检验

​	T检验



#### 进化算法：



​	遗传算法

​	粒子群算法

​	蚁群算法

​	模拟退火算法



#### 算法导论：



​	排序算法：

​	匹配算法：

​	递归

​	贪心

​	回溯法

​	分治策略

​	动态规划



------







蒙特卡洛积分：







