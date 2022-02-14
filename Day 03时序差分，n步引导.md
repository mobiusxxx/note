Day 03

###### MC

on-policy off-policy

On-policy 的目标策略和行为策略是同一个策略，其好处就是简单粗暴，直接利用数据就可以优化其策略，但这样的处理会导致策略其实是在学习一个局部最优，因为On-policy的策略没办法很好的同时保持即探索又利用；

Off-policy将目标策略和行为策略分开，可以在保持探索的同时，更能求到全局最优值。行为策略是用来与环境互动产生数据的策略，即在训练过程中做决策；而目标策略在行为策略产生的数据中不断学习、优化，即学习训练完毕后拿去应用的策略。

![image-20211129153206562](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129153206562.png)

![image-20211129153224072](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129153224072.png)

off-policy的重要性采样

这种技巧可以解决：求解一个概率分布（Distribution）的期望值（Expect）时，用来求解该期望值的样本数据是由另一个概率分布所产生。具体做法是：根据目标策略 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 和行为策略 ![[公式]](https://www.zhihu.com/equation?tex=b) 分别所产生的相同某段序列（在本文Episode中某一段称为Trajectory）的概率的比值来加权求和return（Return是MC法中的一个样本序列（整个Episode）的总奖励），这个比值称为**importance-sampling ratio**。（也就是把一段又一段的序列总价值根据**importance-sampling ratio**加权求和，得到某个state的价值期望)

![image-20211129153539737](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129153539737.png)

![image-20211129154032718](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129154032718.png)

这种简单方式平均的方法叫做原始重要性采样。另一种方法叫做加权重要性采样，它使用了加权平局：

![image-20211129154946292](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129154946292.png)

两种重要性采样的不同可以用偏差和方差来表示。 原始重要性采样的估计是无偏的，而加权重要性采样是有偏的（偏差会渐进地趋于零）。 另一方面，原始重要性采样的方差一般是无界的，因为它的重要性采样率的方差是无界的；而加权重要性采样的任意单个回报的最大权重是1。

蒙特卡洛策略估计

这个算法主要用在离策略的情况，使用加权重要性采样，但是也能用于在策略的情况。 用于在策略时，让目标策略和行为策略一样即可（这种情况下（π=b），W 始终是1）。 近似值 Q 收敛到 qπ （对所有的出现的状态-动作对），而动作由另一个潜在的不同策略 b 提供。

![image-20211129155556367](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129155556367.png)

off-policy 蒙特卡洛控制

在策略的显著特点是，它在估计策略值的同时也用于控制。而离策略方法中，这两个功能是分开的。 用于产生行为的策略，即称作 *行为* 策略，事实上与要评估和提升的策略，即 *目标* 策略，是无关的。 这样分开的好处是，目标策略可以是确定性的（即，贪心的），同时行为策略能持续访问所有的动作。

![image-20211129155904847](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129155904847.png)

总结

这一章的蒙特卡洛方法以 *样本回合* 的方式，从经验中学习价值函数和最优策略。 相比于动态规划（DP）的方法，这至少有三种优势。 首先，它们能够直接从与环境的交互中学习到最优的行为，并不需要知道环境的动态。 其次，它们能够被用于模拟或 *样本模型*。对于相当多的应用来讲，虽然我们很难建立具体的转移概率的模型 （这个转移概率模型是DP方法所需要的），但是，我们可以很容易去估计样本回合。 第三，使用蒙特卡洛方法，我们可以很容易且很有效率地 *聚焦* 到状态的小子集。 对于我们特别感兴趣的区域，可以准确地评估，而不需要费大力气去准确地评估剩余的状态集。

*离策略预测* 指从一个不同的 *行为策略* 产生的数据中学习一个 *目标策略* ，学习这个目标策略的价值函数。 这样的学习方法是基于 *重要性采样* 的，即用两种策略下执行观察到的动作的可能性的比值，来加权回报。 *原始重要性采样* 使用加权回报的简单平均，而 *加权重要性采样* 是使用加权的平均。 原始重要性采样是无偏估计，但是有很大的，可能无限的方差。 而加权重要性采样的方差是有限的，在实际中也更受喜爱。

###### 时序差分学习

TD和MC对于V的估计所采用的方式不同。在MC中一直等到访问后的回报已知，然后使用该回报作为 V(St) 的目标。必须等到回合结束才能确定 V(St) 的增量（这时只有 Gt 已知）。

![image-20211129165131017](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129165131017.png)

而对于TD来说只需要等到下一个时间步。 在时间 t+1，它们立即形成目标并 使用观察到的奖励 Rt+1 进行有用的更新。在实际中，蒙特卡洛更新的目标是 Gt，而TD更新的目标是 Rt+1+γV(St+1)。

![image-20211129165301913](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129165301913.png)

这种TD方法称为 *TD(0)* 或 一步TD

![image-20211129165625759](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129165625759.png)

TD误差：衡量 St 的估计值与更好的估计 Rt+1+γV(St+1) 之间的差异。

![image-20211129170513795](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129170513795.png)

每次TD误差都是 *当时估算* 的误差。因为TD误差取决于下一个状态和下一个奖励，所以直到一个步骤之后才可用。 也就是说，δt 是 V(St+1) 中的误差，在时间 t+1 可用。

Sarsa： on-policy TD控制

1.Sarsa是on-policy的更新方式，它的行动策略和目标策略都是ε-greedy策略。

2.Sarsa是先做出动作后更新。

![image-20211129172318707](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129172318707.png)

Q-learning：off-policy TD控制

Q-Learning的目的是学习特定State下、特定Action的价值。是建立一个Q-Table，以State为行、Action为列，通过每个动作带来的奖赏更新Q-Table。

Q-Learning是off-policy的。异策略是指行动策略和目标策略不是一个策略。Q-Learning中行动策略是ε-greedy策略，要更新Q表的策略是贪婪策略。

![image-20211129172727234](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129172727234.png)

最大化偏差和双学习

在之前的所有控制算法中，都涉及最大化目标策略，在这些算法中，最大估计值被隐含地用作最大值的估计值，这可能导致显着的正偏差。 要了解原因，请考虑单个状态 s，其中有许多动作 a 的真值 q(s,a) 都是零， 但其估计值 Q(s,a) 是不确定的，因此分布在零的上方和下方。 真值的最大值为零，但估计的最大值为正，一个正偏差。我们称之为 *最大化偏差*。

Double Q-learning将使用两个函数 ![[公式]](https://www.zhihu.com/equation?tex=Q%5E%7BA%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=Q%5E%7BB%7D) （对应两个估计器），并且每个 ![[公式]](https://www.zhihu.com/equation?tex=Q) 函数都会使用另一个 ![[公式]](https://www.zhihu.com/equation?tex=Q) 函数的值更新下一个状态。两个 ![[公式]](https://www.zhihu.com/equation?tex=Q) 函数都必须从不同的经验集中学习，这一点很重要，但是要选择要执行的动作可以同时使用两个值函数。 因此，该算法的数据效率不低于Q学习。 在实验中作者为每个动作计算了两个Q值的平均值，然后对所得的平均Q值进行了贪婪探索。

![image-20211129173854136](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129173854136.png)

总结

时序差分算法是一种无模型的强化学习算法。它继承了动态规划(Dynamic Programming)和蒙特卡罗方法(Monte Carlo Methods)的优点，从而对状态值(state value)和策略(optimal policy)进行预测。从本质上来说，时序差分算法和动态规划一样，是一种bootstrapping的算法。同时，也和蒙特卡罗方法一样，是一种无模型的强化学习算法，其原理也是基于了试验。虽然，时序差分算法拥有动态规划和蒙特卡罗方法的一部分特点，但它们也有不同之处。

![image-20211129175424669](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129175424669.png)

n步引导法

n步TD预测

![image-20211129180826634](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129180826634.png)

n步TD就是1步TD方法的一个延申，在一步更新中，目标是第一个奖励加上下一个状态的折扣估计值，我们称之为 *一步回报*：

![image-20211129181052196](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129181052196.png)

任意n步更新的目标是n步回报：

![image-20211129181120798](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129181120798.png)

前n−1个时刻都是将实际的即时奖励作为n步收益的真实值，而最后时刻对应的奖励则是一个估计值。
例如在预测阶段，当前时刻t 的状态价值V ( S t ) 的预测值是从当前时刻t 起一直到t + n − 1执行相应动作获得的折扣即时奖励与第t + n时刻的状态价值估计值的和。

![image-20211129180917764](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129180917764.png)

n步sarsa

主要思想是简单地切换动作状态（状态-动作对），然后使用 ε -贪婪策略。我们根据估计的动作值重新定义n步回报（更新目标）：

![image-20211129182516404](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129182516404.png)

如果 t+n≥T，则 Gt:t+n≐Gt。那么自然算法就是

![image-20211129182600395](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129182600395.png)

![image-20211129181827303](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129181827303.png)

![image-20211129182629833](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129182629833.png)

![image-20211129182648624](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129182648624.png)

n步离策略学习

对于离轨策略下，采用基于**重要度采样**的方法，通过观察行动策略的经验和收益来预测目标策略，重要度采样比是：

![image-20211129183307406](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129183307406.png)

基于重要度采样的离轨策略的策略预测公式为：

![image-20211129183345894](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129183345894.png)

基于重要度采样的离轨策略的策略控制公式为：

![image-20211129183404803](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129183404803.png)

事实上，其与之前的同轨策略不同地方就是多乘一个重要度采样比。

![image-20211129183550431](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129183550431.png)

![image-20211129183607745](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211129183607745.png)

