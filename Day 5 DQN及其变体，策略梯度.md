### Day 5 

#### DQN

DQN其实就是对v函数和q函数的一个近似，由参数w去进行描述：

![image-20211201105132883](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201105132883.png)

![image-20211201105141194](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201105141194.png)

我们根据对神经网络不一样的输出就能得到不同的结果对于状态价值函数，神经网络的输入是状态s的特征向量，输出是状态价值v^(s,w)v^(s,w)。对于动作价值函数，有两种方法，一种是输入状态s的特征向量和动作a，输出对应的动作价值q^(s,a,w)q^(s,a,w)，另一种是只输入状态s的特征向量，动作集合有多少个动作就有多少个输出q^(s,ai,w)q^(s,ai,w)。这里隐含了我们的动作是有限个的离散动作。

![img](https://img2018.cnblogs.com/blog/1042406/201809/1042406-20180928142605652-445522913.jpg)



经验回放:设立一个经验池，每次和环境交互得到的奖励与状态更新情况都保存起来，用于后面目标Q值的更新。

#### Nature DQN

在DQN中目标Q值的计算使用到了当前要训练的Q网络参数来计算Q(ϕ(Sj′),Aj′,w)，而实际上，我们又希望通过yj来后续更新Q网络参数。这样两者循环依赖，迭代起来两者的相关性就太强了。不利于算法的收敛。

Nature DQN使用了两个Q网络，一个当前Q网络用来选择动作，更新模型参数，另一个目标网络Q′用于计算目标Q值。目标Q网络的网络参数不需要迭代更新，而是每隔一段时间从当前Q网络复制过来，即延时更新，这样可以减少目标Q值和当前的Q值相关性。

Nature DQN和上一篇的DQN相比，除了用一个新的相同结构的目标Q网络来计算目标Q值以外，其余部分基本是完全相同的。

两个网络yj的定义：

![image-20211201114020675](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201114020675.png)

![image-20211201114030610](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201114030610.png)

![image-20211201111708953](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201111708953.png)

![image-20211201095926460](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201095926460.png)

#### DDQN

在DDQN之前，目标Q都是通过贪婪法直接得到的，使用max虽然可以快速让Q值向可能的优化目标靠拢，但是很容易过犹不及，导致过度估计(Over Estimation)，所谓过度估计就是最终我们得到的算法模型有很大的偏差(bias)。

在Nature DQN中，计算目标Q值：

![image-20211201115206528](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201115206528.png)

在DDQN这里，不再是直接在目标Q网络里面找各个动作中最大Q值，而是先在当前Q网络中先找出下一状态最大Q值对应的动作，即

![image-20211201115222329](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201115222329.png)

然后利用这个选择出来的动作amax(Sj′,w)在目标网络里面去计算目标Q值。即：

![image-20211201115236201](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201115236201.png)

　综合起来写就是：

![image-20211201115252240](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201115252240.png)

除了目标Q值的计算方式以外，DDQN算法和Nature DQN的算法流程完全相同。

![image-20211201115401895](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201115401895.png)

#### **Prioritized Replay DQN **

Prioritized Replay DQN是通过优化经验池来优化算法，根据每个样本的TD误差绝对值|δ(t)|，给定该样本的优先级正比于|δ(t)|，将这个优先级的值存入经验回放池。

由于引入了经验回放的优先级，那么Prioritized Replay DQN的经验回放池和之前的其他DQN算法的经验回放池就不一样了。因为这个优先级大小会影响它被采样的概率。在实际使用中，我们通常使用SumTree这样的二叉树结构来做我们的带优先级的经验回放池样本的存储。具体的SumTree树结构如下图：

![image-20211201135238006](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201135238006.png)

所有的经验回放样本只保存在最下面的叶子节点上面，一个节点一个样本。内部节点不保存样本数据。而叶子节点除了保存数据以外，还要保存该样本的优先级，就是图中的显示的数字。对于内部节点每个节点只保存自己的儿子节点的优先级值之和，如图中内部节点上显示的数字。

除了经验回放池，现在我们的Q网络的算法损失函数也有优化，之前我们的损失函数是：

![image-20211201135903692](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201135903692.png)

现在我们新的考虑了样本优先级的损失函数是

![image-20211201135921692](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201135921692.png)

其中wj是第j个样本的优先级权重，由TD误差|δ(t)|归一化得到。

#### Dueling DQN

Dueling DQN是通过优化神经网络部分来优化算法，它将Q网络分成两部分，第一部分是仅仅与状态S有关，与具体要采用的动作A无关，这部分我们叫做价值函数部分，记做V(S,w,α),第二部分同时与状态状态S和动作A有关，这部分叫做优势函数(Advantage Function)部分,记为A(S,A,w,β),那么最终我们的价值函数可以重新表示为：

![image-20211201140347636](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201140347636.png)

其中，w是公共部分的网络参数，而α是价值函数独有部分的网络参数，而β是优势函数独有部分的网络参数。

![image-20211201140516334](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201140516334.png)

我们可以直接使用上一节的价值函数的组合公式得到我们的动作价值，但是这个式子无法辨识最终输出里面V(S,w,α)和A(S,A,w,β)各自的作用，举个例子，把一个常量加到V中，并且从A中减去该常量，那么Dueling DQN仍输出相同的Q值。为了可以体现这种可辨识性(identifiability),实际使用的组合公式如下：

![image-20211201141054831](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201141054831.png)

#### 策略梯度

value-based无法解决随机策略问题。Value Based强化学习方法对应的最优策略通常是确定性策略，因为其是从众多行为价值中选择一个最大价值的行为，而有些问题的最优策略却是随机策略，这种情况下同样是无法通过基于价值的学习来求解的。这时也可以考虑使用Policy Based强化学习方法。

就像石头剪刀布，随便出那个都是最优策略，但通过value-based，每次都会出相同的动作，这样反而容易输/

value-based

![image-20211201142948527](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201142948527.png)

policy-based

![image-20211201143003100](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201143003100.png)



在Policy Based强化学习方法下，我们对策略进行近似表示。此时策略π可以被被描述为一个包含参数θ的函数,即：

![image-20211201150254897](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201150254897.png)

我们想要获得最优策略，就是最大化reward，所以是一个梯度上升的问题 。而想要梯度上升，我们就要找到优化的函数目标：

![image-20211201150537877](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201150537877.png)

最终对θ求导的梯度都可以表示为：

![image-20211201150609465](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201150609465.png)

此时我们的梯度式子里面的∇θlogπθ(s,a)部分并不改变，变化的只是后面的Qπ(s,a)]部分。对于∇θlogπθ(s,a),我们一般称为分值函数（score function）。

最常用的策略函数就是softmax策略函数了，它主要应用于离散空间中，softmax策略使用描述状态和行为的特征ϕ(s,a)与参数θ的线性组合来权衡一个行为发生的几率,即:

![image-20211201150909743](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201150909743.png)

通过求导很容易求出对应的分值函数为：

![image-20211201150937431](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201150937431.png)

#### 蒙特卡罗策略梯度reinforce算法

![image-20211201153330780](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201153330780.png)

优化函数：

![image-20211201153415736](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201153415736.png)

![image-20211201153714778](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201153714778.png)



对于蒙特卡洛来说：

![image-20211201153950720](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201153950720.png)

Gt就是它的预期期望，是可知的，从后往前推 

![image-20211201154138967](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201154138967.png)

![image-20211201154202061](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201154202061.png)

![image-20211201154301977](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201154301977.png)



对于这个Loss来说，Gt就是一个权重参数，因为实际at不一定是最优的，它通过Gt来定义它的权重，Gt越大，说明这个at的价值越大，就越接近最优动作。

![image-20211201151031965](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211201151031965.png)

#### Actor-Critic



