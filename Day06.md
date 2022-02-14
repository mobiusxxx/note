# Day06

### Actor-Critic

Actor使用我们上一节讲到的策略函数,负责生成动作(Action)并和环境交互。而Critic使用我们之前讲到了的价值函数，负责评估Actor的表现，并指导Actor下一阶段的动作。

在蒙特卡罗策略梯度reinforce算法中，我们将蒙特卡罗方法来计算每一步的价值，代替了Critic，现在我们使用类似DQN中用的价值函数来替代蒙特卡罗法，作为一个比较通用的Critic。

也就是说在Actor-Critic算法中，我们需要做两组近似，第一组是策略函数的近似：

![image-20211202114335091](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202114335091.png)

第二组是价值函数的近似，对于状态价值和动作价值函数分别是：

![image-20211202114352452](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202114352452.png)

在蒙特卡罗策略梯度reinforce算法中，我们的策略的参数更新公式是：

![image-20211202114448987](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202114448987.png)

分值函数部分不用动，要变成Actor的话改动的是vtvt，这块不能再使用蒙特卡罗法来得到，而应该从Critic得到，也就是用我们之前的Q网络做Critic。

**汇总来说，就是Critic通过Q网络计算状态的最优价值vt, 而Actor利用vt这个最优价值迭代更新策略函数的参数θ,进而选择动作，并得到反馈和新的状态，Critic使用反馈和新的状态更新Q网络参数w, 在后面Critic会使用新的网络参数w来帮Actor计算状态的最优价值vt。**

![image-20211202115524281](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202115524281.png)

### A3C

Actor-Critic算法的缺点就在于无论怎样调参，都很难收敛。为了解决收敛问题，之前的DQN算法，使用了经验回放的技巧，A3C的思路也是如此，它利用多线程的方法，同时在多个线程里面分别和环境进行交互学习，每个线程都把学习的成果汇总起来，整理保存在一个公共的地方。并且，定期从公共的地方把大家的齐心学习的成果拿回来，指导自己和环境后面的学习交互。

通过这种方法，A3C避免了经验回放相关性过强的问题，同时做到了异步并发的学习模型。

![image-20211202135106857](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202135106857.png)

Global Network就是上一节说的共享的公共部分，主要是一个公共的神经网络模型，这个神经网络包括Actor网络和Critic网络两部分的功能。下面有n个worker线程，每个线程里有和公共的神经网络一样的网络结构，每个线程会独立的和环境进行交互得到经验数据，这些线程之间互不干扰，独立运行。

**每个线程和环境交互到一定量的数据后，就计算在自己线程里的神经网络损失函数的梯度，但是这些梯度却并不更新自己线程里的神经网络，而是去更新公共的神经网络。**也就是n个线程会独立的使用累积的梯度分别更新公共部分的神经网络模型参数。每隔一段时间，线程会将自己的神经网络的参数更新为公共神经网络的参数，进而指导后面的环境交互。

与Actor-Critic的网络结构不同，在A3C这里，我们把两个网络放到了一起，即输入状态S,可以输出状态价值V,和对应的策略π。

在Actor-Critic中使用优势函数做评估点时：

![image-20211202145901598](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202145901598.png)

Q(S,A)的值一般可以通过单步采样近似估计，即：

![image-20211202145923443](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202145923443.png)

在A3C中，采样更进一步，使用了N步采样，以加速收敛。这样A3C中使用的优势函数表达为：

![image-20211202145950555](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202145950555.png)

损失函数部分，和Actor-Critic基本相同。有一个小的优化点就是在Actor-Critic策略函数的损失函数中，加入了策略π的熵项,系数为c, 即策略参数的梯度更新和Actor-Critic相比变成了这样：

![image-20211202150330155](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202150330155.png)

#### DDPG

用于解决连续性问题，直接输出的是确定行策略而不是随机性策略。

![image-20211202152910695](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202152910695.png)

![image-20211202153121346](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20211202153121346.png)

目标网络只是用在求target的过程中。如果不是求target用的，就不用目标网络。

DQN采用的是epsilon-greedy的算法去选取a，而DDPG用了正态分布抽样方式。我们用输出的a作为一个正态分布的平均值，加上参数VAR，构造一个正态分布。然后从正态分布中随机出一个新的动作代替a。我们知道a作为正态分布的均值，也是一个有最大概率获得的一个值。这就有点像epsilon-greedy，有一定概率在探索，也有一定概率在开发新的动作。**但这样还是会有过度估计的问题**。

### 