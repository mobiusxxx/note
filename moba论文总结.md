moba论文总结

##### 1.Mastering Complex Control in MOBA Games with Deep Reinforcement Learning

![image-20220124114432822](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220124114432822.png)

1.目标选择注意力机制

将lstm的输出和游戏单元的堆栈结合，给出一个注意力分布，这样目标选择就可以选择最大注意力的单元。

注意力分布：

![image-20220124115632747](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220124115632747.png)

2.学习技能组合的lstm

3.动作解耦

4.action mask

5.Dual-clip PPO

##### 2.Supervised Learning Achieves Human-Level Performance in MOBA Games: A Case Study of Honor of Kings

![image-20220124143720690](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220124143720690.png)

1.多视图意图标签（全局意图和局部意图）

意图标签表示再宏观策略方向该做什么。全局意图是对于整张地图来说，我们要去哪儿，而局部意图是在local map中我们该往那儿走。

2.游戏状态

论文中的游戏状态由团队杀戮差异、金牌差异、游戏时间、炮塔差异等组成，而不是各队分开的一个数据，而且还有个历史坐标的状态，感觉加上这个状态能更好的在时间上进行一个预测。

3.损失函数

![image-20220124154810674](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220124154810674.png)

前两个是动作标签的交叉熵损失，后两个是意图标签的。

##### 3.Dota2

![image-20220125105441230](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220125105441230.png)

1.将奖励设置零和的，敌方获得奖励，我方就要扣除相应奖励。

2.reward计算中自身产生的reward部分和团队产生reward部分比例的变量

​       论文中把奖励分为了个人奖励和团队奖励。

![image-20220125115729848](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220125115729848.png)

3.手术操作

手术操作就是为了让我们在改变网络，动作空间，状态空间等信息的时候，不用重新训练。其实就是类似迁移学习的一个操作，不改变之前我们已经训练好的权重。

![image-20220125114927782](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220125114927782.png)

4.动作空间中的Delay

Delay: 延迟，0-3的整数，用于控制对应动作的生效时间，0对应当前帧，3对应当前这个timestep的最后一帧。

5.长时间的信用分配（不理解）

定义一个变量 $H$​用于表示reward衰减所影响的时间范围：

![image-20220125120441518](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220125120441518.png)

centralized critic + decentralized policy  https://zhuanlan.zhihu.com/p/107357653

![image-20220125161329657](C:\Users\longyuan\AppData\Roaming\Typora\typora-user-images\image-20220125161329657.png)

