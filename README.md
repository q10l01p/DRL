# 强化学习

## 马尔科夫链

### 智能体和环境

马可洛夫链描述的是智能体和环境进行互动的过程。简单说：智能体在一个状态(用 $S$ 代表)下，选择了某个动作(用 $S$ 代表)，进入了另外一个状态，并获得奖励(用 $R$ 代表)的过程。

<div align=center>
<img width="800" src=" https://pic4.zhimg.com/80/v2-3899583b9d750644e978a7f4e7ed6dbf_720w.webp "/>
</div>
<div align=center></div>

在马尔科夫链中，有三个重要的元素：$S$，$A$，$R$。

- S(state): **状态**，在图中用白色圈圈表示。**状态**就是智能体观察到的当前环境的**部分或者全部特征**。
  - **状态空间**就是智能体能够观察到的特征数量。
  - 只有智能体能够观察到的特征才算是状态。
    - 要给智能体最有用的特征。
    - 要注意观察的角度。
- A(action): **动作**，在图中用黑色圈圈表示。**动作**就是智能体做出的**具体行为**。
  - **动作空间**就是该智能体能够做出的动作数量。
- R(reward)奖励: 当我们在某个状态下，完成动作。环境就会给我们反馈，告诉我们这个动作的效果如何。这种效果的数值表达，就是**奖励**。
  - 奖励在强化学习中，起到了很关键的作用，我们会以奖励作为引导，让智能体学习做能获得**最多奖励**的动作。
  - 奖励的设定是**主观**的，也就是说我们为了智能体更好地学习工作，自己定的。

现在我们来总结一下马尔科夫链，其中也包含了强化学习的一般步骤：

1. 智能体在环境中，观察到状态(S)；
2. 状态(S)被输入到智能体，智能体经过计算，选择动作(A);
3. 动作(A)使智能体进入另外一个状态(S)，并返回奖励(R)给智能体;
4. 智能体根据返回，调整自己的策略。 重复以上步骤，一步一步创造马尔科夫链。

### 马可洛夫'链'

在强化学习里，这根本不应该叫做**链**而应该叫马可洛夫**树**！

马尔科夫链之所以是我们现在看到的一条链条。是因为我们站在现在，往**后**看，所以是一条确定的路径。但如果我们往**前**看，就并不是一条路径，而是充满了各种“不确定性”。

假设现在我们来玩这样一个游戏。这个游戏是简化版的大富翁，我们只考虑我们当前所处位置，也就是状态。智能体移动的时候，它可以选择投掷1-3个骰子，根据骰子点数的总和向前移动。

<div align=center>
<img width="800" src=" https://pic3.zhimg.com/80/v2-33c8bf74c9aaa9813fff297ca4fc17ca_720w.webp "/>
</div>
<div align=center></div>

现在，智能体从格子A掷骰子，并移动到格子B。其实经历了两次不确定性。

第一次，是**选择**的过程。智能体主动选择骰子的个数。掷骰子的个数不同，到达格子B的概率也不同。所以**选择会影响到下一个状态**。这种不同动作之间的选择，我们称为智能体的**策略**。策略我们一般用$\pi$表示。我们的任务就是**找到一个策略，能够获得最多的奖励**。

第二次的不确定性，是**环境的随机性**，这是智能体无法控制的。在这个例子里就是骰子的随机性。注意，**并不是所有环境都有随机性**，有些环境是很确定的(例如把以上所有骰子每一面都涂成1点)，但马尔科夫链允许我们有不确定性的存在。

不确定性来自两个方面：

1. 智能体的行动选择（策略）。
2. 环境的不确定性。

<div align=center>
<img width="800" src=" https://pic3.zhimg.com/80/v2-88fdbeb602197457469e269eeb88a3a6_720w.webp "/>
</div>
<div align=center></div>

**不同选择后面的环境随机性给出的概率是不一样的**。虽然我不能控制环境的随机性，但我能控制我的选择，让我避免高风险的低回报的情况出现。

### 总结

1. 马尔科夫链是用来描述智能体和环境互动的过程。
2. 马尔科夫链包含三要素：state，action，reward
   - state：只有智能体能够观察到的才是state。
   - action：智能体的动作。
   - reward：引导智能体工作学习的主观的值。
   - 马尔科夫链的不确定性
   - 核心思想：如果你不希望孩子有某种行为，那么当这种行为出现的时候就进行惩罚。如果你希望孩子坚持某种行为，那么就进行奖励。这也是整个强化学习的核心思想。

## 强化学习中的Q值和V值

马可洛夫告诉我们： 当智能体从一个状态 $S$，选择动作 $A$，会进入另外一个状态 $S^{\prime}$；同时，也会给智能体奖励 $R$。 奖励既有正，也有负。正代表我们鼓励智能体在这个状态下继续这么做；负得话代表我们并不希望智能体这么做。 在强化学习中，我们会用奖励 $R$ 作为智能体学习的引导，期望智能体获得尽可能多的奖励。

但更多的时候，我们并不能单纯通过R来衡量一个动作的好坏。来看下面一个例子：

假设，10天之后进行期末考试，我们今天有两个选择：

1. 放弃吧，我们玩游戏！我们每天可以获得+1心情值；
2. 决心努力一搏，我们开始学习吧！每天我们-2心情值。

从这10天看，我们肯定是选择【1.玩游戏】。因为10天后，我们虽然考试没过，但至少收获10天的快乐。

但事实上，我们再看远一点:

- 因为挂科，接受老师怒吼攻击！心情值马上减5;
- 父母因为我考得好成绩，给了更多的零用钱。心情值加200点。

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-86cf022dcf0a6942109d7751c73fbf15_720w.webp "/>
</div>
<div align=center></div>

所以，假设我们能预知未来，我们一定会选择【2.去复习】

因此，我们必须用长远的眼光来看待问题。我们要把未来的奖励也计算到当前状态下，再进行决策。

### Q和V的意义

所以我们在做决策的时候，需要把眼光放远点，把未来的价值换到当前，才能做出选择。

为了方便，我们希望可以有一种方法衡量我做出每种选择价值。这样，我只要看一下标记，以后的事情我也不用理了，我选择那个动作价值更大，就选那个动作就可以了。

也就是说，我们让复习和游戏都有一个标记，这个标记描述了这个动作的价值:

- 游戏 +500
- 复习 +750

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-47c9fecf94146cb775dfa152ac644ff1_720w.webp "/>
</div>
<div align=center></div>

- Q值:评估**动作**的价值,它代表了智能体**选择这个动作后**，一直到最终状态奖励总和的期望。
- V值:评估**状态**的价值，它代表了智能体**在这个状态下**，一直到最终状态的奖励总和的期望。

价值越高，表示我从**当前状态**到**最终状态**能获得的**平均奖励**将会越高。因为智能体的目标数是获取尽可能多的奖励，所以智能体在当前状态，只需要选择价值高的动作就可以了。

### V值的定义

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-9dfb04172e0505e3aabf3b2c6f5a05a8_720w.webp "/>
</div>
<div align=center></div>

假设现在需要求某状态 $S$ 的 $V$ 值，那么我们可以这样：

1. 我们从 $S$ 点出发，并影分身出若干个自己;
2. 每个分身按照当前的**策略**选择行为;
3. 每个分身一直走到最终状态，并计算一路上获得的所有**奖励总和**;
4. 我们计算每个影分身获得的**平均值**,这个平均值就是我们要求的 $V$ 值。

用大白话总结就是：从某个状态，按照策略，走到最终状态很多很多次；最终获得奖励总和的平均值，就是 $V$ 值。

#### 敲黑板

1. 从V值的计算，我们可以知道，V值代表了这个状态的今后能获得奖励的期望。从这个状态出发，到达最终状态，平均而言能拿到多少奖励。所以我们轻易比较两个状态的价值。
2. V值跟我们选择的策略有很大的关系。我们看这样一个简化的例子，从S出发，只有两种选择，A1，A2；从A1，A2只有一条路径到最终状态，获得总奖励分别为10和20。

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-b47cb3cc1020366668978452be6af441_720w.webp "/>
</div>
<div align=center></div>

现在我们假设策略采用平均策略 $[A1:50\%,A2:50\%]$ ，根据用影分身(如果是学霸直接求期望)，那么我们可以求得V值为15

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-fdcc97a3472f013c4bcc571699e9f860_720w.webp "/>
</div>
<div align=center></div>

现在我们改变策略 $[A1:60\%,A2:40\%]$，那么我们可以求得V值为 $14$，变少了！

<div align=center>
<img width="800" src=" https://pic4.zhimg.com/80/v2-79294603d7a20fad70de57436b7693c3_720w.webp "/>
</div>
<div align=center></div>

**V值是会根据不同的策略有所变化的！**

### Q值的定义

Q值和V值的概念是一致的，都是衡量在马可洛夫树上某一个节点的价值。只不过V值衡量的是**状态节点**的价值，而Q值衡量的是**动作节点**的价值。

<div align=center>
<img width="800" src=" https://pic3.zhimg.com/80/v2-54c4e174d5e7c9ff989c8cba44872bca_720w.webp "/>
</div>
<div align=center></div>

现在我们需要计算，某个状态S0下的一个动作A的Q值：

1. 我们就可以从A这个节点出发，使用影分身之术；
2. 每个影分身走到最终状态,并记录所获得的奖励；
3. 求取所有影分身获得奖励的平均值，这个平均值就是我们需要求的Q值。

用大白话总结就是：从某个状态选取动作A，走到最终状态很多很多次；最终获得奖励总和的平均值，就是Q值。

#### 敲黑板

与V值不同，Q值**和策略并没有直接相关**，而**与环境的状态转移概率相关**，而环境的状态转移概率是不变的。

### V值和Q值关系

1. 都是马可洛夫树上的节点
2. 价值评价的方式是一样的：
   - 从当前节点出发
   - 一直走到最终节点
   - 所有的奖励的期望值
3. 其实Q和V之间是可以相互换算的。

### 从Q到V

我们先来看看，怎样用Q值算V值。

<div align=center>
<img width="800" src=" https://pic2.zhimg.com/80/v2-a2495a7ddab8939f45fd08a0e8094e41_720w.webp "/>
</div>
<div align=center></div>

从定义出发，我们要求的 $V$ 值，就是从状态 $S$ 出发，到最终获取的所获得的奖励总和的期望值。也就是蓝色框部分。

$S$ 状态下有若干个动作，每个动作的 $Q$ 值，就是从这个动作之后所获得的奖励总和的期望值。也就是红色框部分。

假设我们已经计算出每个动作的 $Q$ 值，那么在计算 $V$ 值的时候就不需要一直走到最终状态了，只需要走到动作节点，看一下每个动作节点的 $Q$ 值，根据策略，计算 $Q$ 的期望就是 $V$ 值了。

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-3e3dbd399d5a7cee7228f4de1ea0cdf8_720w.webp "/>
</div>
<div align=center></div>

更正式的公式如下：

<div align=center>
<img width="800" src=" https://pic4.zhimg.com/80/v2-77d655b5c43fa96ec4795d4bb67afaab_720w.webp "/>
</div>
<div align=center></div>

### 从V到Q

现在我们换个角度，看一下怎样从V换算成Q值。

道理还是一样，就是用Q就是V的期望！而且这里不需要关注策略，这里是环境的状态转移概率决定的。

<div align=center>
<img width="800" src=" https://pic4.zhimg.com/80/v2-0b20f36e55bef7d886ebd6f35a9d4d0b_720w.webp "/>
</div>
<div align=center></div>

对，但还差点东西。当我们选择A，并转移到新的状态时，就能获得奖励，我们必须把这个**奖励**也算上！

<div align=center>
<img width="800" src=" https://pic4.zhimg.com/80/v2-da78a4e6a2f0081c5744c2f9197a2573_720w.webp "/>
</div>
<div align=center></div>

更正式的公式如下：

<div align=center>
<img width="800" src=" https://pic3.zhimg.com/80/v2-4de7b8ec713b2a409362061bb325bff2_720w.webp "/>
</div>
<div align=center></div>

> **折扣率** 在强化学习中，有某些参数是人为**主观**制定。这些参数并不能推导，但在实际应用中却能解决问题，所以我们称这些参数为**超参数**，而折扣率就是一个超参数。 与金融产品说的贴现率是类似的。我们计算Q值，目的就是把未来很多步奖励，折算到当前节点。但未来n步的奖励的10点奖励，与当前的10点奖励是否完全等价呢？未必。所以我们人为地给未来的奖励一定的折扣，例如：0.9,0.8，然后在计算到当前的Q值。

### 从V到V

现在我们知道如何从V到Q，从Q到V了。但实际应用中，我们更多会从V到V。

但其实从V到V也是很简单的。把公式代进去就可以了。

<div align=center>
<img width="800" src=" https://pic1.zhimg.com/80/v2-a8b99b60cd8c0335502b59ce6610b2e0_720w.webp "/>
</div>
<div align=center></div>

### 总结

1. 比起记住公式，其实我们更应该注意Q值和V值的意义：他们就像一个路牌一样，告诉我们从马可洛夫树的一个节点出发，下面所有节点的收获的期望值。也就是假设从这个节点开始，走许多许多次，最终获取的奖励的平均值。
2. V就是子节点的Q的期望！但要注意V值和策略相关。
3. Q就是子节点的V的期望！但要注意，记得把R计算在内。

计算某一个节点的Q值和V值，需要许多次试验，取其中的平均值。但实际上，我们不但需要求一个节点的值，而是求所有节点的值。如果我们每一个节点都用同样的方法，消耗必然会很大。所以人们发明了许多方式去计算Q值和V值，基于价值计算的算法就是围绕Q和V展开的。
