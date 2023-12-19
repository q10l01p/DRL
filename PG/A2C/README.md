# A2C

同步优势 `Actor-Critic`（Synchronous Advantage Actor-Critic，A2C）(Mnih et al., 2016) 和 `Actor-Critic` 算法非常相似，只是在 `Actor-Critic` 算法的基础上增加了并行计算的设计。

如图所示，全局行动者和全局批判者在 `Master` 节点维护。每个 `Worker` 节点的增强学习智能体通过协调器和全局行动者、全局批判者对话。在这个设计中，协调器负责收集各个 `Worker` 节点上与环境交互的经验(Experience)，然后根据收集到的轨迹执行一步更新。更新之后，全局行动者被同步到各个 `Worker` 上继续和环境交互。

在 `Master` 节点上，全局行动者和全局批判者的学习方法和 `Actor-Critic` 算法中行动者和批判者的学习方法一致，都是使用 `TD` 平方误差作为批判者的损失函数，以及 `TD` 误差的策略梯度来更新行动者的。

<div align=center>
<img width="800" src="./PNG/A2C 基本框架.png"/>
</div>
<div align=center>图1 A2C 基本框架</div>

在这种设计下，`Worker` 节点只负责和环境交互。所有的计算和更新都发生在 `Master` 节点。

实际应用中，如果希望降低 `Master` 节点的计算负担，一些计算也可以转交给 `Worker` 节点<sup><a href="#ref1">1</a></sup>，比如说，每个 `Worker` 节点保存了当前全局批判者（Critic）。收集了一个轨迹之后，Worker 节点直接在本地计算给出全局行动者（Actor）和全局批判者的梯度。这些梯度信息继而被传送回 `Master` 节点。最后，协调器负责收集和汇总从各个 `Worker` 节点收集到的梯度信息，并更新全局模型。同样地，更新后的全局行动者和全局批判者被同步到各个 `Worker` 节点。

<div align=center>
<img width="800" src="./PNG/A2C 算法.png"/>
</div>
<div align=center>图2 A2C 算法</div>

> 1. <p name = "ref1">这经常取决于每个 Worker 节点的计算能力，比如是否有 GPU 计算能力，等等。</p>