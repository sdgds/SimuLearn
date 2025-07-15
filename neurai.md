```mermaid
graph TD
    subgraph "配置管理 (Configuration)"
        direction LR
        Config["Config (Singleton Class)<br/><b>功能:</b> 管理全局仿真参数<br/><b>关键函数:</b> set_platform(), set_simulate_status()<br/><b>输出:</b> 全局配置对象 (如dt, JIT开关)<br/><b>物理意义:</b> 设定仿真环境和基本物理约束"]
    end

    subgraph "网络构建 (Network Construction)"
        direction LR
        subgraph "神经元模型 (Neuron Models)"
            direction TB
            BaseNeuron["Neuron (Base Class)<br/><b>功能:</b> 定义神经元通用接口<br/><b>关键参数:</b> size, V_th, V_reset"]
            PointNeuron["点神经元模型<br/>LIF, ExpLIF, Izhikevich, HH<br/><b>功能:</b> 实现单个神经元的动力学方程<br/><b>输入:</b> 突触电流 I_syn, 外部激励 I_e<br/><b>输出:</b> 膜电位 V_m, 脉冲<br/><b>物理意义:</b> 模拟单个神经元的电生理活动"]
            MCNeuron["多室神经元 (MCNeuron)<br/><b>功能:</b> 模拟具有复杂树突形态的神经元<br/><b>输入:</b> 各区室的突触电流<br/><b>输出:</b> 各区室的膜电位, 脉冲<br/><b>物理意义:</b> 更精细地模拟神经元形态和电信号整合"]
            Generators["脉冲发生器<br/>Poisson, Bernoulli, SpikeGenerator<br/><b>功能:</b> 产生外部输入脉冲序列<br/><b>输出:</b> 脉冲序列<br/><b>物理意义:</b> 模拟来自大脑其他区域或感觉器官的输入信号"]
            BaseNeuron --> PointNeuron
            BaseNeuron --> MCNeuron
        end

        subgraph "突触模型 (Synapse Models)"
            direction TB
            BaseSynapse["Synapse (Base Class)<br/><b>功能:</b> 定义突触通用接口<br/><b>关键参数:</b> pre, post, conn, weight"]
            StaticSynapse["静态突触 (StaticSynapse)<br/><b>功能:</b> 传递固定强度的突触后电流<br/><b>输入:</b> 突触前脉冲<br/><b>输出:</b> 突触后电流<br/><b>物理意义:</b> 强度不变的神经连接"]
            STDPSynapse["STDP 学习突触<br/>STDPAll2All, STDPNearest<br/><b>功能:</b> 根据脉冲时序依赖可塑性规则调整突触权重<br/><b>输入:</b> 突触前/后脉冲时间<br/><b>输出:</b> 突触后电流, 更新后的权重<br/><b>物理意义:</b> 模拟Hebb学习等生物学习机制"]
            BaseSynapse --> StaticSynapse
            BaseSynapse --> STDPSynapse
        end

        subgraph "连接规则 (Connection Rules)"
            direction TB
            Connector["Connector (Base Class)<br/><b>功能:</b> 定义神经元之间的连接方式<br/><b>实现:</b> OneToOne, AllToAll, FixedProb<br/><b>输入:</b> 突触前/后神经元群体<br/><b>输出:</b> 连接矩阵或连接列表<br/><b>物理意义:</b> 构建神经网络的拓扑结构"]
        end
        
        PointNeuron --> BaseSynapse
        MCNeuron --> BaseSynapse
        Generators --> BaseSynapse
        Connector --> BaseSynapse
    end

    subgraph "网络与仿真引擎 (Network & Simulation Engine)"
        direction LR
        SNet["SNet (Spiking Network)<br/><b>功能:</b> 核心仿真引擎, 组织神经元和突触<br/><b>关键函数:</b> __call__() 执行单步仿真<br/><b>输入:</b> 神经元/突触对象列表<br/><b>输出:</b> 仿真结果"]
        Solvers["数学求解器 (Math Solvers)<br/>Euler, RK4<br/><b>功能:</b> 求解神经元模型的常微分方程(ODE)<br/><b>调用:</b> 被Neuron的__call__方法调用<br/><b>输入:</b> 当前状态, 微分方程, dt<br/><b>输出:</b> 下一时刻的状态"]
        Parallel["并行计算 (Parallelization)<br/><b>技术:</b> mpi4jax<br/><b>功能:</b> 支持多设备/多节点分布式仿真<br/><b>机制:</b> 在SNet中处理跨进程的脉冲传递"]
    end

    subgraph "监控与记录 (Monitoring & Recording)"
        direction LR
        Monitor["MonitorBS<br/><b>功能:</b> 设置监控任务<br/><b>关键参数:</b> 要记录的变量(spikes, v), 监控时间<br/><b>输出:</b> 监控配置"]
        Recorder["Recorder<br/>MemoryRecorder, FileRecorder<br/><b>功能:</b> 从网络中抓取并存储数据<br/><b>输入:</b> SNet中的变量<br/><b>输出:</b> 内存中的数据或HDF5文件<br/><b>物理意义:</b> 模拟实验中的电生理记录"]
    end

    %% 调用关系 (Call Relationships)
    Config -- "提供全局参数" --> SNet
    SNet -- "包含并管理" --> NeuronModels[神经元/发生器]
    SNet -- "包含并管理" --> SynapseModels[突触]
    NeuronModels -- "状态更新依赖" --> Solvers
    SNet -- "执行仿真循环, 调用" --> NeuronModels
    SNet -- "执行仿真循环, 调用" --> SynapseModels
    SNet -- "分布式计算依赖" --> Parallel
    Monitor -- "配置监控任务给" --> SNet
    Recorder -- "从SNet中记录" --> SNet

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef subgraph-style fill:#ececff,stroke:#9370db,stroke-width:2px,color:#333;
    class SNet,Config,BaseNeuron,BaseSynapse,Connector,Solvers,Parallel,Monitor,Recorder subgraph-style
