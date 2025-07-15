```mermaid
graph TD
    subgraph "配置管理 (Configuration)"
        direction LR
        Config["Config (Singleton Class)<br/><b>功能:</b> 管理全局仿真参数<br/><b>关键函数:</b> set_platform(), set_simulate_status()<br/><b>输出:</b> 全局配置对象 (如dt, JIT开关)<br/><b>物理意义:</b> 设定仿真环境和基本物理约束"]
    end

    subgraph "外部模型导入 (External Model Import)"
        direction LR
        BMTKLoader["BMTK/SONATA Loader<br/><b>功能:</b> 解析BMTK生成的network/components文件<br/><b>输入:</b> SONATA格式文件路径<br/><b>输出:</b> 网络拓扑和参数字典"]
        NESTAdapter["NEST Model Adapter<br/><b>功能:</b> 将NEST神经元模板映射到SpikingJelly<br/><b>输入:</b> NEST模型名称和参数<br/><b>输出:</b> 对应的SpikingJelly神经元配置"]
    end

    subgraph "网络构建 (Network Construction)"
        direction LR
        subgraph "神经元模型 (Neuron Models)"
            direction TB
            BaseNeuron["Neuron (Base Class)<br/><b>功能:</b> 定义神经元通用接口<br/><b>关键参数:</b> size, V_th, V_reset"]
            PointNeuron["点神经元模型<br/>LIF, ExpLIF, Izhikevich, HH<br/><b>功能:</b> 实现单个神经元的动力学方程<br/><b>输入:</b> 突触电流 I_syn, 外部激励 I_e<br/><b>输出:</b> 膜电位 V_m, 脉冲<br/><b>物理意义:</b> 模拟单个神经元的电生理活动"]
            MCNeuron["多室神经元 (MCNeuron)<br/><b>功能:</b> 模拟具有复杂树突形态的神经元<br/><b>输入:</b> 各区室的突触电流<br/><b>输出:</b> 各区室的膜电位, 脉冲<br/><b>物理意义:</b> 更精细地模拟神经元形态和电信号整合"]
            Generators["脉冲发生器<br/>Poisson, Bernoulli, SpikeGenerator<br/><b>功能:</b> 产生外部输入脉冲序列<br/><b>输出:</b> 脉冲序列<br/><b>物理意义:</b> 模拟来自大脑其他区域或感觉器官的输入信号"]
            SJWrapper["SpikingJelly Wrapper<br/><b>功能:</b> 将内部神经元模型包装为SpikingJelly兼容层<br/><b>实现:</b> 继承torch.nn.Module<br/><b>输出:</b> 可微分的神经元模型"]
            BaseNeuron --> PointNeuron
            BaseNeuron --> MCNeuron
            PointNeuron --> SJWrapper
            MCNeuron --> SJWrapper
        end

        subgraph "突触模型 (Synapse Models)"
            direction TB
            BaseSynapse["Synapse (Base Class)<br/><b>功能:</b> 定义突触通用接口<br/><b>关键参数:</b> pre, post, conn, weight"]
            StaticSynapse["静态突触 (StaticSynapse)<br/><b>功能:</b> 传递固定强度的突触后电流<br/><b>输入:</b> 突触前脉冲<br/><b>输出:</b> 突触后电流<br/><b>物理意义:</b> 强度不变的神经连接"]
            STDPSynapse["STDP 学习突触<br/>STDPAll2All, STDPNearest<br/><b>功能:</b> 根据脉冲时序依赖可塑性规则调整突触权重<br/><b>输入:</b> 突触前/后脉冲时间<br/><b>输出:</b> 突触后电流, 更新后的权重<br/><b>物理意义:</b> 模拟Hebb学习等生物学习机制"]
            LearnableSynapse["可学习突触<br/><b>功能:</b> 支持梯度下降的突触权重<br/><b>实现:</b> torch.nn.Parameter包装<br/><b>输出:</b> 可通过BPTT优化的突触连接"]
            BaseSynapse --> StaticSynapse
            BaseSynapse --> STDPSynapse
            BaseSynapse --> LearnableSynapse
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
        SJBackend["SpikingJelly Backend<br/><b>功能:</b> 使用SpikingJelly加速仿真<br/><b>技术:</b> CuPy/Torch JIT编译<br/><b>输出:</b> GPU加速的仿真结果"]
        Solvers["数学求解器 (Math Solvers)<br/>Euler, RK4<br/><b>功能:</b> 求解神经元模型的常微分方程(ODE)<br/><b>调用:</b> 被Neuron的__call__方法调用<br/><b>输入:</b> 当前状态, 微分方程, dt<br/><b>输出:</b> 下一时刻的状态"]
        Parallel["并行计算 (Parallelization)<br/><b>技术:</b> mpi4jax<br/><b>功能:</b> 支持多设备/多节点分布式仿真<br/><b>机制:</b> 在SNet中处理跨进程的脉冲传递"]
    end

    subgraph "学习引擎 (Learning Engine)"
        direction LR
        BPTTEngine["BPTT Engine<br/><b>功能:</b> 通过时间反向传播计算梯度<br/><b>技术:</b> SpikingJelly的surrogate gradient<br/><b>输入:</b> 损失函数, 网络输出<br/><b>输出:</b> 参数梯度"]
        LossFunction["Loss Functions<br/><b>实现:</b> MSE, CrossEntropy, SpikingRate<br/><b>功能:</b> 定义学习目标<br/><b>输入:</b> 预测值, 目标值<br/><b>输出:</b> 损失标量"]
        Optimizer["Optimizer<br/><b>实现:</b> Adam, SGD, RMSprop<br/><b>功能:</b> 更新网络参数<br/><b>输入:</b> 梯度<br/><b>输出:</b> 更新后的参数"]
        Trainer["Training Manager<br/><b>功能:</b> 管理训练循环<br/><b>流程:</b> 前向传播→损失计算→反向传播→参数更新<br/><b>输出:</b> 训练后的模型"]
    end

    subgraph "监控与记录 (Monitoring & Recording)"
        direction LR
        Monitor["MonitorBS<br/><b>功能:</b> 设置监控任务<br/><b>关键参数:</b> 要记录的变量(spikes, v), 监控时间<br/><b>输出:</b> 监控配置"]
        Recorder["Recorder<br/>MemoryRecorder, FileRecorder<br/><b>功能:</b> 从网络中抓取并存储数据<br/><b>输入:</b> SNet中的变量<br/><b>输出:</b> 内存中的数据或HDF5文件<br/><b>物理意义:</b> 模拟实验中的电生理记录"]
    end

    %% 调用关系 (Call Relationships)
    Config -- "提供全局参数" --> SNet
    BMTKLoader -- "导入网络结构" --> SNet
    NESTAdapter -- "转换神经元模型" --> PointNeuron
    SNet -- "包含并管理" --> PointNeuron
    SNet -- "包含并管理" --> MCNeuron
    SNet -- "包含并管理" --> Generators
    SNet -- "包含并管理" --> StaticSynapse
    SNet -- "包含并管理" --> STDPSynapse
    SNet -- "包含并管理" --> LearnableSynapse
    SNet -- "使用加速后端" --> SJBackend
    PointNeuron -- "状态更新依赖" --> Solvers
    MCNeuron -- "状态更新依赖" --> Solvers
    SNet -- "分布式计算依赖" --> Parallel
    Monitor -- "配置监控任务给" --> SNet
    Recorder -- "从SNet中记录" --> SNet
    SJWrapper -- "提供可微分接口给" --> BPTTEngine
    LearnableSynapse -- "参数优化" --> BPTTEngine
    BPTTEngine -- "计算损失" --> LossFunction
    BPTTEngine -- "梯度传递给" --> Optimizer
    Trainer -- "协调" --> BPTTEngine
    Trainer -- "管理" --> SNet

    %% 样式定义
    classDef constructStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef simStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef learnStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef importStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    
    class BaseNeuron,PointNeuron,MCNeuron,Generators,BaseSynapse,StaticSynapse,STDPSynapse,Connector,SJWrapper,LearnableSynapse constructStyle;
    class Config,SNet,Solvers,Parallel,Monitor,Recorder,SJBackend simStyle;
    class BPTTEngine,LossFunction,Optimizer,Trainer learnStyle;
    class BMTKLoader,NESTAdapter importStyle;
