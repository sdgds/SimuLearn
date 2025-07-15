graph TB
    %% 核心配置层
    Config[Config 配置管理]
    Config --> |平台设置| Platform[Platform: CPU/GPU/APU]
    Config --> |仿真参数| SimParams[dt, JIT编译, 多线程]
    
    %% 神经元层次结构
    subgraph "神经元模块 (Neuron Module)"
        BaseNeuron[Neuron 基类]
        BaseNeuron --> |继承| StepwiseNeuron[StepwiseNeuron 步进式神经元]
        BaseNeuron --> |继承| MCNeuron[MCNeuron 多室神经元]
        
        %% 具体神经元类型
        StepwiseNeuron --> LIF[LIF 漏积分发放神经元]
        StepwiseNeuron --> ExpLIF[ExpLIF 指数LIF神经元]
        StepwiseNeuron --> HH[HH Hodgkin-Huxley神经元]
        StepwiseNeuron --> Izhikevich[Izhikevich 神经元]
        StepwiseNeuron --> GLIF[GLIF 广义LIF神经元]
        StepwiseNeuron --> AEIF[AEIF 自适应指数积分发放]
        
        %% 生成器类型
        BaseNeuron --> Generator[Generator 信号生成器]
        Generator --> PoissonGen[PoissonGenerator 泊松生成器]
        Generator --> BernoulliGen[BernoulliGenerator 伯努利生成器]
        Generator --> DCGen[DCGenerator 直流生成器]
        Generator --> SpikeGen[SpikeGenerator 脉冲生成器]
        
        %% 多室神经元组件
        MCNeuron --> Compartment[Compartment 神经元室]
        MCNeuron --> MCConn[MCConn 室间连接]
        MCNeuron --> MCConnSynapse[MCConnSynapse 室间突触连接]
    end
    
    %% 突触模块
    subgraph "突触模块 (Synapse Module)"
        BaseSynapse[Synapse 基类]
        BaseSynapse --> StaticSynapse[StaticSynapse 静态突触]
        BaseSynapse --> LearningSynapse[LearningSynapse 学习突触]
        
        LearningSynapse --> STDPAll2All[STDPAll2All 全连接STDP]
        LearningSynapse --> STDPNearest[STDPNearest 最近邻STDP]
        LearningSynapse --> STDPTriplet[STDPTriplet 三元组STDP]
        
        %% 连接规则
        BaseSynapse --> ConnRule[连接规则]
        ConnRule --> All2All[All2All 全连接]
        ConnRule --> One2One[One2One 一对一]
        ConnRule --> FixedTotalNum[FixedTotalNum 固定总数]
    end
    
    %% 网络层
    subgraph "网络模块 (Network Module)"
        SNet[SNet 脉冲神经网络]
        SNet --> |管理| NeuronList[neurons: List 神经元列表]
        SNet --> |管理| SynapseList[synapses: List 突触列表]
        
        Module[Module 基础模块]
        Module --> |继承| SNet
        Module --> |继承| SNetLayer[SNetLayer 网络层]
    end
    
    %% 数学求解器
    subgraph "数学求解器 (Math Solvers)"
        ODESolver[ODE求解器]
        ODESolver --> EulerMethod[Euler 欧拉法]
        ODESolver --> RK4Method[RK4 四阶龙格库塔]
        ODESolver --> ExpEuler[ExpEuler 指数欧拉]
        
        SDESolver[SDE求解器]
        SDESolver --> Milstein[Milstein 方法]
    end
    
    %% 监控系统
    subgraph "监控模块 (Monitor Module)"
        Monitor[Monitor 监控基类]
        Monitor --> MonitorBS[MonitorBS 脑仿真监控]
        
        Recorder[Recorder 记录器]
        Recorder --> MemoryRecorder[MemoryRecorder 内存记录]
        Recorder --> FileRecorder[FileRecorder 文件记录]
        Recorder --> CallbackRecorder[CallbackRecorder 回调记录]
        
        MonitorConf[MonitorConf 监控配置]
        MonitorConf --> |配置| Monitor
    end
    
    %% 并行化模块
    subgraph "并行化模块 (Parallel Module)"
        Parallelize[Parallelize 并行化装饰器]
        MPISupport[MPI支持]
        MultiThread[多线程支持]
    end
    
    %% 梯度计算
    subgraph "梯度模块 (Gradient Module)"
        SurrogateGrad[SurrogateGrad 代理梯度]
        SurrogateGrad --> Sigmoid[Sigmoid 代理函数]
        SurrogateGrad --> Rectangular[Rectangular 矩形函数]
        SurrogateGrad --> Gaussian[Gaussian 高斯函数]
    end
    
    %% 数据流连接
    Config --> SNet
    BaseNeuron --> |输入输出| BaseSynapse
    BaseSynapse --> |连接| BaseNeuron
    
    %% 求解器连接
    ODESolver --> |求解| LIF
    ODESolver --> |求解| MCNeuron
    SDESolver --> |噪声处理| MCNeuron
    
    %% 监控连接
    SNet --> |监控| MonitorBS
    MonitorBS --> |记录| Recorder
    
    %% 并行化连接
    Parallelize --> |装饰| SNet
    MPISupport --> |支持| SNet
    
    %% 梯度连接
    SurrogateGrad --> |反向传播| BaseNeuron
    
    %% 仿真执行流程
    subgraph "仿真执行流程"
        Init[初始化网络]
        Init --> Setup[setup() 设置神经元和突触]
        Setup --> TimeLoop[时间循环]
        TimeLoop --> UpdateNeurons[更新神经元状态]
        UpdateNeurons --> UpdateSynapses[更新突触状态]
        UpdateSynapses --> Record[记录数据]
        Record --> |下一时间步| TimeLoop
        Record --> Output[输出结果]
    end
    
    %% 物理意义标注
    LIF -.-> |膜电位动力学| MembraneEq["dV/dt = (V_rest - V + RI)/τ"]
    HH -.-> |离子通道动力学| HHEq["Hodgkin-Huxley方程组"]
    StaticSynapse -.-> |突触传递| SynEq["I_syn = g * (V_post - E_rev)"]
    STDPAll2All -.-> |可塑性| STDPEq["Δw ∝ A+ * exp(-Δt/τ+)"]
    
    %% 样式定义
    classDef neuronClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef synapseClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef networkClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef mathClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef monitorClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class BaseNeuron,LIF,ExpLIF,HH,Izhikevich,GLIF,AEIF,MCNeuron neuronClass
    class BaseSynapse,StaticSynapse,STDPAll2All,STDPNearest synapseClass
    class SNet,Module,SNetLayer networkClass
    class ODESolver,SDESolver,EulerMethod,RK4Method mathClass
    class Monitor,MonitorBS,Recorder monitorClass
