```mermaid
graph TB
    %% 核心接口层
    A[SimuLearn Core API] --> B[Network Builder]
    A --> C[Simulation Engine]
    A --> D[Learning Engine]
    A --> E[Analysis & Visualization]
    
    %% 网络构建模块
    B --> B1[Node Manager<br/>- 神经元模型注册<br/>- 参数管理<br/>- 类型验证]
    B --> B2[Edge Manager<br/>- 连接规则<br/>- 突触模型<br/>- 权重初始化]
    B --> B3[Network Topology<br/>- 空间结构<br/>- 层次组织<br/>- 群体定义]
    B --> B4[Model Registry<br/>- GLIF模型<br/>- LIF模型<br/>- 自定义模型]
    
    %% 仿真引擎模块
    C --> C1[Simulator Backend<br/>- NEST接口<br/>- NEURON接口<br/>- 自定义求解器]
    C --> C2[State Manager<br/>- 膜电位追踪<br/>- 脉冲记录<br/>- 状态变量管理]
    C --> C3[Event Scheduler<br/>- 时间步进<br/>- 事件队列<br/>- 同步机制]
    C --> C4[Device Manager<br/>- 刺激设备<br/>- 记录设备<br/>- 外部输入]
    
    %% 学习引擎模块
    D --> D1[Gradient Engine<br/>- BPTT实现<br/>- 替代梯度<br/>- 自动微分]
    D --> D2[Learning Rules<br/>- STDP<br/>- 监督学习<br/>- 强化学习]
    D --> D3[Optimizer Interface<br/>- Adam<br/>- SGD<br/>- 自适应学习率]
    D --> D4[Loss Functions<br/>- 分类损失<br/>- 回归损失<br/>- 正则化项]
    
    %% 分析可视化模块
    E --> E1[Data Recorder<br/>- 脉冲数据<br/>- 电压轨迹<br/>- 权重变化]
    E --> E2[Metrics Calculator<br/>- 发放率<br/>- 同步性<br/>- 学习性能]
    E --> E3[Visualization Tools<br/>- 栅格图<br/>- 网络图<br/>- 学习曲线]
    E --> E4[Export Interface<br/>- SONATA格式<br/>- HDF5格式<br/>- 标准格式]
    
    %% 底层支撑模块
    F[Memory Management<br/>- 稀疏矩阵<br/>- 缓存优化<br/>- 内存池] --> A
    G[Parallel Computing<br/>- 多线程<br/>- GPU加速<br/>- 分布式计算] --> A
    H[Configuration System<br/>- JSON配置<br/>- 参数验证<br/>- 默认设置] --> A
    
    %% 数据流连接
    B1 --> C1
    B2 --> C2
    B4 --> D1
    C2 --> D2
    C3 --> D3
    D1 --> E1
    D2 --> E2
    
    %% 样式定义
    classDef coreModule fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef networkModule fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef simModule fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef learnModule fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef analysisModule fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef supportModule fill:#f5f5f5,stroke:#424242,stroke-width:2px
    
    class A coreModule
    class B,B1,B2,B3,B4 networkModule
    class C,C1,C2,C3,C4 simModule
    class D,D1,D2,D3,D4 learnModule
    class E,E1,E2,E3,E4 analysisModule
    class F,G,H supportModule
