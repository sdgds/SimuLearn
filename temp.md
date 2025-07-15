```mermaid
graph TD
    subgraph "输入层: BMTK/SONATA 生态系统"
        direction LR
        BMTK_Builder["BMTK Network Builder"]
        SONATA["SONATA 文件<br/>(nodes.h5, edges.h5, *.csv, *.json)"]
        BMTK_Builder --> SONATA
    end

    subgraph "SimuLearn 核心: 模型加载与转换"
        direction LR
        Loader["SonataLoader<br/><b>功能:</b> 解析SONATA格式文件<br/><b>输入:</b> SONATA文件路径<br/><b>输出:</b> 内部网络表示 (节点, 连接, 参数)"]
        Converter["ModelConverter<br/><b>功能:</b> 将SONATA/NEST模型映射到SpikingJelly<br/><b>输入:</b> 内部网络表示<br/><b>输出:</b> SpikingJelly兼容的层和参数"]
        SONATA --> Loader
        Loader --> Converter
    end

    subgraph "网络构建与仿真 (SpikingJelly)"
        direction LR
        SJ_Network["SpikingJelly Network (torch.nn.Module)<br/><b>功能:</b> 包含LIF神经元层和可训练的连接层(nn.Linear)<br/><b>构建:</b> 由Converter输出的组件组装而成"]
        Simulator["Simulator<br/><b>功能:</b> 运行前向仿真<br/><b>输入:</b> 输入脉冲, 仿真时长<br/><b>输出:</b> 输出脉冲, 神经元状态(膜电位等)<br/><b>物理意义:</b> 模拟网络在给定输入下的动态响应"]
        Converter --> SJ_Network
        SJ_Network --> Simulator
    end

    subgraph "学习引擎 (BPTT)"
        direction LR
        Trainer["BPTT Trainer<br/><b>功能:</b> 通过时间反向传播算法训练网络<br/><b>关键步骤:</b> 1. 运行前向仿真<br/>2. 计算损失函数<br/>3. 自动微分求梯度<br/>4. 更新权重<br/><b>输入:</b> 仿真结果, 目标输出<br/><b>输出:</b> 优化后的网络参数"] 
        Loss["Loss Function<br/>(e.g., van Rossum distance, MSE)<br/><b>功能:</b> 定义学习目标<br/><b>输入:</b> 实际输出, 期望输出<br/><b>输出:</b> 损失值<br/><b>物理意义:</b> 量化网络行为与期望行为的差距"]
        Optimizer["Optimizer (e.g., Adam, SGD)<br/><b>功能:</b> 根据梯度更新网络权重<br/><b>输入:</b> 网络参数, 学习率<br/><b>输出:</b> 更新后的参数"] 
        Simulator --> Trainer
        Trainer --> Loss
        Loss --> Trainer
        Trainer --> Optimizer
        Optimizer --> SJ_Network
    end

    subgraph "监控与输出"
        direction LR
        Monitor["SpikingJelly Monitor<br/><b>功能:</b> 记录脉冲和状态变量<br/><b>输入:</b> 网络中的层<br/><b>输出:</b> 脉冲序列, 膜电位轨迹"] 
        Output["输出<br/>(HDF5, Npz, etc.)<br/><b>功能:</b> 保存训练结果和仿真数据"] 
        SJ_Network --> Monitor
        Trainer --> Output
        Monitor --> Output
    end

    %% 关系
    BMTK_Builder -- "生成" --> SONATA
    SONATA -- "加载" --> Loader
    Loader -- "传递解析数据" --> Converter
    Converter -- "构建" --> SJ_Network
    SJ_Network -- "用于" --> Simulator
    Simulator -- "提供仿真轨迹" --> Trainer
    Trainer -- "驱动" --> Optimizer
    Optimizer -- "更新" --> SJ_Network

    %% 样式定义
    classDef simulationStyle fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px
    classDef learningStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef ioStyle fill:#fffde7,stroke:#f57f17,stroke-width:2px
    classDef coreStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px

    class BMTK_Builder,SONATA ioStyle
    class Loader,Converter coreStyle
    class SJ_Network,Simulator,Monitor,Output simulationStyle
    class Trainer,Loss,Optimizer learningStyle
