```mermaid
graph TD
    subgraph "输入层 (Input Layer) - BMTK/SONATA" 
        direction LR
        style InputLayer fill:#f2f2f2,stroke:#666,stroke-width:2px
        Nodes["节点文件<br/>(nodes.h5, node_types.csv)"]
        Edges["边文件<br/>(edges.h5, edge_types.csv)"]
        Components["模型组件<br/>(neuron_models.json, syn_models.json)"]
    end

    subgraph "SimuLearn 工具包 (SimuLearn Toolkit)"
        direction TB
        subgraph "<font color='blue'>网络构建与加载 (Network Construction & Loading)</font>"
            direction LR
            style Construction fill:#e0f7fa,stroke:#00796b,stroke-width:2px
            SonataLoader["<b>SONATA 加载器 (SonataLoader)</b><br/><b>功能:</b> 解析BMTK/NEST网络文件<br/><b>输入:</b> SONATA格式文件<br/><b>输出:</b> 网络拓扑和参数的Python对象<br/><b>兼容性:</b> 直接利用BMTK和NEST定义的模型"]
            NetworkBuilder["<b>网络构建器 (NetworkBuilder)</b><br/><b>功能:</b> 将解析的数据转换为SpikingJelly模型<br/><b>输入:</b> Loader输出的Python对象<br/><b>输出:</b> SpikingJelly网络 (torch.nn.Module)"]
        end

        SJ_Network["<b>SpikingJelly 网络 (torch.nn.Module)</b><br/>包含SpikingJelly的神经元、突触和连接器"]

        subgraph "<font color='green'>仿真引擎 (Simulation Engine)</font>"
            direction LR
            style Simulation fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
            Simulator["<b>前向模拟器 (Simulator)</b><br/><b>功能:</b> 执行网络的前向传播<br/><b>核心:</b> 调用SpikingJelly网络进行单步或多步仿真<br/><b>输出:</b> 脉冲、膜电位等状态"]
            Monitor["<b>监视器 (Monitor)</b><br/><b>功能:</b> 记录仿真过程中的网络状态<br/><b>实现:</b> 基于SpikingJelly的Monitor<br/><b>输出:</b> HDF5或内存中的数据"]
        end

        subgraph "<font color='orange'>学习引擎 (Learning Engine)</font>"
            direction LR
            style Learning fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
            BPTTEngine["<b>BPTT 训练器 (BPTT Trainer)</b><br/><b>功能:</b> 使用BPTT算法训练网络参数<br/><b>核心:</b> SpikingJelly的函数式编程接口<br/><b>输入:</b> 网络, 输入数据, 目标输出<br/><b>输出:</b> 训练好的网络模型"]
            Loss["<b>损失函数 (Loss Function)</b><br/>定义学习任务的目标"]
            Optimizer["<b>优化器 (Optimizer)</b><br/>如Adam, SGD，更新网络权重"]
        end
    end

    subgraph "输出层 (Output Layer)"
        direction LR
        style OutputLayer fill:#f2f2f2,stroke:#666,stroke-width:2px
        SimResults["仿真结果<br/>(脉冲、膜电位)"]
        TrainedModel["训练后的模型<br/>(优化的网络参数)"]
    end

    %% 流程关系
    Nodes --> SonataLoader
    Edges --> SonataLoader
    Components --> SonataLoader
    SonataLoader --> NetworkBuilder
    NetworkBuilder --> SJ_Network
    
    SJ_Network --> Simulator
    Simulator --> Monitor
    Monitor --> SimResults

    SJ_Network --> BPTTEngine
    Loss --> BPTTEngine
    Optimizer --> BPTTEngine
    BPTTEngine --> TrainedModel

    classDef Construction fill:#e0f7fa,stroke:#00796b,stroke-width:2px,color:blue;
    classDef Simulation fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:green;
    classDef Learning fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:orange;
    class SonataLoader,NetworkBuilder Construction;
    class Simulator,Monitor Simulation;
    class BPTTEngine,Loss,Optimizer Learning;
    class SJ_Network fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;
