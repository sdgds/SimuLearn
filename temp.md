```mermaid
graph TD
    %% BMTK 输入层
    A[BMTK Network Builder] --> B[SONATA Files]
    B --> C[nodes.h5]
    B --> D[edges.h5] 
    B --> E[node_types.csv]
    B --> F[edge_types.csv]
    B --> G[components/*.json]
    
    %% SimuLearn 核心层
    H[SimuLearn Core] --> I[SonataLoader]
    H --> J[ModelConverter]
    H --> K[NetworkBuilder]
    H --> L[BPTTTrainer]
    
    %% 数据流
    C --> I
    D --> I
    E --> I
    F --> I
    G --> I
    
    I --> M[Parse Nodes]
    I --> N[Parse Edges]
    I --> O[Load Parameters]
    
    M --> J
    N --> J
    O --> J
    
    %% 模型转换
    J --> P[NEST to SpikingJelly Mapping]
    P --> Q[LIFNode Creation]
    P --> R[Connection Matrix]
    P --> S[Parameter Mapping]
    
    Q --> K
    R --> K
    S --> K
    
    %% 网络构建
    K --> T[PyTorch nn.Module]
    T --> U[Spiking Neural Network]
    
    %% 训练层
    U --> L
    L --> V[Forward Simulation]
    L --> W[Loss Calculation]
    L --> X[Backward Propagation]
    L --> Y[Weight Update]
    
    %% SpikingJelly 集成
    Z[SpikingJelly Framework] --> AA[activation_based.neuron]
    Z --> BB[activation_based.layer]
    Z --> CC[activation_based.functional]
    
    AA --> Q
    BB --> K
    CC --> L
    
    %% 输出层
    Y --> DD[Trained Weights]
    V --> EE[Spike Trains]
    W --> FF[Training Metrics]
    
    %% 兼容性层
    GG[NEST Compatibility] --> P
    HH[BMTK Ecosystem] --> I
    II[PyTorch Ecosystem] --> L
    
    %% 样式定义
    classDef bmtkStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef simulStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef sjStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef dataStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class A,B,C,D,E,F,G,GG,HH bmtkStyle
    class H,I,J,K,L,M,N,O,P simulStyle
    class Z,AA,BB,CC,Q,R,S,T,U,V,W,X,Y,II sjStyle
    class DD,EE,FF dataStyle
