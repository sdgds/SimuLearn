```mermaid
graph TB
    %% 核心层
    subgraph Core ["🔧 核心层 (Core)"]
        NeuronModels["神经元模型库<br/>NeuronModels<br/>• BaseNeuron<br/>• GLIF3Neuron<br/>• LIFNeuron<br/>• AdExNeuron<br/>• CustomNeuron"]
        NetworkBuilder["网络构建器<br/>NetworkBuilder<br/>• NetworkTopology<br/>• ConnectionManager<br/>• SynapseManager"]
        SimEngine["仿真引擎<br/>SimulationEngine<br/>• TimeStepSimulator<br/>• EventDrivenSimulator<br/>• HybridSimulator"]
    end
    
    %% 数据层
    subgraph Data ["📊 数据层 (Data)"]
        DataLoader["数据加载器<br/>DataLoader<br/>• BMTKLoader<br/>• NESTLoader<br/>• CustomLoader"]
        DataConverter["数据转换器<br/>DataConverter<br/>• FormatConverter<br/>• ParameterMapper<br/>• ConnectivityConverter"]
        DataManager["数据管理器<br/>DataManager<br/>• InputManager<br/>• OutputManager<br/>• StateManager"]
    end
    
    %% 学习层
    subgraph Learning ["🧠 学习层 (Learning)"]
        LearningAlgs["学习算法库<br/>LearningAlgorithms<br/>• BPTT<br/>• STDP<br/>• ReinforcementLearning<br/>• UnsupervisedLearning"]
        Optimizers["优化器<br/>Optimizers<br/>• SGD<br/>• Adam<br/>• RMSprop<br/>• CustomOptimizer"]
        LossFunc["损失函数<br/>LossFunctions<br/>• SpikeLoss<br/>• RateLoss<br/>• VoltageLoss<br/>• CustomLoss"]
    end
    
    %% 训练层
    subgraph Training ["🎯 训练层 (Training)"]
        Trainer["训练器<br/>Trainer<br/>• SupervisedTrainer<br/>• UnsupervisedTrainer<br/>• ReinforcementTrainer"]
        TaskManager["任务管理器<br/>TaskManager<br/>• ClassificationTask<br/>• RegressionTask<br/>• CustomTask"]
        Scheduler["调度器<br/>Scheduler<br/>• LearningRateScheduler<br/>• TrainingScheduler"]
    end
    
    %% 分析层
    subgraph Analysis ["📈 分析层 (Analysis)"]
        Metrics["指标计算<br/>Metrics<br/>• SpikeMetrics<br/>• NetworkMetrics<br/>• LearningMetrics"]
        Visualizer["可视化器<br/>Visualizer<br/>• SpikeRaster<br/>• NetworkActivity<br/>• LearningCurves"]
        Analyzer["分析器<br/>Analyzer<br/>• ConnectivityAnalyzer<br/>• DynamicsAnalyzer<br/>• PerformanceAnalyzer"]
    end
    
    %% 接口层
    subgraph Interface ["🔌 接口层 (Interface)"]
        API["API接口<br/>API<br/>• SimulationAPI<br/>• LearningAPI<br/>• AnalysisAPI"]
        CLI["命令行接口<br/>CLI<br/>• SimuLearnCLI"]
        GUI["图形界面<br/>GUI<br/>• InteractiveGUI"]
    end
    
    %% 数据流向
    Data --> Core
    Core --> Learning
    Learning --> Training
    Core --> Analysis
    Training --> Analysis
    Analysis --> Interface
    Training --> Interface
    Core --> Interface
    
    %% 反馈循环
    Training -.->|参数更新| Core
    Analysis -.->|性能反馈| Training
    Interface -.->|用户输入| Training
    Interface -.->|配置| Core
