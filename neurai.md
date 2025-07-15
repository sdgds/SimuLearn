```mermaid
graph TB
    %% Core Configuration
    Config[Config Management]
    Config --> Platform[Platform: CPU/GPU/TPU]
    Config --> SimSettings[Simulation Settings]
    
    %% Neuron Models
    subgraph "Neuron Models"
        BaseNeuron[Base Neuron Class]
        LIF[LIF Neuron]
        ExpLIF[ExpLIF Neuron]
        HH[Hodgkin-Huxley]
        Izhikevich[Izhikevich Model]
        AEIF[AEIF Model]
        MCNeuron[Multi-Compartment Neuron]
        
        BaseNeuron --> LIF
        BaseNeuron --> ExpLIF
        BaseNeuron --> HH
        BaseNeuron --> Izhikevich
        BaseNeuron --> AEIF
        BaseNeuron --> MCNeuron
    end
    
    %% Spike Generators
    subgraph "Spike Generators"
        PoissonGen[Poisson Generator]
        BernoulliGen[Bernoulli Generator]
        DCGen[DC Generator]
        SpikeGen[Spike Generator]
        CustomGen[Custom Generator]
    end
    
    %% Synapse Models
    subgraph "Synapse Models"
        BaseSynapse[Base Synapse Class]
        StaticSyn[Static Synapse]
        STDPSyn[STDP Synapse]
        STDPAll2All[STDP All-to-All]
        STDPNearest[STDP Nearest]
        STDPTriplet[STDP Triplet]
        
        BaseSynapse --> StaticSyn
        BaseSynapse --> STDPSyn
        STDPSyn --> STDPAll2All
        STDPSyn --> STDPNearest
        STDPSyn --> STDPTriplet
    end
    
    %% Connection Rules
    subgraph "Connection Rules"
        ConnRules[Connection Rules]
        OneToOne[One-to-One]
        AllToAll[All-to-All]
        Random[Random Connection]
        Gaussian[Gaussian Connection]
    end
    
    %% Network Structure
    subgraph "Network Management"
        SNet[SNet - Spiking Network]
        ProxyGroup[Proxy Neuron Group]
        
        SNet --> ProxyGroup
    end
    
    %% Mathematical Solvers
    subgraph "Mathematical Solvers"
        ODESolver[ODE Solver]
        SDESolver[SDE Solver]
        EulerMethod[Euler Method]
        RK4Method[Runge-Kutta 4]
    end
    
    %% Monitoring System
    subgraph "Monitoring & Recording"
        Monitor[Monitor Base]
        MonitorBS[Brain Simulation Monitor]
        Recorder[Recorder Base]
        MemoryRec[Memory Recorder]
        FileRec[File Recorder]
        CallbackRec[Callback Recorder]
        
        Monitor --> MonitorBS
        Recorder --> MemoryRec
        Recorder --> FileRec
        Recorder --> CallbackRec
    end
    
    %% Parallelization
    subgraph "Parallel Computing"
        MPI[MPI Support]
        MultiThread[Multi-threading]
        JAXParallel[JAX Parallelization]
    end
    
    %% Gradient Computation
    subgraph "Gradient & Training"
        SurrogateGrad[Surrogate Gradient]
        BackProp[Backpropagation]
        BPTT[BPTT for SNNs]
    end
    
    %% Data Flow Connections
    Config --> SNet
    
    %% Neurons to Network
    LIF --> SNet
    ExpLIF --> SNet
    HH --> SNet
    Izhikevich --> SNet
    AEIF --> SNet
    MCNeuron --> SNet
    
    %% Generators to Network
    PoissonGen --> SNet
    BernoulliGen --> SNet
    DCGen --> SNet
    SpikeGen --> SNet
    CustomGen --> SNet
    
    %% Synapses to Network
    StaticSyn --> SNet
    STDPAll2All --> SNet
    STDPNearest --> SNet
    STDPTriplet --> SNet
    
    %% Connection Rules to Synapses
    ConnRules --> StaticSyn
    ConnRules --> STDPSyn
    
    %% Solvers to Neurons
    ODESolver --> MCNeuron
    SDESolver --> MCNeuron
    EulerMethod --> LIF
    EulerMethod --> ExpLIF
    RK4Method --> HH
    
    %% Monitoring
    MonitorBS --> SNet
    MemoryRec --> MonitorBS
    FileRec --> MonitorBS
    
    %% Parallelization
    MPI --> SNet
    MultiThread --> SNet
    JAXParallel --> SNet
    
    %% Gradient Flow
    SurrogateGrad --> LIF
    SurrogateGrad --> ExpLIF
    BackProp --> SNet
    BPTT --> SNet
    
    %% Simulation Flow
    subgraph "Simulation Process"
        Init[Initialize Network]
        Setup[Setup Neurons and Synapses]
        TimeLoop[Time Step Loop]
        Update[Update States]
        Record[Record Data]
        
        Init --> Setup
        Setup --> TimeLoop
        TimeLoop --> Update
        Update --> Record
        Record --> TimeLoop
    end
    
    SNet --> Init
    Config --> Setup
    MonitorBS --> Record
