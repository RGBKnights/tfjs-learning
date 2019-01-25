export {Model, LayerType} from "./learning/model";
export {Academy, BuildAgentConfig} from "./learning/academy";
export {AgentConfig} from "./learning/algorithms/agent_config";
export {TeachingConfig} from "./learning/teacher";

export {
    NeuralNetwork,
    ConvolutionalNeuralNetwork,
    ConvolutionalLayer,
    MaxPooling2DLayer,
    FlattenLayer,
    DenseLayer,
    DropoutLayer
} from './learning/networks';

export {QAgent} from "./learning/algorithms/q/qagent";
export {QState} from "./learning/algorithms/q/qstate";
export {QAction} from "./learning/algorithms/q/qaction";
export {QMatrix} from "./learning/algorithms/q/qmatrix";
export {QTransition} from "./learning/algorithms/q/qtransition";

export {setBackend} from "@tensorflow/tfjs";

// require('@tensorflow/tfjs-node');