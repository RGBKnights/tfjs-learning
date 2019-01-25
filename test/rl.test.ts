import {expect, use} from 'chai';
import {range, shuffle} from 'lodash';
import {memory} from '@tensorflow/tfjs-core';

import generated from "sinon-chai";
import {Academy, LayerType, Model} from "../src";

use(generated);

const initialInputSize = 100;
const numActions = 2;
const inputSize = initialInputSize + numActions + initialInputSize;

const model = new Model(null, {stepsPerEpoch: 1, epochs: 1});
model.addLayer(LayerType.DENSE, {units: 128, activation: 'relu', inputShape: [inputSize]});
model.addLayer(LayerType.DENSE, {units: 128, activation: 'relu'});
model.addLayer(LayerType.DENSE, {units: numActions, activation: 'relu'});
model.compile({loss: 'meanSquaredError', optimizer: 'adam'});

const lessonLength = 10;
const lessons = 10;
const randomSteps = 0;
const batchSize = 32;
const memorySize = 100;

const academy = new Academy();
const agent = academy.addAgent({model: model, agentConfig: {batchSize: batchSize, memorySize: memorySize}});
const teacher = academy.addTeacher({
    lessonLength: lessonLength,
    lessonsQuantity: lessons,
    lessonsWithRandom: randomSteps
});
academy.assignTeacherToAgent(agent, teacher);

describe("ReImprove - Real", () => {
    beforeEach(() => {
        academy.resetTeachersAndAgents();
    });

    it('should have no tensor memory overflow', async () => {
        let input = shuffle(range(0, initialInputSize)).map(v => v / initialInputSize);

        let results;
        for (let i = 0; i < lessonLength * lessons; ++i) {
            results = await academy.step([
                {
                    teacherName: teacher,
                    agentsInput: input
                }
            ]);
            academy.addRewardToAgent(agent, results.get(agent) == 1 ? 1.0 : -1.0);
        }

        expect(memory().numTensors).to.be.approximately(memorySize*2, memorySize*0.5);

        for (let i = 0; i < lessonLength * lessons; ++i) {
            results = await academy.step([
                {
                    teacherName: teacher,
                    agentsInput: input
                }
            ]);
            academy.addRewardToAgent(agent, results.get(agent) == 1 ? 1.0 : -1.0);
        }

        expect(memory().numTensors).to.be.approximately(memorySize*2, memorySize*0.5);
    });
});