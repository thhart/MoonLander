/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.itth.moonlander;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.modality.rl.env.RlEnv.Step;
import ai.djl.ndarray.NDList;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.RandomUtils;
import org.apache.logging.log4j.*;
import org.nd4j.shade.guava.util.concurrent.RateLimiter;

/**
 * The {@link EpsilonGreedy} is a simple exploration/excitation agent.
 *
 * <p>It helps other agents explore their environments during training by sometimes picking random
 * actions.
 *
 * <p>If a model based agent is used, it will only explore paths through the environment that have
 * already been seen. While this is sometimes good, it is also important to sometimes explore new
 * paths as well. This agent exhibits a tradeoff that takes random paths a fixed percentage of the
 * time during training.
 */
public class EpsilonGreedy implements RlAgent {
    protected final static Logger logger = LogManager.getLogger(EpsilonGreedy.class);

    private RlAgent baseAgent;
    private Tracker exploreRate;

    private final RateLimiter limiter = RateLimiter.create(0.1);

    public Map<Float, Float> getMap() {
        return map;
    }

    private final Map<Float, Float> map = new ConcurrentHashMap<>();
    private int counter;

    /**
     * Constructs an {@link EpsilonGreedy}.
     *
     * @param baseAgent the (presumably model-based) agent to use for exploitation and to train
     * @param exploreRate the probability of taking a random action
     */
    public EpsilonGreedy(RlAgent baseAgent, Tracker exploreRate) {
        this.baseAgent = baseAgent;
        this.exploreRate = exploreRate;
    }

    /** {@inheritDoc} */
    @Override
    public NDList chooseAction(RlEnv env, boolean training) {
        final NDList arrays;
        final float rate = exploreRate.getNewValue(counter++);
        if (training && RandomUtils.random() < rate) {
            arrays = env.getActionSpace().randomAction();
            return arrays;
        } else {
            arrays = baseAgent.chooseAction(env, training);
        }
        final float anInt = arrays.singletonOrThrow().getFloat();
        map.put(anInt, map.computeIfAbsent(anInt, integer -> 0F) + 1);
        if(limiter.tryAcquire()) {
            System.err.println("");
            System.err.println("Epsilon: " + rate);
            System.err.println("Action: " + map);
        }
        return arrays;
    }

    /** {@inheritDoc} */
    @Override
    public void trainBatch(Step[] batchSteps) {
        baseAgent.trainBatch(batchSteps);
    }
}
