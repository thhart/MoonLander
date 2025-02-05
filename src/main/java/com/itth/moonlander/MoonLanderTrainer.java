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

import java.io.IOException;
import java.nio.file.Paths;
import ai.djl.*;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.rl.agent.*;
import ai.djl.modality.rl.env.RlEnv.Step;
import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.nn.*;
import ai.djl.training.*;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.*;
import me.tongfei.progressbar.ProgressBar;
import org.slf4j.*;

import com.itth.moonlander.samples.TicTacToe;

/**
 An example of training reinforcement learning using {@link TicTacToe} and a {@link QAgent}.

 <p>Note that the current setup, for simplicity, has one agent playing both sides to ensure X
 always wins.
 */
public final class MoonLanderTrainer {

	public static final Device DEVICE = Device.cpu();
	private static final Logger logger = LoggerFactory.getLogger(MoonLanderTrainer.class);

	public static SequentialBlock createBlock() {
		return new SequentialBlock()
				.add(
						arrays -> {
							NDArray board = arrays.get(0); // Shape(N, 9)
							NDArray turn = arrays.get(1).reshape(-1, 1); // Shape(N, 1)
							NDArray action = arrays.get(2).reshape(-1, 1); // Shape(N, 1)

							// Concatenate to a combined vector of Shape(N, 11)
							NDArray combined = NDArrays.concat(new NDList(board, turn, action), 1);

							return new NDList(combined.toType(DataType.FLOAT32, true));
						})
				.add(new Mlp(7, 1, new int[]{128, 64}));
	}

	public static DefaultTrainingConfig createConfig(int epoch, int gamesPerEpoch) {
		final CosineTracker cosineTracker = Tracker.cosine()
				.setMaxUpdates(epoch * gamesPerEpoch)
				//.setBaseValue(0.001F) // last working with cyclic tracker 10/04/23
				.setBaseValue(0.1F)
				.optFinalValue(0.001F).build();
		//Tracker rateTracker = Tracker.fixed(0.001F);
		//final Tracker cyclicalTracker = Tracker.warmUp()
		//				.optWarmUpBeginValue(0.001F).optWarmUpMode(Mode.LINEAR).optWarmUpSteps(512)
		//		.setMainTracker(rateTracker)
		//		.build();
		return new DefaultTrainingConfig(Loss.l2Loss())
				.optDevices(new Device[]{DEVICE})
				.addTrainingListeners(TrainingListener.Defaults.basic())
				.optOptimizer(
						//Adam.builder().optLearningRateTracker(Tracker.fixed(0.0001F)).build());
						//Adam.builder().optLearningRateTracker(Tracker.fixed(0.001F)).build());
						//Adam.builder().optLearningRateTracker(cosineTracker).build());
						Adam.builder().optLearningRateTracker(cosineTracker).build());
	}

	public static TrainingResult runExample(MoonLander moonLander) throws IOException {
		//int epoch = 512;
		int epoch = 128;
		int batchSize = 64;
		int replayBufferSize = 64 * 64;
		//int gamesPerEpoch = Math.toIntExact(1024);
		int gamesPerEpoch = Math.toIntExact(16);
		// Validation is deterministic, thus one game is enough
		int validationGamesPerEpoch = 1;
		float rewardDiscount = 0.9f;
		//Engine engine = Engine.getEngine("PyTorch");
		//System.out.println("Using backend engine: " + engine.getEngineName());
		//System.out.println("Found GPU: " + engine.getGpuCount());
		//try (BaseNDManager manager = (BaseNDManager)NDManager.newBaseManager(Device.cpu())) {
		try (BaseNDManager manager = (BaseNDManager)NDManager.newBaseManager(DEVICE)) {
			try (MoonLanderEnv game = new MoonLanderEnv(moonLander, manager, batchSize, replayBufferSize)) {

				Block block = createBlock();

				try (Model model = Model.newInstance("MoonLander")) {
					model.setBlock(block);

					DefaultTrainingConfig config = createConfig(epoch, gamesPerEpoch);
					try (Trainer trainer = model.newTrainer(config)) {
						trainer.initialize(
								new Shape(batchSize, 5), new Shape(batchSize), new Shape(batchSize));
						trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));
						// Constructs the agent to train and play with
						RlAgent agent = new QAgent(trainer, rewardDiscount);
						Tracker exploreRate =
								LinearTracker.builder()
										.setBaseValue(0.90f)
										.optSlope(-.9f / (epoch * gamesPerEpoch))
										.optMinValue(0.1f)
										.build();
						ConstantTracker constantTracker = new ConstantTracker(0.9F);
						CyclicalTracker exploreCyclic =
								CyclicalTracker.builder()
										.optBaseValue(0.1F)
										.optMaxValue(0.9F)
										.build();
						Tracker tracker = exploreCyclic;
						agent = new EpsilonGreedy(agent, tracker);

						float validationWinRate = 0;
						float trainWinRate = 0;
							for (int i = 0; i < epoch; i++) {
								int trainingWins = 0;
								try (ProgressBar bar = new ProgressBar("Epoch " + i, gamesPerEpoch)) {
									for (int j = 0; j < gamesPerEpoch; j++) {
										float result = game.runEnvironment(agent, true);
										Step[] batchSteps = game.getBatch();
										agent.trainBatch(batchSteps);
										trainer.step();
										bar.step();
										// Record if the game was won
										if (result > 0) {
											trainingWins++;
										}
										constantTracker.subtract(0.0025f);
										moonLander.fireInformation("Epsilon", tracker.getNewValue(0));
										moonLander.fireInformation("Action", ((EpsilonGreedy) agent).getMap());
										//System.err.println("epsilon: " + exploreCyclic.getNewValue(0));
										//manager.debugDump(2);
									}
									save(model);
								}
								trainWinRate = (float)trainingWins / gamesPerEpoch;
								logger.info("Training wins: {}", trainWinRate);

								trainer.notifyListeners(listener -> listener.onEpoch(trainer));

								// Counts win rate after playing {validationGamesPerEpoch} games
								int validationWins = 0;
								for (int j = 0; j < validationGamesPerEpoch; j++) {
									float result = game.runEnvironment(agent, false);
									if (result > 0) {
										validationWins++;
									}
								}

								validationWinRate = (float)validationWins / validationGamesPerEpoch;
								logger.info("Validation wins: {}", validationWinRate);
							}

						trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));

						TrainingResult trainingResult = trainer.getTrainingResult();
						trainingResult.getEvaluations().put("validate_winRate", validationWinRate);
						trainingResult.getEvaluations().put("train_winRate", trainWinRate);
						save(model);
						return trainingResult;
					}
				}
			}
		}
	}


	private static void save(Model model) throws IOException {
		model.save(Paths.get("build/model"), "MoonLander");
		System.err.println("model saved: " + Paths.get("build/model"));
	}

	private static class ConstantTracker implements Tracker {
		private float baseValue = 0.9F;

		public ConstantTracker() {
		}

		public ConstantTracker(float baseValue) {
			this.baseValue = baseValue;
		}

		public float getNewValue(int numUpdate) {
			return baseValue;
		}

		public void setValue(float value) {
			this.baseValue = value;
		}

		public void subtract(float v) {
			baseValue = Math.max(0.1F, baseValue - v);
		}
	}
}