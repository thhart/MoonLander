package com.itth.moonlander.reinforce.dl4j;

import java.util.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.primitives.Doubles;
import org.slf4j.*;

import com.itth.moonlander.MoonLander;
import com.itth.moonlander.reinforce.network.*;

/**
 Helper class used to ease out handling of networks.

 @author mirza */
public final class NetworkUtil {
	// region Members
	/**
	 Name of the network that is used when saving and loading it.
	 */
	public static final String NETWORK_NAME = "trained_network.zip";

	private static final Logger LOG = LoggerFactory.getLogger(NetworkUtil.class);
	private static final int HIDDEN_LAYER_COUNT = 256;
	// endregion

	// region Constructor
	private NetworkUtil() {}
	// endregion

	// region Implementation

	/**
	 Get the network for training.

	 @return Returns {@link MultiLayerNetwork} used for training.
	 */
	public static MultiLayerNetwork getNetwork() {
		return new MultiLayerNetwork(getConfiguration());
	}

	/**
	 Used to get action using epsilon greedy algorithm.

	 @param state   Current state of the game.
	 @param network Network.
	 @param epsilon Epsilon value.
	 @param game
	 @return Returns calculated action.
	 */
	public static Action epsilonGreedyAction(final GameState state,
			final MultiLayerNetwork network,
			final double epsilon, MoonLander game) {
		// https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
		final double random = getRandomDouble();
		if (random < epsilon) {
			return Action.getRandomAction();
		}

		final Action actionFromTheNetwork = getActionFromTheNetwork(state, network);
		game.fire(actionFromTheNetwork);
		return actionFromTheNetwork;
	}

	/**
	 Gets the action from the network based on the current state.

	 @param state   Current state.
	 @param network Network.
	 @return Returns action outputed by the network
	 */
	public static Action getActionFromTheNetwork(final GameState state, final MultiLayerNetwork network) {
		final INDArray output = network.output(toINDArray(state), false);

        /*
        Values provided by the network. Based on them we chose the current best action.
         */
		final float[] outputValues = output.data().asFloat();

		// Find index of the highest value
		final int maxValueIndex = getMaxValueIndex(outputValues);

		final Action actionByIndex = Action.getActionByIndex(maxValueIndex);
		LOG.debug("For values '{}' index of highest value is '{}' and action is '{}'",
				outputValues,
				maxValueIndex,
				actionByIndex
		);

		return actionByIndex;
	}

	/**
	 Update network and q-table with new values.

	 @param state     Current game state.
	 @param action    Taken action.
	 @param score     Achieved score.
	 @param nextState Next game state.
	 @param network   Network.
	 @param game       */
	public static void update(final GameState state,
			final Action action,
			final double score,
			final GameState nextState,
			final MultiLayerNetwork network, MoonLander game) {
		// Get max q score for next state

		// Update network
		final INDArray stateObservation = toINDArray(state);
		final INDArray output = network.output(stateObservation);
		final INDArray updatedOutput = output.putScalar(action.getActionIndex(), score);

		network.fit(stateObservation, updatedOutput);
	}

	/**
	 Puts the thread to sleep for certain amount of time.

	 @param millis Time to sleep.
	 */
	public static void wait(final int millis) {
		try {
			Thread.sleep(millis);
		} catch (final InterruptedException e) {
			LOG.error(e.getMessage(), e);
			Thread.currentThread().interrupt();
		}
	}
	// endregion

	// region Helper
	private static MultiLayerConfiguration getConfiguration() {
		final int numInputs = GameStateHelper.getNumberOfInputs();
		final int numOutputs = Action.values().length;
		return new NeuralNetConfiguration.Builder()
				.seed(12345)    //Random number generator seed for improved repeatability
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(0.001))
				.l2(0.001) // l2 regularization on all layers
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numInputs) // Number of inputs
						.nOut(HIDDEN_LAYER_COUNT)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build())
				.layer(1, new DenseLayer.Builder()
						.nIn(HIDDEN_LAYER_COUNT)
						.nOut(HIDDEN_LAYER_COUNT)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
						.nIn(HIDDEN_LAYER_COUNT)
						.nOut(numOutputs) // Number of possible actions
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.IDENTITY)
						.weightInit(WeightInit.XAVIER)
						.build())
				.backpropType(BackpropType.Standard)
				.build();
	}

	private static INDArray toINDArray(final GameState gameState) {
		return Nd4j.create(new double[][]{Doubles.toArray(Arrays.asList(gameState.getStates()))});
	}

	private static double getRandomDouble() {
		return (Math.random() * ((double)1 + 1 - (double)0)) + (double)0;
	}

	private static int getMaxValueIndex(final float[] values) {
		int maxAt = 0;

		for (int i = 0; i < values.length; i++) {
			maxAt = values[i] > values[maxAt] ? i : maxAt;
		}

		return maxAt;
	}


	private static String getStateWithActionString(final String stateString, final Action action) {
		return stateString + '-' + action;
	}
	// endregion
}
