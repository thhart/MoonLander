package com.itth.moonlander.reinforce.dl4j;

import java.io.*;
import java.time.Duration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.*;

import com.itth.moonlander.MoonLander;
import com.itth.moonlander.reinforce.network.*;
import com.itth.os.realtimechart.RealTimeChart;
import com.itth.os.realtimechart.RealTimeChart.RealTimeEvent;

/**
 Helper class used to ease out network training.

 @author mirza */
public final class NetworkTrainingHelper {
	// region Member
	private static final Logger LOG = LoggerFactory.getLogger(NetworkTrainingHelper.class);
	private static final int NUMBER_OF_GAMES = 5_000;
	private static final int STUCK_SCORE = -1000; // Score which indicates that the player is stuck (running in a loop)
	// endregion

	// region Constructor
	private NetworkTrainingHelper() {}
	// endregion

	// region Implementation
	public static void startTraining(final MoonLander game) {
		final long startTime = System.currentTimeMillis();
		LOG.info("Starting new training session with '{}' games", NUMBER_OF_GAMES);

		final Thread train = new Thread(() -> {
			final MultiLayerNetwork network = NetworkUtil.getNetwork();
			network.init();
			double epsilon = 0.9;
			long counter = 0;
			double heightLowest = 1024;
			for (int i = 1; i <= NUMBER_OF_GAMES; i++) {
				LOG.debug("Starting game session number '{}'", i);
				// Prepare the game world
				game.reset();

				// Get current game state
				GameState state = game.getGameState();
				double gameSessionScore = 0;
				double totalScore = 0;

				while (!game.isLanded()) {

					// Select action based on current state
					final Action action = NetworkUtil.epsilonGreedyAction(state, network, epsilon, game);
					// Decrease epsilon value
					epsilon = Math.max(0.2, epsilon - 0.0001);
					counter++;
					// Change direction based on selected action
					game.input(action);


					// Move the player
					game.step();


					// Get score for selected action
					final double score = GameStateHelper.calculateScore(game);

					// Get next (current) state
					final GameState nextState = game.getGameState();

					// Update network
					NetworkUtil.update(state, action, score, nextState, network, game);
					totalScore += score;

					if (game.getTimeElapsedInSeconds() > 60
							|| game.getVelocityVerticalInKmH() < 0
							|| game.getVelocityVerticalInKmH() > game.calculateVelocityCriticalInKmH()
					) {
						LOG.error("game too long or lost: {}, {}", score, epsilon);
						epsilon = Math.min(1, epsilon + 0.001);
						break;
					}

					// Apply next state
					state = nextState;

					// Increment score
					gameSessionScore = Math.max(score, gameSessionScore);
				}
				RealTimeChart.send(RealTimeEvent.of("score", Duration.ofMillis((long)(totalScore/counter))));
				final double heightLander = game.calculateLanderHeight();
				LOG.debug("Total score for session '{}' is :'{}' with lowest height: '{}'",
						i,
						gameSessionScore,
						heightLander
				);

				if (heightLander < heightLowest) {
					heightLowest = heightLander;
					LOG.info("Current lowest height equals : '{}' at game session : '{}'", heightLowest, i);
				}
			}

			LOG.info("All game sessions are over in '{}' ms, lowest height was '{}'",
					System.currentTimeMillis() - startTime,
					heightLowest);
			saveNetwork(network);
		}
		);

		train.start();
	}
	// endregion

	// region Helper
	private static void saveNetwork(final MultiLayerNetwork network) {
		LOG.debug("Saving trained network");
		try {
			network.save(new File(NetworkUtil.NETWORK_NAME));
		} catch (IOException e) {
			LOG.error("Failed to save network: '{}'", e.getMessage(), e);
		}
	}
	// endregion
}
