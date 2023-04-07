package com.itth.moonlander.reinforce.dl4j;

import java.io.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.*;

import com.itth.moonlander.MoonLander;
import com.itth.moonlander.reinforce.network.*;

/**
 * Helper class used to ease out network evaluation.
 *
 * @author mirza
 */
public class NetworkEvaluationHelper {
    // region Member
    private static final Logger LOG = LoggerFactory.getLogger(NetworkEvaluationHelper.class);
    private static final int NUMBER_OF_GAMES = 100;
    // endregion

    // region Constructor
    private NetworkEvaluationHelper() {}
    // endregion

    // region Implementation
    public static void startEvaluating(final MoonLander game) {
        LOG.info("Starting evaluation of trained network");

        final Thread evaluate = new Thread(() -> {
            final MultiLayerNetwork network = loadNetwork();
            double highscore = 0;
            for (int i = 1; i <= NUMBER_OF_GAMES; i++) {
                game.reset();

                double score = 1024;
                GameState gameState = game.getGameState();
                long timeStarted = System.nanoTime();
                while (!game.isLanded()) {
                    // Get action from the network
                    final Action action = NetworkUtil.getActionFromTheNetwork(gameState, network);

                    // Change direction based on outputted action
                    game.input(action);

                    // Move the player
                    game.step(System.nanoTime() - timeStarted);

                    // Get next (current) state
                    gameState = game.getGameState();

                    // Get current score
                    score = game.getTimeElapsedInSeconds();

                    // Wait so that the user can see what exactly the snake is doing (remove it if you want full speed)
                    NetworkUtil.wait(20);
                }

                LOG.info("Session '{}' ended with score of '{}'", i, score);

                if (score < highscore) {
                    highscore = score;
                }
            }

            LOG.info("Highscore achieved by network is '{}'", highscore);
        });

        evaluate.start();
    }
    // endregion

    // region Helper
    private static MultiLayerNetwork loadNetwork() {
        try {
            return MultiLayerNetwork.load(new File(NetworkUtil.NETWORK_NAME), true);
        } catch (IOException e) {
            LOG.error("Failed to load network: '{}'", e.getMessage(), e);
        }

        return NetworkUtil.getNetwork();
    }
    // endregion
}
