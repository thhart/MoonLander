package com.itth.moonlander.reinforce.dl4j;

import java.time.Duration;
import javafx.geometry.Point2D;

import com.itth.moonlander.MoonLander;

/**
 Helper class used to ease out creation of game states.

 @author mirza */
public final class GameStateHelper {
	private GameStateHelper() {}

	public static GameState createGameState(
			Duration duration, double velocityVertical, Point2D thrust, double velocityCritical, double landerHeight,
			boolean landed, boolean crashed) {
		return new GameState(
				velocityVertical, velocityCritical, landerHeight, thrust.getX(), thrust.getY()
		);
	}

	//public static GameState createGameState(
	//		Duration duration, Point2D velocity, Point2D thrust, double velocityCritical, double landerHeight,
	//		boolean landed, boolean crashed) {
	//	return new GameState(
	//			(double)duration.toSeconds(),
	//			velocity.getX() * 3.6, velocity.getY() * 3.6,
	//			velocityCritical,
	//			thrust.getX(), thrust.getY(),
	//			landerHeight,
	//			landed ? 1.0 : 0, crashed ? 1.0 : 0
	//	);
	//}

	public static double calculateScore(MoonLander game) {
		if (game.isLanded()) return 1000 - game.getTimeElapsedInSeconds();
		if (game.isCrashed()) return 0;
		if (game.getVelocityVerticalInKmH() < 0) return 1;
		//if (game.getVelocityVerticalInKmH() > game.calculateVelocityCriticalInKmH()) return 1;
		//final double landerHeight = game.calculateLanderHeight();
		//return Math.max(4, -game.getTimeElapsedInSeconds()/1000 + (90 - landerHeight));
		return 100 - game.calculateLanderHeight();
	}

	/**
	 Get number of possible states. There are 4 directions in which snake can see. Number of inputs is equal to
	 those 4 directions times how far it can see plus 8 food states.

	 @return Returns number of possible states.
	 */
	public static int getNumberOfInputs() {
		return 5;
	}
}
