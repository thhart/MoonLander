package com.itth.moonlander;

import java.time.Duration;
import java.util.*;
import ai.djl.modality.rl.*;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.*;

import com.itth.os.realtimechart.RealTimeChart;
import com.itth.os.realtimechart.RealTimeChart.RealTimeEvent;

public class MoonLanderEnv implements RlEnv {
	protected final NDManager manager;
	private final ActionSpace actionSpace;
	private final MoonLander moonLander;
	private final ReplayBuffer replayBuffer;
	private State state;

	public MoonLanderEnv(MoonLander moonLander, BaseNDManager manager, final int batchSize, final int bufferSize) {
		this.moonLander = moonLander;
		this.manager = manager;
		replayBuffer = new LruReplayBuffer(batchSize, bufferSize);
		state = State.of(moonLander);
		actionSpace = new ActionSpace();
		actionSpace.add(new NDList(manager.create(0F)));
		actionSpace.add(new NDList(manager.create(1F)));
		actionSpace.add(new NDList(manager.create(2F)));

	}

	public void close() {
		manager.close();
	}

	public ActionSpace getActionSpace() {
		return actionSpace;
	}

	public Step[] getBatch() {
		return replayBuffer.getBatch();
	}

	public NDList getObservation() {
		return state.createObservation(manager);
	}

	public void reset() {
		moonLander.reset();
		heightReached.clear();
		state = State.of(moonLander);
	}

	public Step step(NDList action, boolean training) {
		int move = (int)action.singletonOrThrow().getFloat();
		State preState = state;

		moonLander.input(move);
		moonLander.step();

		state = State.of(moonLander);
		state.turn = -preState.turn;

		MoonLanderStep step = new MoonLanderStep(manager.newSubManager(), preState, state, action, actionSpace);
		if (training) {
			replayBuffer.addStep(step);
		}
		return step;
	}

	static final class MoonLanderStep implements RlEnv.Step {
		private final NDList action;
		private final ActionSpace actionSpace;
		private final NDManager manager;
		private final State postState;
		private final State preState;
		private final NDArray reward;

		private MoonLanderStep(NDManager manager, State preState, State postState, NDList action, ActionSpace actionSpace) {
			this.manager = manager;
			this.postState = postState;
			this.action = action;
			this.actionSpace = actionSpace;
			this.preState = preState;
			reward = manager.create(postState.getReward(action));
		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public void close() {
			//preState.close();
			//postState.close();
			reward.close();
			//manager.close();

		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public NDList getAction() {
			return action;
		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public ActionSpace getPostActionSpace() {
			return actionSpace;
		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public NDList getPostObservation() {
			return postState.createObservation(manager);
		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public NDList getPreObservation() {
			return preState.createObservation(manager);
		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public NDArray getReward() {

			return reward;

		}

		/**
		 {@inheritDoc}
		 */
		@Override
		public boolean isDone() {
			return postState.isLanded()
					//|| postState.seconds > 1 && postState.velocityVertical <= 0 && postState.height >= 90
					|| postState.fuel <= 0
			//		|| postState.height > 12
			|| postState.velocityCritical < postState.velocityVertical

					;
		}
	}

	private static Set<Integer> heightReached = new TreeSet<>();

	/**
	 A helper to manage the state of the game at a moment in time.
	 */
	private static final class State {

		private final double velocityCritical;
		private final double height;
		private final double fuel;
		private final boolean landed;
		private final boolean crashed;
		private final double seconds;
		private final double thrustVertical;
		private final double velocityVertical;
		int turn;
		private volatile NDList observation = null;

		private State(double height, double velocityVertical, double velocityCritical, double thrustVertical, double fuel, boolean landed, int turn, boolean crashed, double seconds) {
			this.height = height;
			this.velocityVertical = velocityVertical;
			this.velocityCritical = velocityCritical;
			this.thrustVertical = thrustVertical;
			this.fuel = fuel;
			this.landed = landed;
			this.turn = turn;
			this.crashed = crashed;
			this.seconds = seconds;
		}


		private static State of(MoonLander moonLander) {
			return of(moonLander.calculateLanderHeight(),
					moonLander.getVelocityVerticalInKmH(),
					moonLander.calculateVelocityCriticalInKmH(),
					moonLander.getThrustVertical(), moonLander.getFuel(), 1,
					moonLander.isLanded(), moonLander.isCrashed(), moonLander.getTimeElapsedInSeconds()
			);
		}

		private void close() {
			if (observation != null) {
				observation.close();
			}
		}

		@SuppressWarnings("SameParameterValue")
		private static State of(double height, double velocityVertical, double velocityCritical, double thrustVertical, double fuel, int turn, final boolean landed, boolean crashed, double seconds) {
			return new State(height, velocityVertical, velocityCritical, thrustVertical, fuel, landed, turn, crashed, seconds);
		}


		private NDList createObservation(NDManager manager) {
			if (observation == null) {
				observation = new NDList(manager.create(new float[]{(float)height, (float)velocityVertical, (float)velocityCritical, (float)thrustVertical, (float) fuel}), manager.create((float)turn));
			}
			return observation;
		}

		public float getReward(NDList action) {
			final float reward = getReward0004(action);
			RealTimeChart.send(RealTimeEvent.of("Reward", Duration.ofMillis((long)reward)));
			return reward;
		}


		public float getReward0004(NDList action) {
			double reward = 0;
			//if(! heightReached.contains((int)height)) {
				heightReached.add((int)height);
				reward = 100 - height; // / Math.max(1, seconds);
			//}
			if(velocityVertical <= 0) {
				reward = -1;
			}
			if(velocityCritical < velocityVertical) {
				reward = -10;
			}
			if(fuel == 0) {
				reward = -10;
			}

			if(isLanded()) {
				reward = isCrashed() ? -100 - velocityVertical: 100 - seconds/10;
			}
			return (float) reward;
		}

		public float getReward0003(NDList action) {
			double reward;
			if (velocityVertical <= 0) {
				reward = height;
			} else if (velocityCritical - velocityVertical < velocityCritical * 0.5) {
				reward = 2;
			} else if (velocityCritical > velocityVertical) {
				reward = 1/height;
			} else {
				reward = 0;
			}
			if(fuel == 0) {
				reward = -1000;
			}

			if(isLanded()) {
				reward = isCrashed() ? -1000 - velocityVertical: 1000 - seconds/10;
			}
			return (float) reward;
		}

		public float getReward0002(NDList action) {
			double reward;
			if (velocityVertical <= 0) {
				reward = -100;
			} else if (velocityCritical < velocityVertical) {
				reward = velocityCritical - velocityVertical;
			} else if (velocityCritical - velocityVertical < velocityCritical * 0.5) {
				reward = 1/height;
			} else {
				reward = 0;
			}
			if(fuel == 0) {
				reward = -100;
			}
			if(isLanded()) {
				reward = isCrashed() ? -500 - velocityVertical: 500 - seconds;
			}
			return (float) reward;
		}

		//public float getReward(NDList action) {
		//	double reward;
		//	if (velocityVertical <= 0) {
		//		if(action.singletonOrThrow().getFloat() == 1) {
		//			reward = 0;
		//		} else {
		//			reward = (float)(- thrustVertical - Math.pow(velocityVertical, 2) + (action.singletonOrThrow().getFloat() == 1 ? 24 : 0));
		//		}
		//	} else if (velocityCritical < velocityVertical) {
		//		reward = (velocityCritical - velocityVertical) - Math.pow(seconds, 2);
		//	} else {
		//		reward = (400 - (velocityCritical - velocityVertical) - height - Math.pow(seconds, 2));
		//	}
		//	if(fuel == 0) {
		//		reward = -2000;
		//	}
		//	if(isLanded()) {
		//		reward += isCrashed() ? -2000 : 2000;
		//	}
		//	return (float)reward;
		//}

		//public float getReward(NDList action) {
		//	float reward;
		//	if (velocityVertical <= 0 || height > 80) {
		//		reward =  (float)(- thrustVertical + (action.singletonOrThrow().getFloat() == 1 ? 4000 : 0));
		//	} else if (isLanded()) {
		//		if (isCrashed()) {
		//			reward =  (float)(-500 - height - velocityVertical);
		//		} else {
		//			reward = 10000 + (float)(1 / seconds);
		//		}
		//	} else if (velocityVertical > velocityCritical) {
		//		reward = (float)(-700 - Math.pow(velocityVertical - velocityCritical, 2));
		//	} else {
		//		reward = 1000 - (float)(velocityCritical - velocityVertical);
		//	}
		//	//return (float)(-750 - Math.abs(velocityCritical - velocityVertical));
		//	return reward;
		//}

		public boolean isLanded() {
			return landed;
		}

		public boolean isCrashed() {
			return crashed;
		}

		public boolean isNegative() {
			return velocityVertical < 0;
		}
	}

}
