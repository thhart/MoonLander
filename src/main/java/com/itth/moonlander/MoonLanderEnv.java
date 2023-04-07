package com.itth.moonlander;

import ai.djl.modality.rl.*;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.*;

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
			manager.close();

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
			return postState.isLanded() || postState.seconds > 16 && postState.height >= 90;
		}
	}


	/**
	 A helper to manage the state of the game at a moment in time.
	 */
	private static final class State {

		private final double velocityCritical;
		private final double height;
		private final boolean landed;
		private final boolean crashed;
		private final double seconds;
		private final double thrustVertical;
		private final double velocityVertical;
		int turn;
		private volatile NDList observation = null;

		private State(double height, double velocityVertical, double velocityCritical, double thrustVertical, boolean landed, int turn, boolean crashed, double seconds) {
			this.height = height;
			this.velocityVertical = velocityVertical;
			this.velocityCritical = velocityCritical;
			this.thrustVertical = thrustVertical;
			this.landed = landed;
			this.turn = turn;
			this.crashed = crashed;
			this.seconds = seconds;
		}

		private static State of(MoonLander moonLander) {
			return of(moonLander.calculateLanderHeight(),
					moonLander.getVelocityVerticalInKmH(),
					moonLander.calculateVelocityCriticalInKmH(),
					moonLander.getThrustVertical(),
					1, moonLander.isLanded(), moonLander.isCrashed(), moonLander.getTimeElapsedInSeconds()
			);
		}

		private void close() {
			if (observation != null) {
				observation.close();
			}
		}

		@SuppressWarnings("SameParameterValue")
		private static State of(double height, double velocityVertical, double velocityCritical, double thrustVertical, int turn, final boolean landed, boolean crashed, double seconds) {
			return new State(height, velocityVertical, velocityCritical, thrustVertical, landed, turn, crashed, seconds);
		}


		private NDList createObservation(NDManager manager) {
			if (observation == null) {
				observation = new NDList(manager.create(new float[]{(float)height, (float)velocityVertical, (float)velocityCritical, (float)thrustVertical}), manager.create((float)turn));
			}
			return observation;
		}

		public float getReward(NDList action) {
			if (velocityVertical <= 0 || height > 80) {
				return (float)(-1000 - height + velocityVertical - thrustVertical + (action.singletonOrThrow().getFloat() == 1 ? 200 : 0));
			}
			if (isLanded()) {
				if (isCrashed()) {
					return (float)(-500 - height - velocityVertical);
				}
				return 10000 + (float)(1 / seconds);
			}
			return 1000 - (float)(velocityCritical - velocityVertical);
			//return (float)(-750 - Math.abs(velocityCritical - velocityVertical));
		}

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
