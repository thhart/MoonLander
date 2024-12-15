package com.itth.breakout;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.itth.os.realtimechart.RealTimeChart;
import com.itth.os.realtimechart.RealTimeChart.RealTimeEvent;

import java.time.Duration;

public class BreakoutEnv implements RlEnv {
  protected final NDManager manager;
  private final ActionSpace actionSpace;
  private final Breakout breakout;
  private final ReplayBuffer replayBuffer;
  private State state;

  public BreakoutEnv(Breakout breakout, BaseNDManager manager, final int batchSize, final int bufferSize) {
    this.breakout = breakout;
    this.manager = manager;
    replayBuffer = new LruReplayBuffer(batchSize, bufferSize);
    state = State.of(breakout);
    actionSpace = new ActionSpace();
    actionSpace.add(new NDList(manager.create(0F)));
    actionSpace.add(new NDList(manager.create(1F)));
    actionSpace.add(new NDList(manager.create(2F)));
  }

  public void reset() {
    breakout.reset();
    state = State.of(breakout);
  }

  public NDList getObservation() {
    return state.createObservation(manager);
  }

  public ActionSpace getActionSpace() {
    return actionSpace;
  }

  public Step step(NDList action, boolean training) {
    int move = (int) action.singletonOrThrow().getFloat();
    State preState = state;

    breakout.input(move);
    breakout.step();

    state = State.of(breakout);
    state.turn = -preState.turn;

    BreakoutStep step = new BreakoutStep(manager.newSubManager(), preState, state, action, actionSpace);
    if (training) {
      replayBuffer.addStep(step);
    }
    return step;
  }

  public Step[] getBatch() {
    return replayBuffer.getBatch();
  }

  public void close() {
    manager.close();
  }

  static final class BreakoutStep implements Step {
    private final NDList action;
    private final ActionSpace actionSpace;
    private final NDManager manager;
    private final State postState;
    private final State preState;
    private final NDArray reward;

    private BreakoutStep(NDManager manager, State preState, State postState, NDList action, ActionSpace actionSpace) {
      this.manager = manager;
      this.postState = postState;
      this.action = action;
      this.actionSpace = actionSpace;
      this.preState = preState;
      reward = manager.create(postState.getReward(action));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NDList getPreObservation() {
      return preState.createObservation(manager);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NDList getAction() {
      return action;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NDList getPostObservation() {
      return postState.createObservation(manager);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ActionSpace getPostActionSpace() {
      return actionSpace;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public NDArray getReward() {

      return reward;

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isDone() {
      return postState.done;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
      // preState.close();
      // postState.close();
      reward.close();
      // manager.close();

    }
  }


  /**
   * A helper to manage the state of the game at a moment in time.
   */
  private static final class State {

    private final double ballX;
    private final boolean done;
    private final double paddleW;
    private final double paddleX;
    int turn;
    private volatile NDList observation = null;

    private State(double paddleX, double paddleW, double ballX, int turn, boolean done) {
      this.paddleX = paddleX;
      this.paddleW = paddleW;
      this.ballX = ballX;
      this.turn = turn;
      this.done = done;
    }


    private static State of(Breakout moonLander) {
      return of(moonLander.paddle.getTranslateX() + moonLander.paddle.getWidth() / 2, moonLander.paddle.getWidth(), moonLander.ball.getTranslateX(), 1, moonLander.isDone());
    }

    private static State of(double paddleX, double paddleW, double ballX, int turn, boolean done) {
      return new State(paddleX, paddleW, ballX, turn, done);
    }

    private void close() {
      if (observation != null) {
        observation.close();
      }
    }

    private NDList createObservation(NDManager manager) {
      if (observation == null) {
        int i = (int) (paddleX - ballX);
        if (i < 0) {
          i = -1;
        } else if (i > 0) {
          i = 1;
        }
        observation = new NDList(manager.create(new float[]{i}), manager.create((float) turn));
        // observation = new NDList(manager.create(new float[]{i}));
      }
      return observation;
    }

    public float getReward(NDList action) {
      double reward;
      if (Math.abs(paddleX - ballX) < paddleW / 2) {
        reward = (paddleW - Math.abs(paddleX - ballX))/10;
      } else {
        reward = -Math.abs(paddleX - ballX);
        //	reward = 0;
      }
      RealTimeChart.send(RealTimeEvent.of("Reward", Duration.ofMillis((long) reward)));
      return (float) reward;
    }
  }

}
