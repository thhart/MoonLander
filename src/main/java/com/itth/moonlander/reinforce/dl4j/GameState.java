package com.itth.moonlander.reinforce.dl4j;

import java.util.Arrays;

/**
 * Class representing current game state.
 *
 * @author mirza
 */
public class GameState {
    private Double[] states;

    public GameState(final Double... states) {
        this.states = states;
    }

    public Double[] getStates() {
        return states;
    }

    @Override
    public String toString() {
        return "GameState{states=" + Arrays.toString(states) + '}';
    }

    /**
     * Builds game state string based on current values.
     * @return Returns e.g. from [false, true, false] -> 010.
     */
    public String getGameStateString() {
        final StringBuilder builder = new StringBuilder();
        for (final Double state : states) {
            final double v = Math.round(state);;
            builder.append(builder.length() > 0 ? " " + v : v);
        }
        return builder.toString();
    }
}
