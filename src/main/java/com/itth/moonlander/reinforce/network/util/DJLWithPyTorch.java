package com.itth.moonlander.reinforce.network.util;

import ai.djl.engine.Engine;

public class DJLWithPyTorch {
    public static void main(String[] args) {
        // Set the default engine to PyTorch
        System.setProperty("ai.djl.default_engine", "PyTorch");
        // Get the PyTorch engine instance
        Engine engine = Engine.getInstance();
        System.out.println("Using backend engine: " + engine.getEngineName());
        System.out.println("Found GPU: " + engine.getGpuCount());
    }
}
