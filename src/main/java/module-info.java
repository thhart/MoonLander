module com.itth.moonlander {
	requires org.apache.logging.log4j;
	requires java.desktop;
	requires org.slf4j;
	requires javafx.graphics;
	requires javafx.controls;
	requires javafx.fxml;
	requires org.apache.commons.collections4;
	requires com.itth.os.realtimechart;
	requires com.codahale.metrics;
	requires deeplearning4j.nn;
	requires nd4j.api;
	requires guava;
	requires ai.djl.api;
	requires ai.djl.model_zoo;
	requires examples;
	requires commons.cli;
	requires progressbar;
	requires ai.djl.pytorch_engine;
	exports com.itth.breakout;
	exports com.itth.moonlander;
	exports com.itth.moonlander.samples;
}