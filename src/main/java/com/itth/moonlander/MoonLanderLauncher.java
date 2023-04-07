package com.itth.moonlander;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.*;
import javafx.stage.Stage;

public class MoonLanderLauncher extends Application {

	public void start(Stage primaryStage) throws Exception {
		final FXMLLoader loader = new FXMLLoader(MoonLanderLauncher.class.getResource("MoonLander.fxml"));
		final Parent load = loader.load();
		Scene scene = new Scene(load, 1000, 1000, true, SceneAntialiasing.BALANCED);
		primaryStage.setScene(scene);
		primaryStage.setWidth(1000);
		primaryStage.setHeight(1000);
		primaryStage.show();
	}

	public static void main(String[] args) {
		launch(args);
	}

}
