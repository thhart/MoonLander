package com.itth.moonlander;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.*;
import com.codahale.metrics.Timer;
import javafx.animation.AnimationTimer;
import javafx.application.*;
import javafx.event.ActionEvent;
import javafx.geometry.*;
import javafx.scene.control.*;
import javafx.scene.input.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import org.apache.logging.log4j.*;

import com.itth.moonlander.reinforce.dl4j.*;
import com.itth.moonlander.reinforce.network.*;

public class MoonLander {
	private static final double AIR_DENSITY = 1.225; // kg/m^3
	private static final double DRAG_COEFFICIENT = 1.5;
	private static final double GRAVITY = 9.81 * 1000; // mm/ms^2
	private static final double LANDING_VELOCITY_THRESHOLD = 10.0; // km/h
	private static final double PIXELS_PER_MM = 10.0 / 1000;
	private static final double THRUST_MAX = GRAVITY * 2; // mm/ms^2
	public ToggleButton buttonHuman;
	public ToggleButton buttonPlay;
	public ToggleButton buttonTrain;
	private boolean crashed = false;
	private Rectangle ground;
	private Label hLabel;
	private Label score;
	private boolean landed = false;
	private javafx.scene.shape.Rectangle lander;
	private long timeLast = 0;
	public Pane pane;
	private long timeStart = 0;
	private Label thrustLabel;
	private Point2D thrustValue = new Point2D(0, 0);
	private Label timeLabel;
	private AnimationTimer timer;
	private Point2D velocity = new Point2D(0, 0);
	private Label labelVelocity;
	private Label labelVCritical;
	protected final static Logger logger = LogManager.getLogger(MoonLander.class);

	public MoonLander() {
		Platform.runLater(() -> {
					try {
						start();
					} catch (Exception e) {
						logger.error(e, e);
					}
				}
		);
	}

	public static void main(String[] args) {
		Application.launch(args);
	}

	public double calculateVelocityCriticalInKmH() {
		return calculateVelocityCriticalInKmH(calculateLanderHeight());
	}

	public GameState getGameState() {
		return GameStateHelper.createGameState(
				Duration.ofNanos(timeLast - timeStart), getVelocityVerticalInKmH(), thrustValue,
				calculateVelocityCriticalInKmH(),
				calculateLanderHeight(), isLanded(), isCrashed());
	}

	public double getThrustVertical() {
		return thrustValue.getY();
	}

	public double getTimeElapsedInSeconds() {
		return (timeLast - timeStart) / 1000_000_000D;
	}

	public void input(Action action) {
		switch (action) {
			case THRUST_UP -> increaseThrust();
			case THRUST_DOWN -> decreaseThrust();
		}
	}

	public void input(int action) {
		switch (action) {
			case 0 -> input(Action.THRUST_UP);
			case 1 -> input(Action.THRUST_DOWN);
		}
	}

	public boolean isCrashed() {
		return crashed;
	}

	public boolean isLanded() {
		return landed;
	}

	public void onHuman(ActionEvent ignoredEvent) {
		if (buttonHuman.isSelected()) {
			reset();
			final long started = System.nanoTime();
			timer = new AnimationTimer() {
				@Override
				public void handle(long now) {
					update( (now - started));
					if (landed) {
						timer.stop();
					}
				}

			};
			timer.start();
		} else {
			if (timer != null) {
				timer.stop();
			}
		}
	}

	public void onPlay(ActionEvent event) {
		//NetworkEvaluationHelper.startEvaluating(this);;
	}

	public void onTrain(ActionEvent ignoredEvent) {
		new Thread(() -> {
			try {
				MoonLanderTrainer.runExample(MoonLander.this);
			} catch (Throwable e) {
				logger.error(e, e);
			}
		}).start();
		//NetworkTrainingHelper.startTraining(this);
	}

	public void reset() {
		velocity = new Point2D(0, 0);
		lander.setY(0);
		thrustValue = new Point2D(0, GRAVITY - 500);
		timeStart = 0;
		timeLast = 0;
		crashed = false;
		landed = false;
		lander.setFill(Color.DARKGRAY);
	}

final private Map<Long, LineMeter> lineMap = new ConcurrentHashMap<>();

	static class LineMeter {
		final Line line;
		final Timer meter;

		private LineMeter(Line line, Timer meter) {
			this.line = line;
			this.meter = meter;
		}

		static LineMeter of(Line line, Timer meter) {
			return new LineMeter(line, meter);
		}
	}


	public static javafx.scene.paint.Color colorGradient(int value, int minValue, int maxValue, final javafx.scene.paint.Color c1, final javafx.scene.paint.Color c2) {
    double factor = Math.min(Math.max((value - minValue) / (double) (maxValue - minValue), 0), 1);
    return c1.interpolate(c2, factor);
 }

	public void fire(Action action) {
		Platform.runLater(() -> {
					try {
						final LineMeter lineMeter = lineMap.computeIfAbsent((long)lander.getY(), integer -> {
							final Line line = new Line(lander.getX() + 40, (long)lander.getY(), lander.getX() + 60, (long)lander.getY());
							pane.getChildren().add(line);
							return LineMeter.of(line, new Timer());
						});
						lineMeter.meter.update(Duration.ofMillis(action == Action.THRUST_UP ? 1000 : action == Action.THRUST_DOWN ? 0 : 500));
						lineMeter.line.setStroke(colorGradient((int)lineMeter.meter.getMeanRate(), 0, 1000, Color.GREEN, Color.RED));
					} catch (Exception e) {
						logger.error(e, e);
					}
				}
		);
	}

	public void start() {
		// Create lander
		lander = new Rectangle(20, 20, Color.DARKGRAY);
		lander.setX(pane.getWidth() / 2 - lander.getWidth() / 2);
		lander.setY(0);

		// Create ground
		ground = new Rectangle(pane.getWidth(), 10_000_0000, Color.GREEN);
		ground.setX(0);
		ground.setY(pane.getHeight() - 40);

		double maxHeight = (pane.getHeight() - 40) / PIXELS_PER_MM;
		javafx.scene.shape.Line heightLine = new Line(0, 0, 0, maxHeight * PIXELS_PER_MM);
		heightLine.setStroke(Color.RED);
		heightLine.setStrokeWidth(3.0);
		heightLine.setTranslateX(pane.getWidth() - 50);

		// Add label to display height of scene
		Label heightLabel = new Label();
		heightLabel.setText(String.format("%.0f m", maxHeight));
		heightLabel.setTranslateX(pane.getWidth() - 60);
		heightLabel.setTranslateY(maxHeight * PIXELS_PER_MM / 2);
		heightLabel.setRotate(Math.toRadians(90));

		// Add label to display current velocity of lander
		labelVelocity = new Label();
		labelVCritical = new Label();
		thrustLabel = new Label();
		hLabel = new Label();
		score = new Label();
		timeLabel = new Label();
		final VBox box = new VBox();
		box.setLayoutX(10);
		box.getChildren().addAll(new Label("SPACE: Reset, UP: Thrust Raise, DOWN: Thrust Lower"), new Separator(Orientation.VERTICAL), labelVelocity, labelVCritical, thrustLabel, timeLabel, hLabel, score);
		pane.getChildren().addAll(lander, ground, heightLine, box, heightLabel);

		// Handle keyboard input
		pane.getScene().addEventFilter(KeyEvent.KEY_PRESSED, event -> {
			if (event.getCode() == KeyCode.UP) {
				increaseThrust();
			}
			if (event.getCode() == KeyCode.DOWN) {
				decreaseThrust();
			}
			if (event.getCode() == KeyCode.SPACE) {
				reset();
				onHuman(null);
			}
		});

	}

	private void decreaseThrust() {
		thrustValue = validateThrust(thrustValue.subtract(0, 100));
	}

	private void increaseThrust() {
		thrustValue = validateThrust(thrustValue.add(0, 100));
	}

	public void step() {
		final CountDownLatch latch = new CountDownLatch(1);
		Platform.runLater(() -> {
					try {
						update(timeLast + 5_000_000);
						latch.countDown();
					} catch (Exception e) {
						logger.error(e, e);
					}
				}
		);
		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	public void step(long time) {
		update(time);
	}

	private void update(long now) {
		// Calculate time since last frame
		double timeDeltaNs = (now - timeLast) / 1_000_000D;
		timeLast = now;

		// Calculate velocity due to gravity
		double accelerationGravity = GRAVITY * PIXELS_PER_MM;
		final double addG = accelerationGravity * timeDeltaNs / 1000_000;
		velocity = velocity.add(0, addG);

		// Calculate velocity due to thrust
		final double accelerationThrust = -thrustValue.getY() * PIXELS_PER_MM;
		final double addT = accelerationThrust * timeDeltaNs / 1000_000;
		velocity = velocity.add(0, addT);

		// Calculate velocity due to air resistance
		//double speed = velocity.magnitude();
		//double airResistanceMagnitude = (1 / 2F) * AIR_DENSITY * speed * speed * DRAG_COEFFICIENT * lander.getWidth() * lander.getHeight();
		//Point2D airResistance = velocity.normalize().multiply(-airResistanceMagnitude);
		//velocity = velocity.add(airResistance.multiply(timeDeltaNs));

		// Update position
		double dx = velocity.getX() * timeDeltaNs * 1000 * PIXELS_PER_MM;
		double dy = velocity.getY() * timeDeltaNs * 1000 * PIXELS_PER_MM;
		lander.setX(lander.getX() + dx);
		lander.setY(Math.max(0, lander.getY() + dy));
		if (lander.getBoundsInLocal().getMinY() <= 0) velocity = new Point2D(velocity.getX(), 0);
		labelVelocity.setText(String.format("Velocity: %.2f km/h", getVelocityVerticalInKmH()));
		labelVCritical.setText(String.format("Critical: %.2f km/h", calculateVelocityCriticalInKmH(calculateLanderHeight())));
		thrustLabel.setText(String.format("Thrust Up: %.2f m/s²", thrustValue.getY() / 1000D));
		double elapsedTime = (now - timeStart) / 1_000_000_000.0;
		timeLabel.setText(String.format("Time: %.2f s", elapsedTime));
		hLabel.setText(String.format("Height: %.2f m", calculateLanderHeight()));
		score.setText(String.format("Score: %.2f", GameStateHelper.calculateScore(this)));
		// Update time label
		// Check for collision with ground
		if (lander.getBoundsInParent().intersects(ground.getBoundsInParent())) {
			if (getVelocityVerticalInKmH() <= LANDING_VELOCITY_THRESHOLD) {
				System.out.println("Landed!");
				landed = true;
				crashed = false;
				lander.setFill(Color.GREEN);
			} else {
				System.out.println("Crashed! " + getVelocityVerticalInKmH() + " km/h");
				landed = true;
				crashed = true;
				lander.setFill(Color.RED);
			}
		}
	}

	public double getVelocityVerticalInKmH() {
		return velocity.getY() * 3600;
	}

	public double calculateVelocityCriticalInKmH(double heightInM) {
		return Math.sqrt(2 * THRUST_MAX / 1000 * heightInM) * 3.6;
	}

	public double calculateLanderHeight() {
		return ((ground.getBoundsInLocal().getMinY() - lander.getBoundsInLocal().getMaxY())) / PIXELS_PER_MM / 1000;
	}

	public Point2D validateThrust(Point2D thrustValue) {
		return new Point2D(0, Math.min(THRUST_MAX, Math.max(GRAVITY - GRAVITY/10, thrustValue.getY())));
	}
}