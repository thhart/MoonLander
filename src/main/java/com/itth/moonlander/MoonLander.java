package com.itth.moonlander;

import com.codahale.metrics.Timer;
import com.itth.moonlander.reinforce.dl4j.GameState;
import com.itth.moonlander.reinforce.dl4j.GameStateHelper;
import com.itth.moonlander.reinforce.network.Action;
import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.geometry.Orientation;
import javafx.geometry.Point2D;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.ToggleButton;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Line;
import javafx.scene.shape.Rectangle;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.Duration;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;

public class MoonLander {
  public static final double FUEL_START = 4000.0;
  public static final Random RANDOM = new Random();
  protected final static Logger logger = LogManager.getLogger(MoonLander.class);
  private static final double AIR_DENSITY = 1.225; // kg/m^3
  private static final double DRAG_COEFFICIENT = 1.5;
  private static final double GRAVITY = 9.81 * 1000; // mm/ms^2
  private static final double LANDING_VELOCITY_THRESHOLD = 10.0; // km/h
  private static final double PIXELS_PER_MM = 10.0 / 1000;
  private static final double THRUST_MAX = GRAVITY * 1.1; // mm/ms^2
  private static final double THRUST_MIN = GRAVITY * 0.9; // mm/ms^2
  // private static final double THRUST_MIN = 0; // mm/ms^2
  final private Map<Long, LineMeter> lineMap = new ConcurrentHashMap<>();
  private final Map<String, Label> map = new ConcurrentHashMap<>();
  private final Random random = new Random();
  public ToggleButton buttonHuman;
  public ToggleButton buttonPlay;
  public ToggleButton buttonTrain;
  public Pane pane;
  private boolean crashed = false;
  private double fuel = FUEL_START; // in liters
  private Label fuelLabel;
  private Rectangle ground;
  private Label hLabel;
  private Label labelVCritical;
  private Label labelVelocity;
  private boolean landed = false;
  private javafx.scene.shape.Rectangle lander;
  private Label score;
  private Label thrustLabel;
  private Point2D thrustValue = new Point2D(0, 0);
  private Label timeLabel;
  private long timeLast = 0;
  private long timeStart = 0;
  private AnimationTimer timer;
  private VBox vBox;
  private Point2D velocity = new Point2D(0, 0);

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
    vBox = new VBox();
    vBox.setLayoutX(10);
    fuelLabel = new Label();
    vBox.getChildren().addAll(new Label("SPACE: Reset, UP: Thrust Raise, DOWN: Thrust Lower"), new Separator(Orientation.VERTICAL), labelVelocity, labelVCritical, thrustLabel, timeLabel, hLabel, score, fuelLabel, new Separator(Orientation.VERTICAL));
    pane.getChildren().addAll(lander, ground, heightLine, vBox, heightLabel);

    // Handle keyboard input
    pane.getScene().addEventFilter(KeyEvent.KEY_PRESSED, event -> {
      if (event.getCode() == KeyCode.UP) {
        thrustIncrease();
      }
      if (event.getCode() == KeyCode.DOWN) {
        thrustDecrease();
      }
      if (event.getCode() == KeyCode.SPACE) {
        reset();
        onHuman(null);
      }
    });

  }

  private void thrustIncrease() {
    thrustValue = validateThrust(new Point2D(0, THRUST_MAX));
  }

  private void thrustDecrease() {
    thrustValue = validateThrust(new Point2D(0, THRUST_MIN));
  }

  public void reset() {
    resetZero();
    // resetRandom();
  }

  public void onHuman(ActionEvent ignoredEvent) {
    if (buttonHuman.isSelected()) {
      reset();
      final long started = System.nanoTime();
      timer = new AnimationTimer() {
        @Override
        public void handle(long now) {
          update((now - started));
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

  public Point2D validateThrust(Point2D thrustValue) {
    final Point2D d = new Point2D(0, fuel > 0 ? Math.min(THRUST_MAX, Math.max(THRUST_MIN, thrustValue.getY())) : 0);
    return d;
  }

  public void resetZero() {
    lander.setY(((int) (0)));
    velocity = new Point2D(0, 0);
    // thrustValue = new Point2D(0, GRAVITY);
    thrustValue = new Point2D(0, THRUST_MIN);
    timeStart = 0;
    timeLast = 0;
    crashed = false;
    landed = false;
    fuel = FUEL_START;
    lander.setFill(Color.DARKGRAY);
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
    // double speed = velocity.magnitude();
    // double airResistanceMagnitude = (1 / 2F) * AIR_DENSITY * speed * speed * DRAG_COEFFICIENT * lander.getWidth() * lander.getHeight();
    // Point2D airResistance = velocity.normalize().multiply(-airResistanceMagnitude);
    // velocity = velocity.add(airResistance.multiply(timeDeltaNs));
    updateFuel(timeDeltaNs);

    // Update position
    double dx = velocity.getX() * timeDeltaNs * 1000 * PIXELS_PER_MM;
    double dy = velocity.getY() * timeDeltaNs * 1000 * PIXELS_PER_MM;
    lander.setX(lander.getX() + dx);
    lander.setY(Math.max(0, lander.getY() + dy));
    if (lander.getBoundsInLocal().getMinY() <= 0) velocity = new Point2D(velocity.getX(), 0);
    labelVelocity.setText(String.format("Velocity: %.2f km/h", getVelocityVerticalInKmH()));
    labelVCritical.setText(String.format("Critical: %.2f km/h", calculateVelocityCriticalInKmH(calculateLanderHeight())));
    fuelLabel.setText(String.format("Fuel: %.2f l", fuel));
    thrustLabel.setText(String.format("Thrust Up: %.2f m/s²", thrustValue.getY() / 1000D));
    double elapsedTime = (now - timeStart) / 1_000_000_000.0;
    timeLabel.setText(String.format("Time: %.2f s", elapsedTime));
    hLabel.setText(String.format("Height: %.2f m", calculateLanderHeight()));
    score.setText(String.format("Score: %.2f", 0F));
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

  private void updateFuel(double timeDeltaNs) {
    double fuelConsumptionRate = Math.abs(thrustValue.getY()) / GRAVITY;
    double fuelConsumed = fuelConsumptionRate * (timeDeltaNs / 10);
    fuel = Math.max(0, fuel - fuelConsumed);
  }

  public double getVelocityVerticalInKmH() {
    return velocity.getY() * 3600;
  }

  public double calculateVelocityCriticalInKmH(double heightInM) {
    final double v = Math.sqrt(2 * THRUST_MAX / 1000 * heightInM) * 3.6;
    return Double.isNaN(v) ? 0 : v;
  }

  public double calculateLanderHeight() {
    return ((ground.getBoundsInLocal().getMinY() - lander.getBoundsInLocal().getMaxY())) / PIXELS_PER_MM / 1000;
  }

  public static void main(String[] args) {
    Application.launch(args);
  }

  public void fire(Action action) {
    Platform.runLater(() -> {
          try {
            final LineMeter lineMeter = lineMap.computeIfAbsent((long) lander.getY(), integer -> {
              final Line line = new Line(lander.getX() + 40, (long) lander.getY(), lander.getX() + 60, (long) lander.getY());
              pane.getChildren().add(line);
              return LineMeter.of(line, new Timer());
            });
            lineMeter.meter.update(Duration.ofMillis(action == Action.THRUST_UP ? 1000 : action == Action.THRUST_DOWN ? 0 : 500));
            lineMeter.line.setStroke(colorGradient((int) lineMeter.meter.getMeanRate(), 0, 1000, Color.GREEN, Color.RED));
          } catch (Exception e) {
            logger.error(e, e);
          }
        }
    );
  }

  public static javafx.scene.paint.Color colorGradient(int value, int minValue, int maxValue, final javafx.scene.paint.Color c1, final javafx.scene.paint.Color c2) {
    double factor = Math.min(Math.max((value - minValue) / (double) (maxValue - minValue), 0), 1);
    return c1.interpolate(c2, factor);
  }

  public void fireInformation(String information, Object value) {
    Platform.runLater(() -> {
          try {
            map.computeIfAbsent(information, s -> {
              final Label label = new Label();
              vBox.getChildren().add(label);
              return label;
            }).setText(information + ": " + value);
          } catch (Exception e) {
            logger.error(e, e);
          }
        }
    );
  }

  public double getFuel() {
    return fuel;
  }

  public GameState getGameState() {
    return GameStateHelper.createGameState(
        Duration.ofNanos(timeLast - timeStart), getVelocityVerticalInKmH(), thrustValue,
        calculateVelocityCriticalInKmH(),
        calculateLanderHeight(), isLanded(), isCrashed());
  }

  public double calculateVelocityCriticalInKmH() {
    return calculateVelocityCriticalInKmH(calculateLanderHeight());
  }

  public boolean isLanded() {
    return landed;
  }

  public boolean isCrashed() {
    return crashed;
  }

  public double getThrustVertical() {
    return thrustValue.getY();
  }

  public double getTimeElapsedInSeconds() {
    return (timeLast - timeStart) / 1000_000_000D;
  }

  public void input(int action) {
    switch (action) {
      case 0 -> input(Action.NOTHING);
      case 1 -> input(Action.THRUST_UP);
      case 2 -> input(Action.THRUST_DOWN);
    }
  }

  public void input(Action action) {
    switch (action) {
      case THRUST_UP -> thrustIncrease();
      case THRUST_DOWN -> thrustDecrease();
      case NOTHING -> thrustEqual();
    }
  }

  private void thrustEqual() {
    thrustValue = validateThrust(new Point2D(0, GRAVITY));
  }

  public void onPlay(ActionEvent event) {
    // NetworkEvaluationHelper.startEvaluating(this);;
  }

  public void onTrain(ActionEvent ignoredEvent) {
    new Thread(() -> {
      try {
        MoonLanderTrainer.runExample(MoonLander.this);
      } catch (Throwable e) {
        logger.error(e, e);
      }
    }).start();
    // NetworkTrainingHelper.startTraining(this);
  }

  public void resetRandom() {
    lander.setY(pane.getHeight() - 100);
    // lander.setY(random.nextInt((int)(pane.getHeight() - 50)));
    velocity = new Point2D(0, random.nextDouble(calculateVelocityCriticalInKmH() / 3600));
    thrustValue = new Point2D(0, GRAVITY);
    // thrustValue = new Point2D(0, THRUST_MIN);
    timeStart = 0;
    timeLast = 0;
    crashed = false;
    landed = false;
    fuel = FUEL_START;
    lander.setFill(Color.DARKGRAY);
  }

  public void step() {
    final CountDownLatch latch = new CountDownLatch(1);
    Platform.runLater(() -> {
          try {
            update(timeLast + 100_000_000);
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
}