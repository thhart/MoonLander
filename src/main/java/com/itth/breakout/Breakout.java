package com.itth.breakout;

import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import javafx.animation.*;
import javafx.application.*;
import javafx.geometry.Point2D;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.*;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.util.Duration;
import org.apache.logging.log4j.*;
import org.nd4j.shade.guava.util.concurrent.RateLimiter;

public class Breakout extends Application {
	protected final static Logger logger = LogManager.getLogger(Breakout.class);

	private static final double BALL_RADIUS = 5;
	private static final double PADDLE_WIDTH = 80;
	private static final double PADDLE_HEIGHT = 10;
	private static final int SCENE_WIDTH = 400;
	private static final int SCENE_HEIGHT = 400;
	private static final int VELOCITY = 400;
	protected Circle ball;

	private Point2D ballVelocity = new Point2D(100, VELOCITY); // pixels per second
	protected Rectangle paddle;
	private boolean paddleMoveLeft = false;
	private boolean paddleMoveRight = false;
	private Timeline timeline;
	private long lastUpdateTime = System.nanoTime(); // nanoseconds
	private int score = 0;
	private final AtomicBoolean stopped = new AtomicBoolean(false);
	private final Semaphore semaphoreReset = new Semaphore(0);
	private volatile CountDownLatch latchReset;

	public void input(int move) {
		Platform.runLater(() -> {
					try {
						switch (move) {
							case 1 -> {
								if (paddle.getTranslateX() > 0) {
									paddle.setTranslateX(paddle.getTranslateX() - 1);
								}
							}
							case 2 -> {
								if (paddle.getTranslateX() < SCENE_WIDTH - PADDLE_WIDTH) {
									paddle.setTranslateX(paddle.getTranslateX() + 1);
								}
							}
						}
					} catch (Exception e) {
						logger.error(e, e);
					}
				}
		);
	}

	@Override
	public void start(Stage primaryStage) {
		Pane root = new Pane();
		Scene scene = new Scene(root, SCENE_WIDTH, SCENE_HEIGHT);

		ball = new Circle(BALL_RADIUS, Color.BLACK);
		ball.setTranslateX(SCENE_WIDTH / 2D);
		ball.setTranslateY(BALL_RADIUS);

		paddle = new Rectangle(PADDLE_WIDTH, PADDLE_HEIGHT, Color.BLUE);
		paddle.setTranslateX(SCENE_WIDTH / 2D - PADDLE_WIDTH / 2);
		paddle.setTranslateY(SCENE_HEIGHT - PADDLE_HEIGHT * 2);

		Text scoreText = new Text(10, 20, "Score: 0");
		root.getChildren().addAll(ball, paddle, scoreText);

		scene.setOnKeyPressed(e -> {
			if (e.getCode() == KeyCode.LEFT) {
				paddleMoveLeft = true;
			} else if (e.getCode() == KeyCode.RIGHT) {
				paddleMoveRight = true;
			}
		});

		scene.setOnKeyReleased(e -> {
			if (e.getCode() == KeyCode.LEFT) {
				paddleMoveLeft = false;
			} else if (e.getCode() == KeyCode.RIGHT) {
				paddleMoveRight = false;
			}
		});

		primaryStage.setScene(scene);
		primaryStage.setTitle("Breakout");
		primaryStage.show();

		timeline = new Timeline(new KeyFrame(Duration.millis(1000 / 1000D), e -> {
			long currentTime = System.nanoTime();
			double elapsedSeconds = (currentTime - lastUpdateTime) / 1e9;
			lastUpdateTime = currentTime;

			double deltaX = ballVelocity.getX() * elapsedSeconds;
			double deltaY = ballVelocity.getY() * elapsedSeconds;
			ball.setTranslateX(ball.getTranslateX() + deltaX);
			ball.setTranslateY(ball.getTranslateY() + deltaY);
			scoreText.setText("Score: " + score);

			if (paddleMoveLeft && paddle.getTranslateX() > 0) {
				paddle.setTranslateX(paddle.getTranslateX() - 1);
			}
			if (paddleMoveRight && paddle.getTranslateX() < SCENE_WIDTH - PADDLE_WIDTH) {
				paddle.setTranslateX(paddle.getTranslateX() + 1);
			}

			if (ball.getTranslateX() - BALL_RADIUS <= 0 && ballVelocity.getX() < 0) {
				ballVelocity = new Point2D(-ballVelocity.getX(), ballVelocity.getY());
			} else if (ball.getTranslateX() + BALL_RADIUS >= SCENE_WIDTH && ballVelocity.getX() > 0) {
				ballVelocity = new Point2D(-ballVelocity.getX(), ballVelocity.getY());
			}

			if (ball.getTranslateY() - BALL_RADIUS <= 0 && ballVelocity.getY() < 0) {
				ballVelocity = new Point2D(ballVelocity.getX(), -ballVelocity.getY());
			}

			if (isDone()) {
				return;
			}

			if (ball.getBoundsInParent().intersects(paddle.getBoundsInParent()) && ballVelocity.getY() > 0) {
			    // Calculate the distance between the center of the ball and the center of the paddle
			    double ballPaddleDistance = ball.getTranslateX() - (paddle.getTranslateX() + PADDLE_WIDTH / 2);

			    // Calculate the proportion of the distance from the center of the paddle
			    // This value will be between -1 (ball hits left side of paddle) and 1 (ball hits right side of paddle)
			    double proportion = ballPaddleDistance / (PADDLE_WIDTH / 2);

			    // Calculate the new horizontal velocity of the ball based on the proportion
			    double newVelocityX = proportion * VELOCITY;

			    // Update the ball velocity with the new values
			    ballVelocity = new Point2D(newVelocityX, -ballVelocity.getY());
			    score++;
			}

			//if (ball.getBoundsInParent().intersects(paddle.getBoundsInParent()) && ballVelocity.getY() > 0) {
			//	ballVelocity = new Point2D(ballVelocity.getX(), -ballVelocity.getY());
			//	score++;
			//}
		}));

		timeline.setCycleCount(Timeline.INDEFINITE);
		//timeline.play();
		new Thread(() -> {
			try {
				BreakoutTrainer.runExample(Breakout.this);
			} catch (Throwable e) {
				logger.error(e, e);
			}
		}).start();
		new Thread(() -> {
			try {
				while (true) {
					semaphoreReset.acquire();
					Platform.runLater(() -> {
								try {
									timeline.stop();
									lastUpdateTime = System.nanoTime();
									ballVelocity = new Point2D(random.nextDouble(200) - 100, VELOCITY); // reset ball velocity
									ball.setTranslateX(SCENE_WIDTH / 2);
									ball.setTranslateY(BALL_RADIUS);
									paddle.setTranslateX(SCENE_WIDTH / 2D - PADDLE_WIDTH / 2);
									paddle.setTranslateY(SCENE_HEIGHT - PADDLE_HEIGHT * 2);
									score = 0;
									stopped.set(false);
									timeline.play();
									latchReset.countDown();
									latchReset = null;
								} catch (Exception e) {
									logger.error(e, e);
								}
							}
					);
				}
			} catch (Throwable e) {
				logger.error(e, e);
			}
		}).start();

	}

	protected boolean isDone() {
		if (ball.getTranslateY() + BALL_RADIUS >= SCENE_HEIGHT) {
			stopped.set(true);
			timeline.stop();
			return true;
		}
		if (score == 10) {
			stopped.set(true);
			timeline.stop();
			return true;
		}
		return false;
	}


	private final Random random = new Random();

	protected void reset() {
		CountDownLatch myLatch;
		if (latchReset == null) {
			synchronized (this) {
				if (latchReset == null) latchReset = new CountDownLatch(1);
				myLatch = latchReset;
			}
		} else {
			myLatch = latchReset;
		}
		semaphoreReset.release();
		try {
			myLatch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	private final RateLimiter limiter = RateLimiter.create(1000);

	public void step() {
		limiter.acquire();
	}

	public void start() {
	}
}
