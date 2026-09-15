
# Task 5 Analysis

## 5.1 Set A Normal Equation

Theta values:

                   Feature  Normal Equation    Batch GD         SGD
          theta0 intercept      6883.416640 6883.416640 6887.674452
        theta1 irradiation      8258.296494 5870.922353 7652.297928
 theta2 module temperature       -25.544992 3115.734646  819.696797
theta3 ambient temperature       -15.929397 -944.562686 -224.305477
           theta4 sin hour       -29.832570  -32.672111  -45.783702
           theta5 cos hour      -408.772537 -396.589614 -408.096033

The largest absolute weight is:

theta1 irradiation

with weight:

8258.296494

The signs should be interpreted using the physical relationship
between solar irradiation, temperature and AC power.

## 5.2 Set B vs Set A

Set A daytime RMSE:
723.4147 kW

Set B daytime RMSE:
3417.6468 kW

Difference:
2694.2320 kW

Difference as percentage of plant peak:
9.86 %

## 5.3 Solver comparison

Normal equation calculates the least-squares solution directly.

Batch gradient descent repeatedly updates all parameters using the
complete training dataset.

Stochastic gradient descent updates parameters one training example
at a time.

For this relatively small dataset, the normal equation is convenient.
For very large datasets such as 10 million rows, gradient descent is
more practical because explicitly forming and inverting X-transpose-X
can become expensive.

## 5.4 Batch vs SGD cost curves

Batch gradient descent normally produces a smoother cost curve because
every update uses the entire training dataset.

SGD is noisier because every parameter update is based on one training
row.

## 5.5 Residuals

The residual is:

actual - predicted

The worst hours can be identified from the saved residual analysis.
Large errors can occur around sunrise and sunset because the relationship
between weather variables and plant output changes rapidly during those
periods. Cloud variation, shading, inverter behavior and sensor/API
differences can also contribute.
