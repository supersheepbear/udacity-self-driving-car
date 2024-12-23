#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
* This is scaffolding, do not modify
*/
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;

  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  //Initialization flag set to false 
  is_initialized_ = false;

  // Augmented state dimension
  n_aug_ = 7;

  // time when the state is true, in us
  time_us_ = 0;

  ///* Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  // NIS
  Radar_NIS_ = 0;
  Lidar_NIS_ = 0;

  //state covariance matrix
  P_ << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1;

  // State dimension
  n_x_ = 5;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Noise matrix
  R_radar = MatrixXd(3, 3);

  R_laser = MatrixXd(2, 2);

  R_radar << std_radr_ * std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0, std_radrd_*std_radrd_;

  R_laser << std_laspx_ * std_laspx_, 0,
    0, std_laspy_*std_laspy_;

  // Augmented matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  // Initialization
  if (!is_initialized_)
  {
    // first measurement
    cout << "Uncented Kalman Filter " << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      //Convert radar from polar to cartesian coordinates and initialize state.
      const double ro = meas_package.raw_measurements_(0);
      const double phi = meas_package.raw_measurements_(1);
      const double roDot = meas_package.raw_measurements_(2);
      //Initialize radar cartesian coordinate positions and velocities
      x_ << ro * cos(phi), ro * sin(phi), roDot, phi, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      //Initialize laser position values
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
  // Prediction

  // get timestamp difference
  const double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }

}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t)
{
  AugmentedSigmaPoints();
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

/**
* Generate augmented sigma points
*/
void UKF::AugmentedSigmaPoints()
{
  // initialize xsig_aug_
  Xsig_aug_.fill(0.0);

  //augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

}

/**
* sigma points prediction function
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
* @param {MatrixXd} meas_package augmented output.
*/
void UKF::SigmaPointPrediction(double delta_t)
{
  //predict sigma points
  for (int i = 0; i< 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t*cos(yaw);
      py_p = p_y + v * delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}

/**
* Calculate predict mean and Covariance
*/
void UKF::PredictMeanAndCovariance()
{
  // set weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i<2 * n_aug_ + 1; i++)
  {  //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
    x_ = x_ + weights_(i)*Xsig_pred_.col(i);
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {  //iterate over sigma points


     // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}


/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package)
{

  // Measurement dimension
  int n_z = 2;

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  z_pred.fill(0.0);

  // Measurement
  VectorXd z = VectorXd(n_z);
   z << meas_package.raw_measurements_(0),
        meas_package.raw_measurements_(1);

  // Cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  // Innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  //mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = Xsig_pred_.col(i)(0);
    double py = Xsig_pred_.col(i)(1);

    Zsig.col(i) << px, py;
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Add measurcouement noise covariance matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R_laser;

  // Calculate cross correlation matrix

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S*K.transpose();
  Lidar_NIS_ = z.transpose() * S.inverse() * z;
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  // Measurement dimension
  int n_z = 3;

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  z_pred.fill(0.0);

  // Radar measurement
  VectorXd z = VectorXd(n_z);

  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);

  // Cross correlation
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  // Innovation covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    // Extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // Measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y * p_y);                        //r
    Zsig(1, i) = atan2(p_y, p_x);                                 //phi
    Zsig(2, i) = (p_x*v1 + p_y * v2) / sqrt(p_x*p_x + p_y * p_y);   //r_dot

    z_pred += weights_(i) * Zsig.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  S = S + R_radar;

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // Angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  // Angle normalization
  while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S*K.transpose();

  // Updata Radar NIS
  Radar_NIS_ = z.transpose() * S.inverse() * z;
}
