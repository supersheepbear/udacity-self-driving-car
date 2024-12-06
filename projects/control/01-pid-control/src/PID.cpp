#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) 
{
  // Initialize errors
  p_error     = 0;
  i_error     = 0;
  d_error     = 0;
  history_cte = 0;
  // Initialize P,I,D to argument values
  PID::Kp = Kp;
  PID::Ki = Ki;
  PID::Kd = Kd;
}

void PID::UpdateError(double cte) 
{
  // Update P error
  p_error = cte;
  // Update I error
  i_error += cte;
  // Update D error
  d_error = cte - history_cte;
  // Store cte to history cte
  history_cte = cte;
}

double PID::TotalError() 
{
  // calculate total error for PID
  const double total_error = -1 * Kp * p_error - Ki * i_error - 1 * Kd * d_error;
  return total_error;
}

