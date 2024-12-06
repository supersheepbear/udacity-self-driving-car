#ifndef LANE_H
#define LANE_H

#include <math.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

// Constants for lane status
namespace lane_status
{
  const int left_lane = 0;
  const int mid_lane = 1;
  const int right_lane = 2;
  const int unknown_lane = 3;
}

// Constants for lane change status
namespace lane_change_status
{
  const int lane_change_on_going = 0;
  const int stay_in_lane = 1;
}

// Constants for determining lane change necessity for onw lane vehicles
namespace lane_change_necessity
{
  // Target s detal threshold to enable lane change
  const double s_delta_threshold = 30.0F;
  // The maximum front vehicle absolute velocity threhold to consider lane change
  const double v_threshold = 45.0F;
}

// Constants for determining lane change feasibility for adjacent lane vehicles
namespace lane_change_feasibility
{
  // absolute target s delta threshold to allow lane change
  const double abs_s_delta_thresh = 12.0F;

  // target time to collision threshold to allow lane change
  const double time_to_collision_thresh = 3.5F;
}

// Constants for cost calculation
namespace lane_change_cost_cal
{
  // the time(sec) to look ahead for cost calculation
  const double look_ahead_time = 5.0F;
}

// function prototypes
int determine_vehicle_lane(double d);

bool check_if_lane_change_needed(vector<vector<double>> sensor_fusion, int host_lane, int prev_size, double host_s, double *closest_front_target_speed);

bool check_goal_lane_change_feasibility(vector<vector<double>> sensor_fusion, int goal_lane, int prev_size, double host_s, double host_speed);

int determin_which_lane_to_go(vector<vector<double>> sensor_fusion, int prev_size, int host_lane, double host_s, double host_speed);

#endif  // LANE_H