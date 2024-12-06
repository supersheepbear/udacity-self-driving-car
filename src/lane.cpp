#include <iostream>
#include "lane.h"
#include <math.h>

static bool check_lane_change_space_condition(double target_s, double host_s);
static bool check_lane_change_time_condition(double target_s, double host_s, double target_speed, double host_speed);
static double calculate_lane_score(vector<vector<double>> sensor_fusion, int goal_lane, double prev_size, double host_s, double host_speed);

/***************************************************
Given the vehicle d in frenet coordinate, determine 
what lane the vehicle belongs to.
****************************************************/
int determine_vehicle_lane(double d)
{
  int lane;
  if (d < (2 + 4 * 0 + 2) && d >(2 + 4 * 0 - 2))
  {
    lane = lane_status::left_lane;
  }
  else if (d < (2 + 4 * 1 + 2) && d >(2 + 4 * 1 - 2))
  {
    lane = lane_status::mid_lane;
  }
  else if (d < (2 + 4 * 2 + 2) && d >(2 + 4 * 2 - 2))
  {
    lane = lane_status::right_lane;
  }
  else
  {
    lane = lane_status::unknown_lane;
  }
  return lane;
}


/***************************************************
Check own lane vehicle that is in front of host, and 
determine if lane change is needed based on front 
vehicle speed.
****************************************************/
bool check_if_lane_change_needed(vector<vector<double>> sensor_fusion, int host_lane, int prev_size, double host_s,
  double *closest_front_target_speed)
{
  bool lane_change_needed = false;
  double closest_front_target_s = 1000.0;
  // Loop through all target vehicles
  for (int i = 0; i < sensor_fusion.size(); i++)
  {
    // Parse the frenet coordinate d of target vehicle
    double d = sensor_fusion[i][6];
    // Check target vehicle lane status
    int target_lane = determine_vehicle_lane(d);
    // if target vehicle has the same lane status as host vehicle, apply logic 
    if (target_lane == host_lane)
    {
      double target_vx = sensor_fusion[i][3];
      double target_vy = sensor_fusion[i][4];
      double target_speed = sqrt(target_vx * target_vx + target_vy * target_vy);
      double target_s = sensor_fusion[i][5];
      target_s += ((double)prev_size*0.02*target_speed);
      // if front target is too close and move slowly
      if ((target_s > host_s)
        && ((target_s - host_s) < lane_change_necessity::s_delta_threshold)
        && (target_speed < lane_change_necessity::v_threshold))
      {
        lane_change_needed = true;
        if( target_s < closest_front_target_s)
        {
          closest_front_target_s = target_s;
          // sensor fusion vehicle speed is in m/s, need to be trasformed to kph
          *closest_front_target_speed = target_speed/2.237;
        }
      }
    }
  }
  return lane_change_needed;
}


/***************************************************
Based one current lane status, check lane change
feasibility accordingly.
****************************************************/
int determin_which_lane_to_go(vector<vector<double>> sensor_fusion, int prev_size, int host_lane, double host_s, double host_speed)
{
  bool change_lane_left_possible;
  bool change_lane_right_possible;
  double host_lane_score = calculate_lane_score(sensor_fusion, host_lane, prev_size, host_s, host_speed);
  double left_lane_score = 0.0;
  double right_lane_score = 0.0;
  double mid_lane_score = 0.0;
  // based on cost, determine which lane to go.
  int final_goal_lane = host_lane;

  if (host_lane == lane_status::left_lane)
  {
    left_lane_score = host_lane_score;
    // left lane can only change right to mid lane
    change_lane_right_possible = check_goal_lane_change_feasibility(sensor_fusion, lane_status::mid_lane, prev_size, host_s, host_speed);
    if (change_lane_right_possible)
    {
      mid_lane_score = calculate_lane_score(sensor_fusion, lane_status::mid_lane, prev_size, host_s, host_speed);
      if (mid_lane_score > left_lane_score)
      {
        final_goal_lane = lane_status::mid_lane;
      }
    }
  }
  else if (host_lane == lane_status::mid_lane)
  { 
    mid_lane_score = host_lane_score;
    // host in mid lane can change to left or right
    change_lane_left_possible = check_goal_lane_change_feasibility(sensor_fusion, lane_status::left_lane, prev_size, host_s, host_speed);
    change_lane_right_possible = check_goal_lane_change_feasibility(sensor_fusion, lane_status::right_lane, prev_size, host_s, host_speed);
    //std::cout << "change_lane_left_possible: " << change_lane_left_possible << std::endl;
    //std::cout << "change_lane_right_possible: " << change_lane_right_possible << std::endl;
    // situation that change lane left is possible, but change lane right is impossible.

    if(  (change_lane_left_possible  )
       &&(!change_lane_right_possible))
    {
      left_lane_score = calculate_lane_score(sensor_fusion, lane_status::left_lane, prev_size, host_s, host_speed);
      if (left_lane_score > mid_lane_score)
      {
        final_goal_lane = lane_status::left_lane;
      }
    }
    // situation that change lane right is possible, but change lane left is impossible.
    else if (   (change_lane_right_possible   )
             && (!change_lane_left_possible))
    {
      right_lane_score = calculate_lane_score(sensor_fusion, lane_status::right_lane, prev_size, host_s, host_speed);
      if (right_lane_score > mid_lane_score)
      {
        final_goal_lane = lane_status::right_lane;
      }
    }
    // Situation that change lane left anre change lane right are both impossible.
    // Check all three lanes' score to decide which lane to go.
    else if (   (change_lane_left_possible  )
             && (change_lane_right_possible))
    {
      left_lane_score = calculate_lane_score(sensor_fusion, lane_status::left_lane, prev_size, host_s, host_speed);
      right_lane_score = calculate_lane_score(sensor_fusion, lane_status::right_lane, prev_size, host_s, host_speed);

      if (left_lane_score > mid_lane_score)
      {
        if (right_lane_score > left_lane_score)
        {
          final_goal_lane = lane_status::right_lane;
        }
        else
        {
          final_goal_lane = lane_status::left_lane;
        }
      }
      else if(right_lane_score > mid_lane_score)
      {
        final_goal_lane = lane_status::right_lane;
      }
    }
  }
  else if (host_lane == lane_status::right_lane)
  {
    right_lane_score = host_lane_score;
    // host in right lane can change to mid lane
    change_lane_left_possible = check_goal_lane_change_feasibility(sensor_fusion, lane_status::mid_lane, prev_size, host_s, host_speed);
    if (change_lane_left_possible)
    {
      mid_lane_score = calculate_lane_score(sensor_fusion, lane_status::mid_lane, prev_size, host_s, host_speed);
      if (mid_lane_score > right_lane_score)
      {
        final_goal_lane = lane_status::mid_lane;
      }
    }
  }
  else
  {
    // do nothing. do not change lane if host is out of bound.
  }
  return final_goal_lane;
}


/***************************************************
Check if lane change for the goal lane is feasible.
****************************************************/
bool check_goal_lane_change_feasibility(vector<vector<double>> sensor_fusion, int goal_lane, int prev_size, double host_s, double host_speed)
{
  // Init lane change feasibility to true
  bool lane_change_feasible = true;

  // Define intermediate conditions booleans
  bool space_condition_met = true;
  bool time_condition_met = true;

  // Loop through all target vehicles
  for (int i = 0; i < sensor_fusion.size(); i++)
  {
    // Parse the frenet coordinate d of target vehicle
    double d = sensor_fusion[i][6];
    // Check target vehicle lane status
    int target_lane = determine_vehicle_lane(d);
    // if target vehicle is in the lane number that is identical to the goal lane that we want to change lane to.
    if (target_lane == goal_lane)
    {
      double target_vx = sensor_fusion[i][3];
      double target_vy = sensor_fusion[i][4];
      double target_speed = sqrt(target_vx * target_vx + target_vy * target_vy);
      double target_s = sensor_fusion[i][5];
      target_s += ((double)prev_size*0.02*target_speed);
      space_condition_met = check_lane_change_space_condition(target_s, host_s);
      time_condition_met = check_lane_change_time_condition(target_s, host_s, target_speed, host_speed);
      lane_change_feasible &= space_condition_met;
      lane_change_feasible &= time_condition_met;
    }
  }
  //std::cout << "space_condition_met: " << space_condition_met << std::endl;
  //std::cout << "time_condition_met: " << time_condition_met << std::endl;
  return lane_change_feasible;
}

/***************************************************
Check adjacent lane vehicle space condition for if 
lane change is feasible.
****************************************************/
static bool check_lane_change_space_condition (double target_s, double host_s)
{
  bool space_condition_met = true;
  // if target s and host s are too small, don't allow lane change.
  if (fabs(target_s - host_s) < lane_change_feasibility::abs_s_delta_thresh)
  {
    space_condition_met = false;
  }
  else
  {
    space_condition_met = true;
  }
  return space_condition_met;
}


/***************************************************
Check time condition for if lane change is feasible.
****************************************************/
static bool check_lane_change_time_condition(double target_s, double host_s, double target_speed, double host_speed)
{
  bool time_condition_met = true;

  double time_to_collision = (host_s - target_s) / (target_speed - host_speed);
  // if target and host movement indicates a possible collision in time dimension, not allow lane change
  if(  (time_to_collision > 0)
     &&(time_to_collision < lane_change_feasibility::time_to_collision_thresh))
  {
    time_condition_met = false;
  }
  else
  {
    time_condition_met = true;
  }
  return time_condition_met;
}

/***************************************************
Calculate a score for current lane, for determining
if it is worth to change to this lane.
****************************************************/
static double calculate_lane_score(vector<vector<double>> sensor_fusion, int goal_lane, double prev_size, double host_s, double host_speed)
{
  // variables to find out the closet front target
  bool this_lane_has_target = false;
  int closet_front_target_id;
  double closest_front_target_s_delta = 1000.0;
  double closest_front_target_speed;

  // the score of current lane
  double lane_score;
  
  // Loop through all target vehicles
  for (int i = 0; i < sensor_fusion.size(); i++)
  {
    // Parse the frenet coordinate d of target vehicle
    double d = sensor_fusion[i][6];
    // Check target vehicle lane status
    int target_lane = determine_vehicle_lane(d);
    // only check target that is on the goal lane.
    if (target_lane == goal_lane)
    {
      double target_vx = sensor_fusion[i][3];
      double target_vy = sensor_fusion[i][4];
      double target_speed = sqrt(target_vx * target_vx + target_vy * target_vy);
      double target_s = sensor_fusion[i][5];
      target_s += ((double)prev_size*0.02*target_speed);
      double delta_s = target_s - host_s;
      if(  (delta_s > 0.0)
         &&(delta_s < closest_front_target_s_delta))
      {
        closet_front_target_id = i;
        closest_front_target_s_delta = delta_s;
        closest_front_target_speed = target_speed;
        this_lane_has_target = true;
      }
    }
  }

  // If no target at all on this lane
  if (!this_lane_has_target)
  {
    // free to change lane, give a very large score to make sure we change to this lane!
    lane_score = 10000.0;
  }
  else
  {
    // look ahead a certain time to determine the future space that host is able to travel.
    // The space that host can travel in this designed time is designed to be the score of chaning to this lane
    lane_score = (closest_front_target_speed - host_speed) * lane_change_cost_cal::look_ahead_time;
    lane_score += closest_front_target_s_delta;
  }
  return lane_score;
}