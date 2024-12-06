# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program
   
### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).  

To run the simulator on Mac/Linux, first make the binary file executable with the following command:
```shell
sudo chmod u+x {simulator_file_name}
```

### Goals
This project simulates a virtual highway with numerous cars running at speeds +- 10 mph of a 50 mph speed limit.

The goal of was to program the "ego" car to:

* The car is able to drive at least 4.32 miles without incident.
* Stay close to the speed limit without exceeding it
* Accelerate and decelerate within jerk limits
* Avoid all collisions
* Drive inside the lane lines, except when changing lanes
* Able to change lane when there is open on side lanes

As shown in the [video](https://github.com/x327397818/UDC-Term3_project1/tree/master/Video), this implementation can run at least 10 miles / 11 minutes without incident and meet all points in the [rubric](https://review.udacity.com/#!/rubrics/1020/view).

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
# Trajectories
For trajectory generation, my code are basically the same as Q&A section.
# Car following strategy
I have found out that, when following a target vehicle, the host vehicle is not moving so stable to have a reference velocity to be 50mph all the time.
My solution is simple. I found out the closest front vehicle velocity, and try to manage the host vehicle speed around its speed.
Here is the code in main.c line 129 to line 137:
```c
                if (ref_vel > closest_front_target_speed)
                {
                  ref_vel -= 0.224;
                }
                else
                {
                  ref_vel += 0.224;
                }
```

# lane change state
my strategy is to have following host lane change state:
```c
// Constants for lane change status
namespace lane_change_status
{
  const int lane_change_on_going = 0;
  const int stay_in_lane = 1;
}
```
If current state is stay_in_lane, host vehicle will be moving just in the current lane, and actively determine which lane to go(stay in own lane or change lane?)

If algorithm determines to change lane, the state will become from stay_in_lane to lane_change_on_going, host vehicle will be in a lane change manuever until the lane change is finished. After lane change finished, the lane change state will change back to stay_in_lane. 


# lane change neccessity check

The function 'check_if_lane_change_needed' in lane.cpp, line 41 to line 77, is trying to determine if lane change is neccesary. This is by checking, if the target vehicle that is in front of host vehicle is moving too slowly, and too close.

Once the lane change is determine to be needed, then it goes to function 'determin_which_lane_to_go' in lane.cpp, line 80 to 182.
This function firstly determine which lane is possible for a change, based on which lane the host vehicle is located at. Apprarently:

* left lane vehicle can change to mid lane only
* mid lane vehicle can change to left lane or right lane
* right lane vehicle can change to mid lane only

The left lane and right lane has only one option, while the mid lane has two options.
For each possible lane, there is a feasibility check donw.

# lane change feasibility check
Below are the conditions that are checked for the feasibilty of lane change. This is actually from production vehicle blind spot detection logic.
* space condition. If any of adjacent lane vehicle is within host vehicle s value +- 12 meters, don't allow lane change
* time condition. If time to collision value, calculated as (host_s - target_s) / (target_speed - host_speed), is smaller than 3.5 m/s, don't allow lane change.

Code can be found through lane.cpp line 185 to 263

# lane scoring(cost function)
If lane change is possible, for all the lanes that host vehicle is feasible go, including the host vehicle lane itself, give a score to each lane, representing how much worthy it is to go such lane.

The score is based on looking ahead in 5 sec, for how much space the host vehicle is able to travel in such lane. The code can be found in lane.cpp line 264 to line 320.

The actuall lane scording equation is:
```
lane_score = (closest_front_target_speed - host_speed) * look_ahead_time +  closest_front_target_s_delta;
```
Based on score on each feasible lane, host vehicle will determine to keep stay_in_lane status, or changes to lane_change_on_going state, which means doing a lane change.





