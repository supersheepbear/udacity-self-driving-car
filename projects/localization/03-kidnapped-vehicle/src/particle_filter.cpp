/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*      Author: Tiffany Huang
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // Set the number of particles. Initialize all particles to first position (based on estimates of 
  // x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
  // Guassian noise
  double std_x, std_y, std_theta;

  // Initialize number of particles
  num_particles = 50;

  // Standard deviations of GPS measurements
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // Creates a normal (Gaussian) distribution for x,y and theta

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // iterate through all particles and initialze
  for (int i = 0; i < num_particles; i++)
  {
    // initialize
    Particle particle_temp;
    particle_temp.id = i;
    particle_temp.x = dist_x(gen);
    particle_temp.y = dist_y(gen);
    particle_temp.theta = dist_theta(gen);
    particle_temp.weight = 1.0;
    particles.push_back(particle_temp);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  default_random_engine gen;
  // Define random Guassian noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Iterate through all the particles
  for (int i = 0; i < num_particles; i++)
  {
    // In case devided by zero, check yaw change
    if (fabs(yaw_rate) > 0.00001)
    {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    else
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // Add random Gaussian noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  // minumum distance
  double minumum_distsquare;
  int minumum_distID;
  // Iterate through all the particles
  for (unsigned int i = 0; i < observations.size(); i++)
  {
    minumum_distsquare = numeric_limits<double>::max();
    minumum_distID = -1;
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      double xdiff = observations[i].x - predicted[j].x;
      double ydiff = observations[i].y - predicted[j].y;
      double distancesquare = xdiff * xdiff + ydiff * ydiff;
      if (distancesquare < minumum_distsquare)
      {
        minumum_distsquare = distancesquare;
        minumum_distID = predicted[j].id;
      }
    }
    observations[i].id = minumum_distID;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
  const std::vector<LandmarkObs> observations, Map map_landmarks)
{
  // landmarks size
  const int landmarks_size = map_landmarks.landmark_list.size();
  weights.clear();
  for (int i = 0; i < num_particles; i++)
  {
    vector<LandmarkObs> landmark_obj;
    vector<LandmarkObs> trans_obj;

    for (int j = 0; j < landmarks_size; j++)
    {
      LandmarkObs temp_obj;
      float landmarkX = map_landmarks.landmark_list[j].x_f;
      float landmarkY = map_landmarks.landmark_list[j].y_f;
      double xdiff = particles[i].x - landmarkX;
      double ydiff = particles[i].y - landmarkY;
      double distance_square = xdiff * xdiff + ydiff * ydiff;
      if (distance_square < sensor_range * sensor_range)
      {
        temp_obj.x = map_landmarks.landmark_list[j].x_f;
        temp_obj.y = map_landmarks.landmark_list[j].y_f;
        temp_obj.id = map_landmarks.landmark_list[j].id_i;
        // push back to feasible landmarks object vector
        landmark_obj.push_back(temp_obj);
      }
    }

    for (unsigned int j = 0; j < observations.size(); j++)
    {
      LandmarkObs temp_obj;
      temp_obj.x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
      temp_obj.y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
      temp_obj.id = observations[j].id;
      trans_obj.push_back(temp_obj);
    }
    dataAssociation(landmark_obj, trans_obj);
    particles[i].weight = 1.0;
    for (unsigned int j = 0; j < trans_obj.size(); j++) 
    {
      double observationX = trans_obj[j].x;
      double observationY = trans_obj[j].y;
      double Xfound;
      double Yfound;
      unsigned int k = 0;
      bool Matchfound = false;
      while (!Matchfound && k < landmark_obj.size())
      {
        if (landmark_obj[k].id == trans_obj[j].id)
        {
          Matchfound = true;
          Xfound = landmark_obj[k].x;
          Yfound = landmark_obj[k].y;
        }
        k++;
      }
      double xdiff = observationX - Xfound;
      double ydiff = observationY - Yfound;
      double Gauss_norm = 1 / (2 * M_PI*std_landmark[0] * std_landmark[1]);
      double weight = Gauss_norm * exp(-(xdiff*xdiff / (2 * std_landmark[0]*std_landmark[0]) + (ydiff*ydiff / (2 * std_landmark[1]*std_landmark[1]))));
      particles[i].weight *= weight;
    }
    weights.push_back(particles[i].weight);
  }
  weight_max =-1.0;
  for (int i = 0; i < num_particles; i++) 
  {
    if (particles[i].weight > weight_max) 
    {
      weight_max = particles[i].weight;
    }
  }
}

void ParticleFilter::resample()
{
  // Note: This function takes reference from resampling course content and https://github.com/darienmt/CarND-Kidnapped-Vehicle-P3
  default_random_engine gen;

  // Distributions for double and int.
  uniform_real_distribution<double> distDouble(0.0, weight_max);
  uniform_int_distribution<int> distInt(0, num_particles - 1);
  int index = distInt(gen);
  double beta = 0.0;
  // the wheel method
  vector<Particle> resamples;
  for (int i = 0; i < num_particles; i++)
  {
    beta += distDouble(gen) * 2.0;
    while (beta > weights[index]) 
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resamples.push_back(particles[index]);
  }

  particles = move(resamples);
}

Particle ParticleFilter::SetAssociations(Particle particle, const std::vector<int> associations,
  const std::vector<double> sense_x, const std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
