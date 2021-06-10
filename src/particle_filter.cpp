/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::max_element;
using std::normal_distribution;
using std::numeric_limits;
using std::string;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::vector;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */

  // Set the number of particles
  num_particles = 10;

  // random Gaussian noise range 0 ~ std
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  for (int id = 0; id < num_particles; ++id) {
    Particle particle;
    particle.id = id;
    particle.x = x + dist_x(gen);              // add noise
    particle.y = y + dist_y(gen);              // add noise
    particle.theta = theta + dist_theta(gen);  // add noise
    particle.weight = 1.0;                     // set default as 1.0

    // initialize particels
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // random Gaussian noise range 0 ~ std
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    double theta = particles[i].theta;
    double d_theta = yaw_rate * delta_t;

    // predict update
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x = particles[i].x + velocity * cos(theta) * delta_t;
      particles[i].y = particles[i].y + velocity * sin(theta) * delta_t;
      particles[i].theta = theta + d_theta;
    } else {
      particles[i].x = particles[i].x + (velocity / yaw_rate) *
                                            (sin(theta + d_theta) - sin(theta));
      particles[i].y = particles[i].y + (velocity / yaw_rate) *
                                            (cos(theta) - cos(theta + d_theta));
      particles[i].theta = theta + d_theta;
    }

    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

  for (int i = 0; i < observations.size(); ++i) {
    // using nearest-neighbors search
    double min_distance = numeric_limits<double>::infinity();
    int nearest_id = -1;

    for (auto p : predicted) {
      // calculate Euclidean distance
      double distance = dist(observations[i].x, observations[i].y, p.x, p.y);

      // set nearest-neighbors data association
      if (distance < min_distance) {
        min_distance = distance;
        nearest_id = p.id;
      }
    }
    observations[i].id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no
   * scaling). The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // landmark measurement uncertainty
  double sig_x, sig_y;
  sig_x = std_landmark[0];
  sig_y = std_landmark[1];

  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

  // update weights for particles
  for (unsigned int i = 0; i < num_particles; ++i) {
    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta = particles[i].theta;

    // reset weight
    particles[i].weight = 1.0;

    // transform VEHICLE'S to MAP'S coordinate system
    vector<LandmarkObs> observations_m;
    for (auto obs : observations) {
      LandmarkObs obs_m;
      obs_m.id = obs.id;
      obs_m.x = x_part + (cos(theta) * obs.x) - (sin(theta) * obs.y);
      obs_m.y = y_part + (sin(theta) * obs.x) + (cos(theta) * obs.y);
      observations_m.push_back(obs_m);
    }

    // filter by sensor exploration range
    vector<LandmarkObs> predictions_m;
    for (auto landmark : map_landmarks.landmark_list) {
      // move the original point MAP'S coordinate to VEHICLE'S
      LandmarkObs lm_m;
      if (fabs(landmark.x_f - x_part) <= sensor_range &&
          fabs(landmark.y_f - y_part) <= sensor_range) {
        lm_m.id = landmark.id_i;
        lm_m.x = landmark.x_f;
        lm_m.y = landmark.y_f;
        predictions_m.push_back(lm_m);
      }
    }

    // Association
    dataAssociation(predictions_m, observations_m);

    // calculate the particle's weight by Multivariate-Gaussian
    for (auto obs_m : observations_m) {
      for (auto pred_m : predictions_m) {
        if (obs_m.id == pred_m.id) {
          double exponent;
          exponent = pow(obs_m.x - pred_m.x, 2) / (2 * pow(sig_x, 2)) +
                     pow(obs_m.y - pred_m.y, 2) / (2 * pow(sig_y, 2));

          particles[i].weight *= (gauss_norm * exp(-1.0 * exponent));
        }
      }
    }
    weights.push_back(particles[i].weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // resample new particels
  vector<Particle> resample_particles;

  // generate index
  uniform_int_distribution<int> dist_index(0, num_particles - 1);
  int index = dist_index(gen);

  // declare selector
  double beta = 0.0;

  // generate for weight
  double max_weight = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> dist_weight(0.0, max_weight);

  for (int i = 0; i < num_particles; ++i) {
    beta += (2.0 * dist_weight(gen));

    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resample_particles.push_back(particles[index]);
  }
  // reset particles and weights
  particles = resample_particles;
  weights.clear();
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}