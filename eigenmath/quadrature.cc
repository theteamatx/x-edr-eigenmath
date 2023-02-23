// Copyright 2023 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "quadrature.h"

#include <array>

namespace eigenmath {

const std::array<double, 2> GaussLegendreCoeffs<2>::knots = {
    {-0.577350269189626, 0.577350269189626}};

const std::array<double, 2> GaussLegendreCoeffs<2>::weights = {{1.0, 1.0}};

const std::array<double, 3> GaussLegendreCoeffs<3>::knots = {
    {-0.774596669241483, 0.0, 0.774596669241483}};

const std::array<double, 3> GaussLegendreCoeffs<3>::weights = {
    {0.555555555555556, 0.888888888888889, 0.555555555555556}};

const std::array<double, 4> GaussLegendreCoeffs<4>::knots = {
    {-0.861136311594053, -0.339981043584856, 0.339981043584856,
     0.861136311594053}};

const std::array<double, 4> GaussLegendreCoeffs<4>::weights = {
    {0.347854845137454, 0.652145154862546, 0.652145154862546,
     0.347854845137454}};

const std::array<double, 5> GaussLegendreCoeffs<5>::knots = {
    {-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683,
     0.906179845938664}};

const std::array<double, 5> GaussLegendreCoeffs<5>::weights = {
    {0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366,
     0.236926885056189}};

const std::array<double, 6> GaussLegendreCoeffs<6>::knots = {
    {-0.932469514203152, -0.661209386466265, -0.238619186083197,
     0.238619186083197, 0.661209386466265, 0.932469514203152}};

const std::array<double, 6> GaussLegendreCoeffs<6>::weights = {
    {0.171324492379170, 0.360761573048139, 0.467913934572691, 0.467913934572691,
     0.360761573048139, 0.171324492379170}};

const std::array<double, 7> GaussLegendreCoeffs<7>::knots = {
    {-0.949107912342759, -0.741531185599394, -0.405845151377397, 0.0,
     0.405845151377397, 0.741531185599394, 0.949107912342759}};

const std::array<double, 7> GaussLegendreCoeffs<7>::weights = {
    {0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469,
     0.381830050505119, 0.279705391489277, 0.129484966168870}};

const std::array<double, 8> GaussLegendreCoeffs<8>::knots = {
    {-0.960289856497536, -0.796666477413627, -0.525532409916329,
     -0.183434642495650, 0.183434642495650, 0.525532409916329,
     0.796666477413627, 0.960289856497536}};

const std::array<double, 8> GaussLegendreCoeffs<8>::weights = {
    {0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362,
     0.362683783378362, 0.313706645877887, 0.222381034453374,
     0.101228536290376}};

const std::array<double, 9> GaussLegendreCoeffs<9>::knots = {
    {-0.968160239507626, -0.836031107326636, -0.613371432700590,
     -0.324253423403809, 0.0, 0.324253423403809, 0.613371432700590,
     0.836031107326636, 0.968160239507626}};

const std::array<double, 9> GaussLegendreCoeffs<9>::weights = {
    {0.081274388361574, 0.180648160694857, 0.260610696402935, 0.312347077040003,
     0.330239355001260, 0.312347077040003, 0.260610696402935, 0.180648160694857,
     0.081274388361574}};

const std::array<double, 10> GaussLegendreCoeffs<10>::knots = {
    {-0.973906528517172, -0.865063366688985, -0.679409568299024,
     -0.433395394129247, -0.148874338981631, 0.148874338981631,
     0.433395394129247, 0.679409568299024, 0.865063366688985,
     0.973906528517172}};

const std::array<double, 10> GaussLegendreCoeffs<10>::weights = {
    {0.066671344308688, 0.149451349150581, 0.219086362515982, 0.269266719309996,
     0.295524224714753, 0.295524224714753, 0.269266719309996, 0.219086362515982,
     0.149451349150581, 0.066671344308688}};

const std::array<double, 12> GaussLegendreCoeffs<12>::knots = {
    {-0.981560634246719, -0.904117256370475, -0.769902674194305,
     -0.587317954286617, -0.367831498998180, -0.125233408511469,
     0.125233408511469, 0.367831498998180, 0.587317954286617, 0.769902674194305,
     0.904117256370475, 0.981560634246719}};

const std::array<double, 12> GaussLegendreCoeffs<12>::weights = {
    {0.047175336386512, 0.106939325995318, 0.160078328543346, 0.203167426723066,
     0.233492536538355, 0.249147045813403, 0.249147045813403, 0.233492536538355,
     0.203167426723066, 0.160078328543346, 0.106939325995318,
     0.047175336386512}};

}  // namespace eigenmath
