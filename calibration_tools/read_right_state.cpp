// Copyright 2026 The RPent Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Read-only libfranka state probe: no control, recovery, gripper or setters.
#include <franka/robot.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Serialize a robot-state array, rejecting values that JSON cannot represent.
template <typename T>
void array_json(const T& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ",";
    }
    if (!std::isfinite(values[index])) {
      throw std::runtime_error("Robot state contains a non-finite value");
    }
    std::cout << values[index];
  }
  std::cout << "]";
}

// Read one state from the selected robot; stdout is reserved for the JSON record.
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: read_franka_state ROBOT_IP\n";
    return 2;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    auto state = robot.readOnce();
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    std::cout << std::setprecision(17)
              << "{\"host_time_s\":" << std::chrono::duration<double>(now).count()
              << ",\"robot_time_ms\":" << state.time.toMSec()
              << ",\"robot_mode\":" << static_cast<int>(state.robot_mode)
              << ",\"has_errors\":"
              << (bool(state.current_errors) ? "true" : "false")
              << ",\"O_T_EE\":";
    array_json(state.O_T_EE);
    std::cout << ",\"F_T_EE\":";
    array_json(state.F_T_EE);
    std::cout << ",\"EE_T_K\":";
    array_json(state.EE_T_K);
    std::cout << ",\"q\":";
    array_json(state.q);
    std::cout << ",\"dq\":";
    array_json(state.dq);
    std::cout << "}" << std::endl;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
