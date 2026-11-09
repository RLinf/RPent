// Copyright 2026 The RPent Authors.
// Licensed under the Apache License, Version 2.0.
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Memory.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Statistics.h>
#include <opencv2/imgproc.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace py = pybind11;
using namespace rtabmap;
using Matrix = std::array<std::array<float, 4>, 4>;

class Localizer {
    Rtabmap map;
    std::mutex mutex;
    bool closed = false;

    void require_open() const {
        if (closed) throw std::runtime_error("RTAB-Map localizer is closed");
    }

public:
    explicit Localizer(const std::string &database) {
        ParametersMap parameters = {
            {"Mem/IncrementalMemory", "false"}, {"Mem/InitWMWithAllNodes", "true"},
            {"Mem/LocalizationReadOnly", "true"}, {"Mem/LocalizationDataSaved", "false"},
            {"Rtabmap/MemoryThr", "0"}, {"Rtabmap/TimeThr", "0"},
            {"RGBD/Enabled", "true"}, {"RGBD/CreateOccupancyGrid", "false"},
            {"RGBD/LinearUpdate", "0"}, {"RGBD/AngularUpdate", "0"},
            {"Rtabmap/StartNewMapOnLoopClosure", "false"},
            {"RGBD/ForceOdom3DoF", "true"}, {"Reg/Force3DoF", "true"},
            {"Reg/Strategy", "0"}, {"Vis/EstimationType", "1"},
            {"Vis/MinInliers", "30"}, {"Vis/PnPReprojError", "3"},
            {"RGBD/ProximityBySpace", "false"}, {"RGBD/ProximityByTime", "false"}
        };
        py::gil_scoped_release release;
        map.init(parameters, database, true);
    }

    ~Localizer() {
        if (!closed) {
            try { map.close(false); } catch (...) {}
        }
    }

    void close() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex);
        if (!closed) {
            map.close(false);
            closed = true;
        }
    }

    py::dict goal(py::object target) {
        const bool label = py::isinstance<py::str>(target);
        const std::string name = label ? target.cast<std::string>() : "";
        int id = label ? 0 : target.cast<int>();
        Transform pose;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex);
            require_open();
            if (label) id = map.getMemory()->getSignatureIdByLabel(name);
            std::map<int, Transform> poses;
            std::multimap<int, Link> links;
            map.getGraph(poses, links, true, true);
            const auto found = poses.find(id);
            if (id > 0 && found != poses.end()) pose = found->second;
        }
        py::dict result;
        result["status"] = pose.isNull() ? "unknown_goal" : "goal";
        if (!pose.isNull()) {
            result["node_id"] = id;
            result["xy"] = py::cast(std::array<float, 2>{pose.x(), pose.y()});
            result["yaw"] = pose.theta();
        }
        return result;
    }

    py::list list_nodes() {
        std::map<int, Transform> poses;
        std::map<int, std::string> labels;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex);
            require_open();
            std::multimap<int, Link> links;
            map.getGraph(poses, links, true, true);
            labels = map.getMemory()->getAllLabels();
        }
        py::list result;
        for (const auto &entry : poses) {
            if (entry.first <= 0 || entry.second.isNull()) continue;
            py::dict node;
            node["node_id"] = entry.first;
            node["label"] = labels[entry.first];
            node["pose"] = py::cast(std::array<float, 3>{
                entry.second.x(), entry.second.y(), entry.second.theta()});
            result.append(node);
        }
        return result;
    }

    py::dict inspect_node(int id) {
        if (id <= 0) throw std::invalid_argument("node_id must be positive");
        Transform pose;
        std::string label;
        double stamp = 0;
        cv::Mat image;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex);
            require_open();
            std::map<int, Transform> poses;
            std::multimap<int, Link> links;
            map.getGraph(poses, links, true, true);
            const auto found = poses.find(id);
            if (found == poses.end() || found->second.isNull())
                throw std::invalid_argument("Unknown RTAB-Map node");
            pose = found->second;
            Transform odom, groundTruth;
            int mapId, weight;
            std::vector<float> velocity;
            GPS gps;
            EnvSensors sensors;
            if (!map.getMemory()->getNodeInfo(id, odom, mapId, weight, label,
                    stamp, groundTruth, velocity, gps, sensors, true))
                throw std::invalid_argument("Node metadata unavailable");
            image = map.getMemory()->getImageCompressed(id).clone();
        }
        py::dict result;
        result["node_id"] = id;
        result["label"] = label;
        result["pose"] = py::cast(std::array<float, 3>{pose.x(), pose.y(), pose.theta()});
        result["observed_at_s"] = stamp;
        result["image"] = py::none();
        if (!image.empty()) {
            py::array_t<uint8_t> buffer(image.total() * image.elemSize());
            std::memcpy(buffer.mutable_data(), image.data, buffer.nbytes());
            result["image"] = buffer;
        }
        return result;
    }

    py::dict localize(py::array_t<uint8_t, py::array::c_style> rgb,
                      py::dict calibration, double stamp_ns, const Matrix &mount_matrix,
                      const std::array<float, 3> &odometry) {
        const int width = calibration["width"].cast<int>();
        const int height = calibration["height"].cast<int>();
        const auto p = calibration["params"].cast<std::array<double, 4>>();
        if (width < 32 || width > 4096 || height < 32 || height > 4096 ||
            rgb.ndim() != 3 || rgb.shape(0) != height ||
            rgb.shape(1) != width || rgb.shape(2) != 3)
            throw std::invalid_argument("Expected H x W x 3 uint8 RGB image matching camera dimensions");
        for (const auto value : p)
            if (!std::isfinite(value)) throw std::invalid_argument("Invalid intrinsics");
        if (p[0] <= 0 || p[1] <= 0 || p[2] < 0 || p[2] >= width || p[3] < 0 || p[3] >= height)
            throw std::invalid_argument("Invalid intrinsics");
        for (const auto &row : mount_matrix)
            for (const auto value : row)
                if (!std::isfinite(value)) throw std::invalid_argument("Invalid camera transform");
        for (const auto value : odometry)
            if (!std::isfinite(value)) throw std::invalid_argument("Invalid odometry");
        if (!std::isfinite(stamp_ns)) throw std::invalid_argument("Invalid timestamp");
        const auto &m = mount_matrix;
        const Transform mount(m[0][0], m[0][1], m[0][2], m[0][3],
                              m[1][0], m[1][1], m[1][2], m[1][3],
                              m[2][0], m[2][1], m[2][2], m[2][3]);
        CameraModel camera(p[0], p[1], p[2], p[3], mount, 0, cv::Size(width, height));
        // Own the converted pixels before releasing the GIL.
        cv::Mat image;
        cv::Mat view(height, width, CV_8UC3, const_cast<uint8_t *>(rgb.data()));
        cv::cvtColor(view, image, cv::COLOR_RGB2BGR);
        int reference_id;
        bool matched;
        Transform base;
        std::map<std::string, float> metrics;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex);
            require_open();
            SensorData data(image, cv::Mat(), camera, 0, stamp_ns * 1e-9);
            Transform odom(odometry[0], odometry[1], 0, 0, 0, odometry[2]);
            const bool processed = map.process(data, odom);
            const auto &stats = map.getStatistics();
            reference_id = stats.loopClosureId();
            // Propagated odometry is not a fresh visual localization.
            matched = processed && reference_id > 0;
            base = map.getLastLocalizationPose();
            const auto &values = stats.data();
            for (const auto &metric : {std::make_pair("inliers", "Loop/Visual_inliers/"),
                                       std::make_pair("matches", "Loop/Visual_matches/")}) {
                const auto found = values.find(metric.second);
                if (found != values.end()) metrics[metric.first] = found->second;
            }
        }
        py::dict result;
        result["status"] = "no_match";
        result["reference_id"] = reference_id;
        for (const auto &metric : metrics) result[py::str(metric.first)] = metric.second;
        if (matched && !base.isNull()) {
            const auto pose = base * mount;
            Matrix matrix{};
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col) matrix[row][col] = pose(row, col);
            matrix[3][3] = 1;
            result["status"] = "localized";
            result["T_map_camera"] = py::cast(matrix);
        }
        return result;
    }
};

PYBIND11_MODULE(lekiwi_localization, module) {
    py::class_<Localizer>(module, "Localizer")
        .def(py::init<const std::string &>(), py::arg("database"))
        .def("localize", &Localizer::localize, py::arg("rgb").noconvert(),
             py::arg("camera"), py::arg("stamp_ns"), py::arg("T_base_camera"),
             py::arg("odometry"))
        .def("goal", &Localizer::goal, py::arg("target"))
        .def("list_nodes", &Localizer::list_nodes)
        .def("inspect_node", &Localizer::inspect_node, py::arg("node_id"))
        .def("close", &Localizer::close);
}
