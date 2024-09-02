#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/ColorRGBA.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <hmcl_msgs/LaneArray.h>
#include <hmcl_msgs/Lane.h>
#include <hmcl_msgs/BehaviorFactor.h>
#include <vector>
#include <cmath>
#include <map>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
class ModelPredictive {
public:
    ModelPredictive();
    void detected_object_callback(const autoware_msgs::DetectedObjectArray::ConstPtr& msg);
    void globalTrajCallback(const hmcl_msgs::LaneArray::ConstPtr& lane_msg);
    void globalTrajCallback1(const hmcl_msgs::Lane::ConstPtr& msg);
    void currentPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void run();
    void detected_object_kinematic(int obj);
    void Intuitive_Artificial_potential_field_2();
    std::pair<double, double> cartesian_to_frenet(const std::vector<std::vector<double>>& centerline, const std::vector<double>& point);
    std::pair<double, double> frenet_to_cartesian(const std::vector<std::vector<double>>& centerline, double s, double l);
    void OG_PUB();
    visualization_msgs::MarkerArray visualize_local_path(const std::vector<hmcl_msgs::Waypoint>& waypoints);
    void Bezier_Curve();
    std::vector<std::pair<double, double>> calc_curve(int num_points);
    void Trajectory_Generation();
    void compute_local_path();
    void compute_local_path1();
    int calculate_distance_pose2local();
    int calculate_distance_pose2local_1();
    double distance(double x1, double y1, double x2, double y2);
    void local_lane_to_local_points();
    void local_lane_to_local_points1();
    void globalTrajCallback2(const hmcl_msgs::LaneArray::ConstPtr& lane_msg);
private:
    std::vector<int> obj; 
    autoware_msgs::DetectedObjectArray objects_data;
    ros::NodeHandle nh_;
    ros::Subscriber target_sub_;
    ros::Subscriber global_traj_sub_;
    ros::Subscriber current_pose_sub_;
    ros::Subscriber global_traj1_sub_;
    ros::Subscriber global_traj2_sub_;
    ros::Publisher overtaking_traj_pub_;
    ros::Publisher marker_pub_;
    std::vector<std::vector<double> > local_points, optimal_points, global_points,local_points1;
    int model_predicted_num;
    double dt;
    std::vector<double> control;
    std::vector<double> state;
    double qx, qy, qz, qw;  
    double x, y;
    std::map<int, std::vector<double>> target_vel_x, target_vel_y; // Assuming use of id as key
    std::map<int, double> target_x, target_y, target_velocity_x, target_velocity_y, target_orientation_x, target_orientation_y, target_orientation_z, target_orientation_w, target_yaw_veh, target_angular_z_veh;
    std::vector<double> cx, cy, cqx, cqy, cqz, cqw, global_cx,global_cy;
    std::map<int, std::vector<double>> target_veh_dic_x, target_veh_dic_y;
    std::map<int, std::vector<double>> range_point_x, range_point_y, repulsed_potential_field_point_x_veh, repulsed_potential_field_point_y_veh;
    
    std::vector<double> first_point, second_point, third_point, fourth_point,fifth_point,sixth_point;
    std::vector<std::pair<double, double>> B;
    std::vector<double> repulsed_potential_field_point_x;
    std::vector<double> repulsed_potential_field_point_y;
    double target_velocity;
    double target_angular_z;
    double gain;
    double sigma;
    double radius;
    double repulsed_s, repulsed_d;
    double repulsed_x, repulsed_y;
    hmcl_msgs::LaneArray global_lane_array;
    hmcl_msgs::LaneArray global_lane_array1;
    hmcl_msgs::Lane local_lane;
    hmcl_msgs::Lane local_lane1;
    double pose_x, pose_y;
    bool global_traj_available = false;
    bool init_obs = false;
      
};

ModelPredictive::ModelPredictive() : model_predicted_num(5), dt(0.1) {
    target_sub_ = nh_.subscribe("/tracking_side/objects", 1, &ModelPredictive::detected_object_callback, this);
    global_traj_sub_ = nh_.subscribe("/optimal_traj", 1, &ModelPredictive::globalTrajCallback, this);
    global_traj1_sub_ = nh_.subscribe("/local_traj", 1, &ModelPredictive::globalTrajCallback1, this);
    global_traj2_sub_ = nh_.subscribe("/global_traj", 1, &ModelPredictive::globalTrajCallback2, this);
    current_pose_sub_ = nh_.subscribe("/current_pose", 1, &ModelPredictive::currentPoseCallback, this);
    overtaking_traj_pub_ = nh_.advertise<hmcl_msgs::Lane>("/local_traj1", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/over_traj_viz", 1);
    // behavior_factor_sub = nh_.subscribe('/behavior_factor', 1, &ModelPredictive::behaviorCallback, this);
}

void ModelPredictive::currentPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    qx = msg->pose.orientation.x;
    qy = msg->pose.orientation.y;
    qz = msg->pose.orientation.z;
    qw = msg->pose.orientation.w;
    x = msg->pose.position.x;
    pose_x=msg->pose.position.x;
    y = msg->pose.position.y;
    pose_y=msg->pose.position.y;
    // std::cout << pose_x << ", " << pose_y << std::endl;
}
// void ModelPredictive::behaviorCallback(const hmcl_msgs::BehaviorFactor& msg){
//     sEgo = msg->sEgo 
// }

void ModelPredictive::detected_object_callback(const autoware_msgs::DetectedObjectArray::ConstPtr& msg) {
    // std::cout << "Object_detected" << std::endl;
    // std::cout << "Number of objects detected: " << msg->objects.size() << std::endl; // 오브젝트 개수 출력

    target_veh_dic_x.clear();
    target_veh_dic_y.clear();
    obj.clear();
    objects_data = *msg;  
    for (const auto& obj : msg->objects) {
        int obj_key = obj.id; // Using ID as key

        target_velocity_x[obj_key] = obj.velocity.linear.x;
        target_velocity_y[obj_key] = obj.velocity.linear.y;
        target_x[obj_key] = obj.pose.position.x;
        target_y[obj_key] = obj.pose.position.y;
        target_orientation_x[obj_key] = obj.pose.orientation.x;
        target_orientation_y[obj_key] = obj.pose.orientation.y;
        target_orientation_z[obj_key] = obj.pose.orientation.z;
        target_orientation_w[obj_key] = obj.pose.orientation.w;
        target_yaw_veh[obj_key] = obj.velocity.angular.z;

        detected_object_kinematic(obj_key); // Custom function to handle object kinematics based on label or other identifiers
        
        // Additional logic for velocity history and LSTM model might be included here.
    }
    if (global_traj_available){
    compute_local_path();
    compute_local_path1();
    }
    Intuitive_Artificial_potential_field_2(); // Assuming another method handles potential field calculation
}
double ModelPredictive::distance(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow((x2 - x1), 2) + std::pow((y2 - y1), 2));
}

void ModelPredictive::detected_object_kinematic(int obj) {
    double model_prediction_x = target_x[obj];
    double target_vel_x = target_velocity_x[obj];
    double target_yaw = target_yaw_veh[obj];

    double model_prediction_y = target_y[obj];
    double target_vel_y = target_velocity_y[obj];
    double target_angular_z = target_angular_z_veh[obj];

    std::vector<double> predicted_x, predicted_y;

    for (int i = 0; i < model_predicted_num; ++i) {
        predicted_x.push_back(model_prediction_x);
        model_prediction_x += target_vel_x;
        target_yaw += target_angular_z;
        target_vel_x = pow((pow(target_vel_x,2)+pow(target_vel_y,2)),1/2) * std::cos(target_yaw);
    }
    target_veh_dic_x[obj] = predicted_x;

    for (int j = 0; j < model_predicted_num; ++j) {
        predicted_y.push_back(model_prediction_y);
        model_prediction_y += target_vel_y;
        target_yaw += target_angular_z;  // Note: This recalculates target_yaw; ensure this is the desired behavior
        target_vel_y = pow((pow(target_vel_x,2)+pow(target_vel_y,2)),1/2) * std::sin(target_yaw);
    }
    target_veh_dic_y[obj] = predicted_y;
}

void ModelPredictive::globalTrajCallback1(const hmcl_msgs::Lane::ConstPtr& data) {
    

    // Clear the vectors before populating them
    cx.clear();
    cy.clear();
    cqx.clear();
    cqy.clear();
    cqz.clear();
    cqw.clear();
    local_points.clear();  // 추가된 local_points 벡터 초기화

    // Iterate through the waypoints and populate the vectors
    for (const auto& waypoint : data->waypoints) {
        cx.push_back(waypoint.pose.pose.position.x);
        cy.push_back(waypoint.pose.pose.position.y);
        cqx.push_back(waypoint.pose.pose.orientation.x);
        cqy.push_back(waypoint.pose.pose.orientation.y);
        cqz.push_back(waypoint.pose.pose.orientation.z);
        cqw.push_back(waypoint.pose.pose.orientation.w);
    }

    // Populate local_points with x and y positions
    for (const auto& waypoint : data->waypoints) {
        local_points.push_back({waypoint.pose.pose.position.x, waypoint.pose.pose.position.y});
    }
    // std::cout << "local_points size: " << local_points.size() << std::endl;
}
int ModelPredictive::calculate_distance_pose2local(){
  float min_dist = 1000.0;
  float dist = 1000.0;
  int min_idx = 0;

  for(int i=0; i<local_lane.waypoints.size(); i++){
    dist = distance(local_lane.waypoints[i].pose.pose.position.x,local_lane.waypoints[i].pose.pose.position.y,pose_x,pose_y);
    if(min_dist > dist){
      min_dist = dist;
      min_idx=i;
    }
  }

  return min_idx;
}

int ModelPredictive::calculate_distance_pose2local_1(){
  float min_dist = 1000.0;
  float dist = 1000.0;
  int min_idx = 0;

  for(int i=0; i<local_lane1.waypoints.size(); i++){
    dist = distance(local_lane1.waypoints[i].pose.pose.position.x,local_lane1.waypoints[i].pose.pose.position.y,pose_x,pose_y);
    if(min_dist > dist){
      min_dist = dist;
      min_idx=i;
    }
  }

  return min_idx;
}
// void ModelPredictive::globalTrajCallback(const hmcl_msgs::LaneArray::ConstPtr& msg) {
//     // Clear previous data
//     std::cout << "global_traj" << std::endl;
//     cx.clear();
//     cy.clear();
//     cqx.clear();
//     cqy.clear();
//     cqz.clear();
//     cqw.clear();
//     optimal_points.clear(); 
//     // Iterate through all lanes and waypoints in the received message
//     for (const auto& lane : msg->lanes) {
//         for (const auto& waypoint : lane.waypoints) {
//             cx.push_back(waypoint.pose.pose.position.x);
//             cy.push_back(waypoint.pose.pose.position.y);
//             cqx.push_back(waypoint.pose.pose.orientation.x);
//             cqy.push_back(waypoint.pose.pose.orientation.y);
//             cqz.push_back(waypoint.pose.pose.orientation.z);
//             cqw.push_back(waypoint.pose.pose.orientation.w);
//         }
//     }
//     for (const auto& lane : msg->lanes) {
//         for (const auto& waypoint : lane.waypoints) {
//             optimal_points.push_back({waypoint.pose.pose.position.x, waypoint.pose.pose.position.y});
//         }
//     }
//     std::cout << "local_points size: " << local_points.size() << std::endl;
//     compute_local_path()
// }
void ModelPredictive::globalTrajCallback(const hmcl_msgs::LaneArray::ConstPtr& lane_msg){ 
    // std::cout << "optimal_traj" << std::endl;
    cx.clear();
    cy.clear();
    cqx.clear();
    cqy.clear();
    cqz.clear();
    cqw.clear();
    optimal_points.clear();
    global_lane_array = *lane_msg;

    global_traj_available = true;
    // Iterate through all lanes and waypoints in the received message
    for (int i=0; i<    global_lane_array.lanes[0].waypoints.size(); i++) {
            cx.push_back(    global_lane_array.lanes[0].waypoints[i].pose.pose.position.x);
            cy.push_back(    global_lane_array.lanes[0].waypoints[i].pose.pose.position.y);
            cqx.push_back(    global_lane_array.lanes[0].waypoints[i].pose.pose.orientation.x);
            cqy.push_back(    global_lane_array.lanes[0].waypoints[i].pose.pose.orientation.y);
            cqz.push_back(    global_lane_array.lanes[0].waypoints[i].pose.pose.orientation.z);
            cqw.push_back(    global_lane_array.lanes[0].waypoints[i].pose.pose.orientation.w);
        }
    for (const auto& lane : lane_msg->lanes) {
        for (const auto& waypoint : lane.waypoints) {
            optimal_points.push_back({waypoint.pose.pose.position.x, waypoint.pose.pose.position.y});
        }
    }

    
    //  std::cout << cx.size() << std::endl;
    
}
void ModelPredictive::globalTrajCallback2(const hmcl_msgs::LaneArray::ConstPtr& lane_msg){ 
    // std::cout << "global_traj" << std::endl;
    global_cx.clear();
    global_cy.clear();
    global_points.clear();
    global_lane_array1 = *lane_msg;
    // Iterate through all lanes and waypoints in the received message
    for (int i=0; i<    global_lane_array1.lanes[1].waypoints.size(); i++) {
            global_cx.push_back(    global_lane_array1.lanes[1].waypoints[i].pose.pose.position.x);
            global_cy.push_back(    global_lane_array1.lanes[1].waypoints[i].pose.pose.position.y);
        }
    for (const auto& lane : global_lane_array1.lanes) {
        for (const auto& waypoint : lane.waypoints) {
            global_points.push_back({waypoint.pose.pose.position.x, waypoint.pose.pose.position.y});
        }
    }

    
    // std::cout << global_cx.size() << std::endl;
}
void ModelPredictive::compute_local_path(){
//   std::cout << "compute_local_path" << std::endl;
  int local_size = 100;
  local_lane = global_lane_array.lanes[0];
  int minidx = calculate_distance_pose2local();
  local_lane.waypoints.clear();
  local_lane.header.frame_id = "map";
  local_lane.header.stamp = ros::Time::now();
  for(int i=minidx+1; i<global_lane_array.lanes[0].waypoints.size(); i++){
    if(global_lane_array.lanes[0].waypoints.size()<minidx+2){
      ROS_INFO("LACK OF POINTS FOR LOCAL");
      break;
    }
    hmcl_msgs::Waypoint wp;
    double wp1x = global_lane_array.lanes[0].waypoints[i-1].pose.pose.position.x;
    double wp1y = global_lane_array.lanes[0].waypoints[i-1].pose.pose.position.y;
    double wp2x = global_lane_array.lanes[0].waypoints[i].pose.pose.position.x;
    double wp2y = global_lane_array.lanes[0].waypoints[i].pose.pose.position.y;
    double dist = distance(wp1x,wp1y,wp2x,wp2y);
    if(dist > 0.5){
      int n = static_cast<int>(dist/0.5);
      // ROS_INFO("ADD %d WAYPOINTS", n);
      for(int j = 0; j < n-1; j++){
        wp =     global_lane_array.lanes[0].waypoints[i-1];
        wp.pose.pose.position.x = wp1x+(wp2x-wp1x)/n*(j+1);
        wp.pose.pose.position.y = wp1y+(wp2y-wp1y)/n*(j+1);
        local_lane.waypoints.push_back(wp);
        if(local_lane.waypoints.size()>local_size) break;        
      }
    }
    wp =     global_lane_array.lanes[0].waypoints[i];
    local_lane.waypoints.push_back(wp);
    if(local_lane.waypoints.size()>local_size) break;        

  }
//   ROS_INFO("local size: %zu",local_lane.waypoints.size());
  local_lane_to_local_points();
}
void ModelPredictive::compute_local_path1(){
//   std::cout << "compute_local_path" << std::endl;
  int local_size = 100;
  local_lane1 = global_lane_array1.lanes[1];
  int minidx = calculate_distance_pose2local_1();
  local_lane1.waypoints.clear();
  local_lane1.header.frame_id = "map";
  local_lane1.header.stamp = ros::Time::now();
  for(int i=minidx+1; i<global_lane_array1.lanes[1].waypoints.size(); i++){
    if(global_lane_array1.lanes[1].waypoints.size()<minidx+2){
      ROS_INFO("LACK OF POINTS FOR LOCAL");
      break;
    }
    hmcl_msgs::Waypoint wp;
    double wp1x = global_lane_array1.lanes[1].waypoints[i-1].pose.pose.position.x;
    double wp1y = global_lane_array1.lanes[1].waypoints[i-1].pose.pose.position.y;
    double wp2x = global_lane_array1.lanes[1].waypoints[i].pose.pose.position.x;
    double wp2y = global_lane_array1.lanes[1].waypoints[i].pose.pose.position.y;
    double dist = distance(wp1x,wp1y,wp2x,wp2y);
    if(dist > 0.5){
      int n = static_cast<int>(dist/0.5);
      // ROS_INFO("ADD %d WAYPOINTS", n);
      for(int j = 0; j < n-1; j++){
        wp =     global_lane_array1.lanes[1].waypoints[i-1];
        wp.pose.pose.position.x = wp1x+(wp2x-wp1x)/n*(j+1);
        wp.pose.pose.position.y = wp1y+(wp2y-wp1y)/n*(j+1);
        local_lane1.waypoints.push_back(wp);
        if(local_lane1.waypoints.size()>local_size) break;        
      }
    }
    wp =     global_lane_array1.lanes[1].waypoints[i];
    local_lane1.waypoints.push_back(wp);
    if(local_lane1.waypoints.size()>local_size) break;        

  }
//   ROS_INFO("local size: %zu",local_lane1.waypoints.size());
  local_lane_to_local_points1();
}
void ModelPredictive::local_lane_to_local_points() {

    local_points.clear();
    
    for (const auto& waypoint : local_lane.waypoints) { // 실제 사용 중인 waypoints 벡터로 변경해야 함
        local_points.push_back({waypoint.pose.pose.position.x, waypoint.pose.pose.position.y});
    }
    
    // std::cout << "local_points size: " << local_points.size() << std::endl;
}
void ModelPredictive::local_lane_to_local_points1() {

    local_points1.clear();
    
    for (const auto& waypoint : local_lane1.waypoints) { // 실제 사용 중인 waypoints 벡터로 변경해야 함
        local_points1.push_back({waypoint.pose.pose.position.x, waypoint.pose.pose.position.y});
    }
    
    // std::cout << "local_points size: " << local_points.size() << std::endl;
}
std::pair<double, double> ModelPredictive::cartesian_to_frenet(const std::vector<std::vector<double>>& centerline, const std::vector<double>& point) {
    Eigen::MatrixXd centerline_mat(centerline.size(), 2);
    for (size_t i = 0; i < centerline.size(); ++i) {
        centerline_mat(i, 0) = centerline[i][0];
        centerline_mat(i, 1) = centerline[i][1];
    }

    Eigen::MatrixXd diffs = centerline_mat.bottomRows(centerline.size() - 1) - centerline_mat.topRows(centerline.size() - 1);
    Eigen::VectorXd dists = diffs.rowwise().norm();
    Eigen::VectorXd arclength(dists.size() + 1);
    arclength(0) = 0.0;
    for (int i = 0; i < dists.size(); ++i) {
        arclength(i + 1) = arclength(i) + dists(i);
    }

    Eigen::Vector2d point_vec(point[0], point[1]);
    double min_dist = std::numeric_limits<double>::infinity();
    double s = 0, l = 0;

    for (int i = 0; i < diffs.rows(); ++i) {
        Eigen::Vector2d p1 = centerline_mat.row(i);
        Eigen::Vector2d p2 = centerline_mat.row(i + 1);

        Eigen::Vector2d line_vec = p2 - p1;
        Eigen::Vector2d point_to_p1 = point_vec - p1;
        double line_len = line_vec.norm();
        double proj_length = point_to_p1.dot(line_vec) / line_len;
        Eigen::Vector2d proj_point = p1 + (proj_length / line_len) * line_vec;

        double dist = (point_vec - proj_point).norm();

        // 벡터 외적을 사용하여 방향 판단
        Eigen::Vector2d perp_vec(-line_vec.y(), line_vec.x());  // line_vec에 수직인 벡터
        double side = point_to_p1.dot(perp_vec);  // 점이 선분의 왼쪽에 있는지 오른쪽에 있는지를 결정

        if (dist < min_dist) {
            min_dist = dist;
            s = arclength(i) + proj_length;
            l = (side < 0) ? -dist : dist;  // 왼쪽이면 음수, 오른쪽이면 양수
        }
    }

    return std::make_pair(s, l);
}


std::pair<double, double> ModelPredictive::frenet_to_cartesian(const std::vector<std::vector<double>>& centerline, double s, double l) {
    Eigen::MatrixXd centerline_mat(centerline.size(), 2);
    for (size_t i = 0; i < centerline.size(); ++i) {
        centerline_mat(i, 0) = centerline[i][0];
        centerline_mat(i, 1) = centerline[i][1];
    }

    Eigen::MatrixXd diffs = centerline_mat.bottomRows(centerline.size() - 1) - centerline_mat.topRows(centerline.size() - 1);
    Eigen::VectorXd dists = diffs.rowwise().norm();
    Eigen::VectorXd arclength(dists.size() + 1);
    arclength(0) = 0.0;
    for (int i = 0; i < dists.size(); ++i) {
        arclength(i + 1) = arclength(i) + dists(i);
    }

    int segment_index = std::lower_bound(arclength.data(), arclength.data() + arclength.size(), s) - arclength.data() - 1;
    if (segment_index < 0) {
        segment_index = 0;
    } else if (segment_index >= centerline.size() - 1) {
        segment_index = centerline.size() - 2;
    }

    Eigen::Vector2d p1 = centerline_mat.row(segment_index);
    Eigen::Vector2d p2 = centerline_mat.row(segment_index + 1);

    Eigen::Vector2d segment_vector = p2 - p1;
    double segment_length = dists(segment_index);
    Eigen::Vector2d segment_unit_vector = segment_vector / segment_length;

    Eigen::Vector2d base_point = p1 + segment_unit_vector * (s - arclength(segment_index));

    Eigen::Vector2d normal_vector(-segment_unit_vector(1), segment_unit_vector(0));

    Eigen::Vector2d cartesian_point = base_point + normal_vector * l;

    return std::make_pair(cartesian_point(0), cartesian_point(1));
}

void ModelPredictive::OG_PUB() {
    hmcl_msgs::Lane trajectory;
    int i = 0;
    // std::cout<<cx.size()<<std::endl;

    while (i < cx.size()) {
        hmcl_msgs::Waypoint waypoint;
        waypoint.pose.pose.position.x = cx[i];
        waypoint.pose.pose.position.y = cy[i];
        waypoint.pose.pose.orientation.x = cqx[i];
        waypoint.pose.pose.orientation.y = cqy[i];
        waypoint.pose.pose.orientation.z = cqz[i];
        waypoint.pose.pose.orientation.w = cqw[i];
        trajectory.waypoints.push_back(waypoint);
        i++;
    }
    overtaking_traj_pub_.publish(trajectory);

    // Visualize the local path using markers
    visualization_msgs::MarkerArray marker_array = visualize_local_path(trajectory.waypoints);
    marker_pub_.publish(marker_array);
    // std::cout << "work2222" << std::endl;
}

visualization_msgs::MarkerArray ModelPredictive::visualize_local_path(const std::vector<hmcl_msgs::Waypoint>& waypoints) {
    visualization_msgs::MarkerArray marker_array;
    for (size_t i = 0; i < waypoints.size(); ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "local_path";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position = waypoints[i].pose.pose.position;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0;
        marker_array.markers.push_back(marker);
    }
    return marker_array;
}

void ModelPredictive::Bezier_Curve() {
    // std::cout << "work!" << std::endl;
    // std::cout << repulsed_potential_field_point_x.size() << std::endl;

    int num = repulsed_potential_field_point_x.size();
    printf("%d", num);
    std::vector<int> indices = {(num - 1) / 5, (num - 1) * 2 / 5,(num - 1) / 5*3, (num - 1) * 4 / 5};

    first_point = {pose_x,pose_y};
    second_point = {repulsed_potential_field_point_x[indices[0]], repulsed_potential_field_point_y[indices[0]]};
    third_point = {repulsed_potential_field_point_x[indices[1]], repulsed_potential_field_point_y[indices[1]]};
    fourth_point = {repulsed_potential_field_point_x[indices[2]], repulsed_potential_field_point_y[indices[2]]};
    fifth_point = {repulsed_potential_field_point_x[indices[3]], repulsed_potential_field_point_y[indices[3]]};
    sixth_point = {repulsed_potential_field_point_x[num - 1], repulsed_potential_field_point_y[num - 1]};
    if (sixth_point.size() == 2) {
        // std::cout << "second_point: (" << second_point[0] << ", " << second_point[1] << ")" << std::endl;
        // std::cout << "fourth_point: (" << fourth_point[0] << ", " << fourth_point[1] << ")" << std::endl;
        // std::cout << "sixth_point: (" << sixth_point[0] << ", " << sixth_point[1] << ")" << std::endl;
    } else {
        std::cerr << "Unexpected size of sixth_point vector." << std::endl;
    }
    // std::cout << indices[0] << std::endl;
    // std::cout << indices[1] << std::endl;
    // std::cout << indices[2] << std::endl;
    // std::cout << indices[3] << std::endl;
    B = calc_curve(100);

    Trajectory_Generation();
}


std::vector<std::pair<double, double>> ModelPredictive::calc_curve(int num_points) {
    std::vector<std::pair<double, double>> curve_points;
    for (int i = 0; i <= num_points; ++i) {
        double t = static_cast<double>(i) / num_points;
        double x = std::pow(1 - t, 5) * first_point[0] +
                   5 * std::pow(1 - t, 4) * t * second_point[0] +
                   10 * std::pow(1 - t, 3) * std::pow(t, 2) * third_point[0] +
                   10 * std::pow(1 - t, 2) * std::pow(t, 3) * fourth_point[0] +
                   5 * (1 - t) * std::pow(t, 4) * fifth_point[0] +
                   std::pow(t, 5) * sixth_point[0];
        
        double y = std::pow(1 - t, 5) * first_point[1] +
                   5 * std::pow(1 - t, 4) * t * second_point[1] +
                   10 * std::pow(1 - t, 3) * std::pow(t, 2) * third_point[1] +
                   10 * std::pow(1 - t, 2) * std::pow(t, 3) * fourth_point[1] +
                   5 * (1 - t) * std::pow(t, 4) * fifth_point[1] +
                   std::pow(t, 5) * sixth_point[1];

        curve_points.emplace_back(x, y);
    }
    return curve_points;
}

void ModelPredictive::Trajectory_Generation() {
    hmcl_msgs::Lane trajectory_msg;  // Trajectory message to be published

    // Ensure B has at least two points to calculate orientation
    if (B.size() < 2) {
        ROS_WARN("Not enough points in B to generate a trajectory.");
        return;
    }

    for (size_t i = 0; i < B.size(); ++i) {
        hmcl_msgs::Waypoint wp;
        wp.pose.pose.position.x = B[i].first;  // Assuming B is a vector of pairs or a similar structure
        wp.pose.pose.position.y = B[i].second;
        wp.pose.pose.position.z = 0.0;  // Assuming z is constant

        // Calculate orientation based on the next waypoint
        if (i < B.size() - 1) {
            double delta_x = B[i + 1].first - B[i].first;
            double delta_y = B[i + 1].second - B[i].second;
            double yaw = std::atan2(delta_y, delta_x);

            // Convert yaw to quaternion
            tf2::Quaternion quat;
            quat.setRPY(0, 0, yaw);  // Roll and Pitch are 0, Yaw is calculated

            wp.pose.pose.orientation = tf2::toMsg(quat);
        } else {
            // For the last waypoint, you might need to set a default orientation
            wp.pose.pose.orientation.w = 1.0;  // Default to no rotation (identity quaternion)
        }

        trajectory_msg.waypoints.push_back(wp);
    }

    // Publish the trajectory
    overtaking_traj_pub_.publish(trajectory_msg);

    // Visualize the local path using markers
    visualization_msgs::MarkerArray marker_array = visualize_local_path(trajectory_msg.waypoints);
    marker_pub_.publish(marker_array);

    // ROS_INFO("Trajectory generation and publishing completed.");
}
void ModelPredictive::Intuitive_Artificial_potential_field_2() {
    // track_boundary:10.2
    // std::cout << "Artificial_potential_field"<<std::endl;
    auto start_time = std::clock();
    double gain = 10;
    double target_velocity = 300;
    double sigma = 1 / target_velocity;
    double radius = round(std::sqrt(1 / sigma) + 1) * 10000000000;
    std::map<int, bool> direction_list;
    std::vector<double> s1_list;
    std::vector<double> d1_list;
    repulsed_potential_field_point_x.clear();  // 이전 값을 지우기 위해 초기화
    repulsed_potential_field_point_y.clear(); 

    double s_ego, d_ego;

    // Cartesian to Frenet conversion for each object
    std::tie(s_ego, d_ego) = cartesian_to_frenet(local_points, {pose_x,pose_y});

    for (const auto& obj : objects_data.objects) {
        int obj_key = obj.id;
        double s1, d1;
        // double s_ego, d_ego;

        // // Cartesian to Frenet conversion for each object
        // std::tie(s_ego, d_ego) = cartesian_to_frenet(local_points, {pose_x,pose_y});
        std::tie(s1, d1) = cartesian_to_frenet(local_points, {target_veh_dic_x[obj_key][0], target_veh_dic_y[obj_key][0]});
        if (std::abs(d1) > 2 || s1-s_ego<-3){

        }
        else {
            s1_list.push_back(s1);
            d1_list.push_back(d1);
            if (d1 > 0){
            direction_list[s1_list.size()]=true;

        }
        else{
            direction_list[s1_list.size()]=false;
        }
        }
        
        // s1_list.push_back(s1);
        // d1_list.push_back(d1);
    }
    // std::cout << "d1_list contents: ";
    for (const auto& value : d1_list) {
        // std::cout << value << " ";
    }
    // std::cout << "local_points: " << std::endl;
    // for (size_t i = 0; i < local_points.size(); ++i) {
    //     std::cout << "[ ";
    //     for (size_t j = 0; j < local_points[i].size(); ++j) {
    //         std::cout << local_points[i][j] << " ";
    //     }
    //     std::cout << "]" << std::endl;
    // }
    // // std::cout<< local_points.size()<< std::endl;
    // std::cout<< local_points1.size()<< std::endl;
    if (local_points.size() == 0 || local_points1.size() == 0){
        OG_PUB();
    }
    
    else{
        for (const auto& j : local_points) {
            double s, d;
            double s_global_and_opt, d_global_and_opt;
            std::tie(s, d) = cartesian_to_frenet(local_points, j);
            std::tie(s_global_and_opt, d_global_and_opt) = cartesian_to_frenet(local_points1, j);
            for (size_t i = 0; i < s1_list.size() && i < d1_list.size(); ++i) {
                double s1 = s1_list[i];
                double d1 = d1_list[i];
                bool direction = direction_list[i];
                // std::cout << "d1" << d1<< std::endl;
                for (int i = 0; i < model_predicted_num; ++i) {
                    if (direction) {
                        d=d1;
                        d -= gain * std::sqrt(std::max(0.0, 1 - sigma * std::pow(s - s1, 2)));
                    } else {
                        d=d1;
                        d += gain * std::sqrt(std::max(0.0, 1 - sigma * std::pow(s - s1, 2)));
                    }
                }
        
            // std::cout<< "calculating"<< std::endl;
            // std::tie(s1, d1) = cartesian_to_frenet(local_points, {target_veh_dic_x[obj_key][0], target_veh_dic_y[obj_key][0]});
            if (d >  4-d_global_and_opt){
                d = 4-d_global_and_opt;
            }
            if (d <  -4-d_global_and_opt){
                d = -4-d_global_and_opt;
            }
            }
            double repulsed_s = s;
            double repulsed_d = d;
            // std::cout << "global_d!!!!!!!!!!!!!!" << d_global_and_opt<< std::endl;
            // std::cout << "d!!!!!!!!!!!!!!" << repulsed_d << std::endl;
            // Frenet to Cartesian conversion for the repulsed point
            double repulsed_x, repulsed_y;
            std::tie(repulsed_x, repulsed_y) = frenet_to_cartesian(local_points, repulsed_s, repulsed_d);
            repulsed_potential_field_point_x.push_back(repulsed_x);
            repulsed_potential_field_point_y.push_back(repulsed_y);
        }
        // std::cout << repulsed_potential_field_point_x.size() << std::endl;

        // if (objects_data.objects.empty()) {
        if (d1_list.empty()) {
            if (init_obs){
                if(std::abs(d_ego) < 0.5){
                    OG_PUB();
                    std::cout << " BACK_OG_PUB!!!!!!!!!!!!!"<<std::endl;
                    init_obs = false;
                }
                else{
                    Bezier_Curve();
                    std::cout << " SMOOTHING_OG_PUB!!!!!!!!!!!!!"<<std::endl;

                }
            }
            else{
                OG_PUB();
                std::cout << " OG_PUB!!!!!!!!!!!!!"<<std::endl;

            }
        }
        else{
            Bezier_Curve();
            std::cout << " Bezier!!!!!!!!!!!!!"<<std::endl;
            init_obs = true;
        }

        auto end_time = std::clock();
        // std::cout << "Execution time: " << (end_time - start_time) / CLOCKS_PER_SEC << " seconds" << std::endl;
        // std::cout << start_time << std::endl;
        // std::cout << end_time << std::endl;
    }
}
void ModelPredictive::run() {
    ros::Rate loop_rate(10);  // Adjust the rate as necessary
    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ModelPredictive");
    ModelPredictive model_predictive;
    model_predictive.run();
    return 0;
}
