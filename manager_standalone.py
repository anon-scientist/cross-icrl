"""
Demo skript, shows how to move the 3pi robot model in a running gazebo simulation "by hand" 
without reference to episodes or similar.
"""
from gazebo_sim.simulation.ThreePiManager import ThreePiManager, TwistAction
import time ;

def init():
  ec = {"debug": "yes", "contact_topic":"/vehicle/contact_sensor", "lidar":"/vehicle/lidar", "world_name":"/world/Forschungsprojekt_world", "robot_name":"3pi_robot_with_front_cam", "vehicle_prefix":"/vehicle", "camera_topic":"/vehicle/camera"} ; 
  m = ThreePiManager(ec) ;
  fwd = TwistAction("f", [0.4, 0.4]) ;
  stop = TwistAction("f", [0., 0.]) ;
  left = TwistAction("f", [0.6, 0.2]) ;
  right = TwistAction("f", [0.2, 0.6]) ;
  return m, fwd, stop, left, right ;

if __name__=="__main__":
  m, fwd, stop, left, right = init() ;
  for i in range(0,10):
    m.gz_perform_action(fwd) ;
    time.sleep(0.2) ;
    m.gz_perform_action(left) ;
    time.sleep(0.2) ;
    m.gz_perform_action(right) ;
    time.sleep(0.2) ;
    m.gz_perform_action(stop) ;
    time.sleep(0.2) ;

  
