""" Shows that we can easily request services directly from gazebo
Without going by ROS2 bridges!!
https://gazebosim.org/api/transport/13/python.html
"""

from scipy.spatial.transform import Rotation


from gz.transport13 import Node ;
import numpy as np
from gz.msgs10.empty_pb2 import Empty ;
from gz.msgs10.scene_pb2 import Scene ;
from gz.msgs10.image_pb2 import Image ;
from gz.msgs10.twist_pb2 import Twist ;
from gz.msgs10.boolean_pb2 import Boolean ;
from gz.msgs10.quaternion_pb2 import Quaternion ;
from gz.msgs10.pose_pb2 import Pose ;
from gz.msgs10.pose_v_pb2 import Pose_V ;
from gz.msgs10.world_control_pb2 import WorldControl ;


import time ;


def posecb(msg: Pose_V):
  for m in msg.pose:
    if m.name == "3pi_robot":
      print ("model", m.name, m.position) ;


def trigger_pause(self, pause):
        tmp = WorldControl() ;
        tmp.step = False ;
        tmp.multi_step=0 ;
        tmp.pause = pause
        tmp.reset.all = False; 
        tmp.reset.time_only = False; 
        tmp.reset.model_only = False; 

        result = False ;
        while result == False:
          result, response = self.request("/world/race_tracks_world/control", tmp, WorldControl, Boolean, 100) ;
        print("pause", pause) ;



def set_entity_pose_request(node, name, position, orientation):
         pos_req = Pose() ;
         pos_req.name = name
         pos_req.position.x = position[0]
         pos_req.position.y = position[1]
         pos_req.position.z = position[2]
         #print("!!", pos_req.orientation.header)
         pos_req.orientation.x = orientation[0]
         pos_req.orientation.y = orientation[1]
         pos_req.orientation.z = orientation[2]
         pos_req.orientation.w = orientation[3]
         print("Pose request", pos_req)

         result = False ;
         while result == False:
           result, response =  node.request( "/world/race_tracks_world/set_pose",  pos_req, Pose, Boolean, 200) ;
           print(result, response.data)
           if response.data == True: break ;


def main():
    node = Node()

    trigger_pause(node, False) ;
    print("set pose service")

    orientation = Rotation.from_euler('xyz',[0.,0.,15],degrees=True).as_quat(canonical=False)
    orientation = [float(o) for o in orientation] ;

    set_entity_pose_request(node, "3pi_robot",[3.,3.,10.],orientation) ;
    time.sleep(1.0)

    # call service
    ## querying names of all objects in the world
    service_name = "/world/race_tracks_world/scene/info"
    request = Empty()
    timeout = 5000 
    result, response = node.request(service_name, request, Empty, Scene, timeout)
    print("Result:", result)
    print(len(response.model))
    for m in response.model:
      print(m.name, m.id) ;


    ## publish Twist msg
    pubObj = node.advertise("/vehicle/motor", Twist)
    twistObj = Twist() ;
    pubObj.publish(twistObj)

    # subscribe to camera message
    if node.subscribe(Pose_V, "/world/race_tracks_world/dynamic_pose/info", posecb):
      print("subscribed!") ;

    while True:
      time.sleep(0.01) ;
  
    



main()
