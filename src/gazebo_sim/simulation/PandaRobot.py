import math
import PyKDL as kdl
import numpy as np
from gazebo_sim.utils.math import Transform, Vector3, Quaternion

class JointLimit:
    def __init__(self, lower: float, upper: float):
        """
        Creates an instance of the JointLimit class.

        Args:
            lower (float): The lower limit of the joint in radians.
            upper (float): The upper limit of the joint in radians.
        """
        self.lower = lower
        self.upper = upper
    
class Joint:
    def __init__(self, rotation: float = 0.0, joint_limit: JointLimit = None):
        """
        Creates an instance of the Joint class.

        Args:
            rotation (float): The rotation value of the joint in degrees or radians.
            joint_limit (JointLimit): An JointLimit instance defining the joint's limits.
        """
        # If no rotation was passed, set default rotation (0.0)
        self.rotation = rotation
        # If no limit was passed, set default limits (-π to π)
        self.limit = joint_limit if joint_limit else JointLimit(-math.pi, math.pi)

class PandaRobot():
    def __init__(self,actions):
        self.actions = actions
        self.hand = Transform()
        self.joint_amount = 7;
        self.joints = []
        self.limits = []
        self.limits.append(JointLimit(-2.897246558310587, 2.897246558310587))
        self.limits.append(JointLimit(-1.762782544514273, 1.762782544514273))
        self.limits.append(JointLimit(-2.897246558310587, 2.897246558310587))
        self.limits.append(JointLimit(-3.07177948351002, -0.06981317007977318))
        self.limits.append(JointLimit(-2.897246558310587, 2.897246558310587))
        self.limits.append(JointLimit(-0.0174532925199433, 3.752457891787809))
        self.limits.append(JointLimit(-2.897246558310587, 2.897246558310587))

        for iter in range(self.joint_amount):
            self.joints.append(Joint(0,self.limits[iter]))
        self.create_chain()

        self.debug = False


    def get_joint_rotation_with_num(self,joint,amount):
        lower = self.joints[joint].limit.lower
        upper = self.joints[joint].limit.upper

        return lower + (upper - lower) * amount
    
    def get_current_joint_rotations(self):
        rotations = []
        for joint in self.joints:
            rotations.append(joint.rotation)
        return rotations

    def hand_callback(self,msg):
        """
        Callback function to process pose data for the robot hand.
        """
        for entity in msg.pose:
            entity_name = entity.name
            entity_id = entity.id

            # Hand (panda_hand)
            if entity_name == "panda_hand" and entity_id == 43:
                self.hand.position = Vector3(entity.position.x, entity.position.y, entity.position.z)
                self.hand.rotation = Quaternion(entity.orientation.w, entity.orientation.x, entity.orientation.y, entity.orientation.z)

    def joints_callback(self,msg):
        """
        Callback function to process data of the robot joints.
        """

        for entity in msg.joint:
            entity_name = entity.name

            # Hand (panda_hand)
            if entity_name.startswith("panda_joint") and entity_name != "panda_joint_world":
                # Get Index from "panda_jointX"
                joint_index = int(entity_name.split("_joint")[-1]) - 1

                # get and set rotation
                rotation = entity.axis1.position
                self.joints[joint_index].rotation = rotation

                # get and set joint limit
                limit_lower = entity.axis1.limit_lower
                limit_upper = entity.axis1.limit_upper
                self.joints[joint_index].limit.lower = limit_lower
                self.joints[joint_index].limit.upper = limit_upper 

    def create_chain(self):
        self.kdl_chain = kdl.Chain()
        joint1 = kdl.Joint(kdl.Joint.Fixed)
        frame1 = kdl.Frame(kdl.Vector(0.0, 0.0, 0.333))
        self.kdl_chain.addSegment(kdl.Segment(joint1, frame1))

        # Segment 2
        joint2 = kdl.Joint(kdl.Joint.RotZ)
        frame2 = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, -np.pi / 2)) * kdl.Frame(kdl.Vector(0.0, 0.0, 0.0))
        self.kdl_chain.addSegment(kdl.Segment(joint2, frame2))

        # Segment 3
        joint3 = kdl.Joint(kdl.Joint.RotZ)
        frame3 = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, np.pi / 2)) * kdl.Frame(kdl.Vector(0.0, 0.0, 0.316))
        self.kdl_chain.addSegment(kdl.Segment(joint3, frame3))

        # Segment 4
        joint4 = kdl.Joint(kdl.Joint.RotZ)
        frame4 = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, np.pi / 2)) * kdl.Frame(kdl.Vector(0.0825, 0.0, 0.0))
        self.kdl_chain.addSegment(kdl.Segment(joint4, frame4))

        # Segment 5
        joint5 = kdl.Joint(kdl.Joint.RotZ)
        frame5 = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, -np.pi / 2)) * kdl.Frame(kdl.Vector(-0.0825, 0.0, 0.384))
        self.kdl_chain.addSegment(kdl.Segment(joint5, frame5))

        # Segment 6
        joint6 = kdl.Joint(kdl.Joint.RotZ)
        frame6 = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, np.pi / 2)) * kdl.Frame(kdl.Vector(0.0, 0.0, 0.0))
        self.kdl_chain.addSegment(kdl.Segment(joint6, frame6))

        # Segment 7
        joint7 = kdl.Joint(kdl.Joint.RotZ)
        frame7 = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, np.pi / 2)) * kdl.Frame(kdl.Vector(0.088, 0.0, 0.0))
        self.kdl_chain.addSegment(kdl.Segment(joint7, frame7))

        # End-Effector
        joint_ee = kdl.Joint(kdl.Joint.RotZ)
        frame_ee = kdl.Frame(kdl.Rotation.EulerZYX(0.0, 0.0, -np.pi)) * kdl.Frame(kdl.Vector(0.0, 0.0, -0.1))
        self.kdl_chain.addSegment(kdl.Segment(joint_ee, frame_ee))

    def chain_info(self):
        ret = f"Nr. Segments: {self.kdl_chain.getNrOfSegments()}"
        cumulative_position = kdl.Vector(0, 0, 0)  # Kumulative Höhe berechnen
        for i in range(self.kdl_chain.getNrOfSegments()):
            seg = self.kdl_chain.getSegment(i)
            pos = seg.getFrameToTip().p
            rot = seg.getFrameToTip().M.GetRPY()
            cumulative_position += pos  # Kumulativ addieren
            ret += f"\n\nSegment {i + 1}: Joint Type = {seg.getJoint().getType()}"
            ret += f"\nPosition = {cumulative_position}"
            ret += f"\nRotation (RPY) = {rot}"

    def compute_inverse_kinematics(self, target_position):
        if self.debug: print("Calculating Inverse Kinematics")
        # Number of Joints
        num_joints = self.kdl_chain.getNrOfJoints()
        # Set Weight matrix for Task Space (only consider position, ignore rotation)
        L = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])

        # Extract joint limits as arrays
        lower_limits = [joint.limit.lower for joint in self.joints]
        upper_limits = [joint.limit.upper for joint in self.joints]

        # Create PyKDL JntArray
        q_min = kdl.JntArray(num_joints)
        q_max = kdl.JntArray(num_joints)

        for i in range(num_joints):
            q_min[i] = lower_limits[i]
            q_max[i] = upper_limits[i]

        # Forward Kinematics Solver
        fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)

        # Inverse Velocity Solver
        ik_solver_vel = kdl.ChainIkSolverVel_wdls(self.kdl_chain)
        ik_solver_vel.setWeightTS(L)

        # Inverse Solver
        ik_solver = kdl.ChainIkSolverPos_NR_JL(self.kdl_chain, q_min, q_max, fk_solver, ik_solver_vel, maxiter=900, eps=1e-3)
        
        # Create target frame from target position
        target_frame = kdl.Frame(kdl.Vector(*target_position))

        # Start pose (Initial Guess)
        initial_guess = kdl.JntArray(num_joints)
        for i in range(len(self.joints)):
            # set initial guess to mean value between q_min and q_max
            initial_guess[i] = (q_min[i] + q_max[i]) / 2.0

        # Compute Inverse Kinematics
        result_joints = kdl.JntArray(num_joints)
        ret = ik_solver.CartToJnt(initial_guess, target_frame, result_joints)
        
        if ret >= 0:
            if self.debug: print("Inverse Kinematics found!")
            return [result_joints[i] for i in range(num_joints)]
        else:
            if self.debug: print("No Inverse Kinematics found!")
            return None

    def compute_inverse_kinematic_approx(self,target_position,eps=0.05):
        # Number of Joints
        num_joints = self.kdl_chain.getNrOfJoints()
        # Set Weight matrix for Task Space (only consider position, ignore rotation)
        L = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])

        # Extract joint limits as arrays
        lower_limits = [joint.limit.lower for joint in self.joints]
        upper_limits = [joint.limit.upper for joint in self.joints]

        # Create PyKDL JntArray
        q_min = kdl.JntArray(num_joints)
        q_max = kdl.JntArray(num_joints)

        for i in range(num_joints):
            q_min[i] = lower_limits[i]
            q_max[i] = upper_limits[i]

        # Levenberg–Marquardt IK with task-space tolerance (e.g., 3 cm)
        ik_solver = kdl.ChainIkSolverPos_LMA(self.kdl_chain, L, maxiter=500, eps=eps)

        # Create target frame from target position
        target_frame = kdl.Frame(kdl.Vector(*target_position))

        # Start pose (Initial Guess)
        initial_guess = kdl.JntArray(num_joints)
        for i in range(len(self.joints)):
            # set initial guess to mean value between q_min and q_max
            initial_guess[i] = (q_min[i] + q_max[i]) / 2.0

        # Compute Inverse Kinematics
        result_joints = kdl.JntArray(num_joints)
        ret = ik_solver.CartToJnt(initial_guess, target_frame, result_joints)
        
        if ret >= 0:
            if self.debug: print("Approximate Inverse Kinematics found!")
            return [result_joints[i] for i in range(num_joints)]
        else:
            if self.debug: print("No Approximate Inverse Kinematics found!")
            return None

    def compute_forward_kinematic(self, joints):
        num_joints = self.kdl_chain.getNrOfJoints()
        if len(joints)!= num_joints:
            raise ValueError(f"Expected {num_joints} joint angles, got {len(joints)}.")

        q = kdl.JntArray(num_joints)
        for i, ang in enumerate(joints):
            q[i] = float(ang)

        fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)

        frame = kdl.Frame()
        ret = fk_solver.JntToCart(q,frame,-1)

        if ret >= 0:
            if self.debug: print("Forward Kinematics found!")
            return [round(frame.p[0],3),round(frame.p[1],3),round(frame.p[2],3)]
        else:
            if self.debug: print("No Forward Kinematic found!")
            return None