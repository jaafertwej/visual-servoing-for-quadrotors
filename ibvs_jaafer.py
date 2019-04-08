from __future__ import division
import numpy as np
import scipy
from scipy import ndimage
import rospy
import tf

deg2rad = np.pi/180


from sensor_msgs.msg import Image
from mavros_msgs.msg import Corner
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import apriltag

from geometry_msgs.msg import Twist, PoseStamped
from quaternion import Quaternion
from dualQuaternion import DualQuaternion



class Camera():

    def __init__(self, d_image, depth, f, u0, v0):
        self.d_image = d_image
        self.depth = depth
        self.f = f
        self.u0 = u0
        self.v0 = v0
        self.K = np.array([[476.7030836014194, 0.0              , 512.0],
                           [0.0              , 476.7030836014194, 512.0],
                           [0.0              , 0.0              , 1.0  ]])

        self.bridge = CvBridge()
        self.detector = apriltag.Detector()
        rospy.init_node('image_listener')
        image_topic = "/iris/camera_red_iris/image_raw"
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.T_callback)
        rospy.sleep(1)

        self.dq_AR = self.dq
        rospy.Subscriber(image_topic, Image, self.image_callback)
        self.velocity_publisher = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        

        self.pos_publisher = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
        # self.dq_c = self.dq
        self.image_pub = rospy.Publisher('modified_image', Image, queue_size=10)


    def get_d_features(self, corners):
        # corners = np.linalg.pinv(self.K).dot(np.hstack((corners, np.ones((4,1)))).T).T
        m00, m10, m01 = self.moments(corners)
        self.a_star = m00
        x = m10
        y = m01

        an = self.depth
        xn = x * an
        yn = y * an

        s_d = np.array([xn, yn, an])

        return s_d


    def moments(self, corners):

        corners = np.linalg.pinv(self.K).dot(np.hstack((corners, np.ones((4,1)))).T).T
        corners = np.vstack((corners, corners[0]))
        m00 = 0
        m10 = 0
        m01 = 0

        for i in range(1, corners.shape[0]):
            m00 = m00 + ((corners[i-1, 0] * corners[i, 1]) - (corners[i, 0] * corners[i-1, 1]))
            m10 = m10 + (((corners[i-1, 0] * corners[i, 1]) - (corners[i, 0] * corners[i-1, 1])) * (corners[i-1, 0] + corners[i, 0]))
            m01 = m01 + (((corners[i-1, 0] * corners[i, 1]) - (corners[i, 0] * corners[i-1, 1])) * (corners[i-1, 1] + corners[i, 1]))
            
        a = m00/2
        if a < 0:
            x = (m10 * -1)/(6 * m00)
            y = (m01 * -1)/(6 * m00)

        else:
            x = m10/(6 * m00)
            y = m01/(6 * m00)
        # print('a, x, y: {},{},{}'.format(a, x, y))
        return a, x, y

    def get_features(self, corners):
        # corners = np.linalg.pinv(self.K).dot(np.hstack((corners, np.ones((4,1)))).T).T
        # corners = np.hstack((corners, np.ones((4,1))))
        # p = np.zeros_like(corners)
        a, x, y = self.moments(corners)


        an = self.depth * np.sqrt(self.a_star/a)
        xn = x * an
        yn = y * an

        s = np.array([xn, yn, an])

        # for i in range(p.shape[0]):
        #     p[i] = corners[i]/np.linalg.norm(corners[i])
        # q = np.sum(p, axis= 0)

        return s

    def image_callback(self, msg):

        # print("Received an image!")
        try:
            # Convert your ROS Image message to OpenCV2

            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.result = self.detector.detect(image2)


            corners_d = self.detector.detect(self.d_image)[0].corners
            # s_B = self.get_d_features(corners_d)
            s_B = np.linalg.pinv(self.K).dot(np.hstack((corners_d, np.ones((4,1)))).T)
            lambdaa = 5
            Hz = 20
            rate = rospy.Rate(Hz)
            my_dq = self.dq

            dt = 1./Hz # control sampling time

            if len(self.result) != 0:

                corners = self.result[0].corners
                for i in range(4):
                    cv2.circle(image, (int(corners[i, 0]), int(corners[i ,1])), 3, (0,0,255), -1)
                    cv2.circle(image, (int(corners_d[i, 0]), int(corners_d[i ,1])), 3, (0,255,0), -1)
                
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))  

                
                T = np.array([[ np.cos(1.57079) , 0, -np.sin(1.57079) ,  0   ],
                              [ 0                , 1, 0                 ,  0   ],
                              [ np.sin(1.57079) , 0, np.cos(1.57079)  ,  -0.03],
                              [ 0                , 0, 0                 ,  1   ]])

                Ls = Jacobian(s_B.T, 1)
                # s_A = self.get_features(corners)
                s_A = np.linalg.pinv(self.K).dot(np.hstack((corners, np.ones((4,1)))).T)
                # q = cam.get_features(corners)
                e = s_A - s_B
                e_n = e[:2, :]
                e_n = e_n.T.reshape(8,1)
                print(e_n)
                

                # V_AA = lambdaa * e.reshape((3,1))

                V_AA = -lambdaa * np.linalg.pinv(Ls).dot(e_n)
                # V_AA = np.vstack((V_AA, 1))
                # V_AA = T.dot(np.array(V_AA))
                # print(V_AA)
                # V_AA = np.vstack((V_AA, np.zeros((3,1))))
                T_j = get_TT2(np.linalg.pinv(T))
                T_AR = get_TT2(my_dq.to_matrix())
                # V_AA = np.vstack((V_AA, 1))
                # T_AR = cam.dq.to_matrix()
                ja = V_AA.flatten()
                control_law_AR = np.array([ja[0], ja[1], ja[2], 0, 0, ja[3]])
                # calculate the vlocity of the camera wrt the origin frame
                control_law_AR = T_AR.dot(T_j.dot(control_law_AR.flatten()))




                v = control_law_AR[:3].flatten()
                # w = control_law_AR[3:]
                # print(w)
                w = np.array([0, 0, control_law_AR[5]])

                theta = np.linalg.norm(w)

                if theta == 0:
                    u = [0, 0, 1]
                else: 
                    u = (w/np.linalg.norm(w)).flatten()

                r = Quaternion.from_angle_axis(dt * theta, u)
                dq_update = get_dual(r, dt * v)
                
                my_dq = dq_update * my_dq

                print(v)
                print(w)
                pose = PoseStamped()

                # self.set_dq(self.dq_AR)

                j = my_dq.to_pose()
                pose.pose.position.x = j[0]
                pose.pose.position.y = j[1]
                pose.pose.position.z = j[2]
                pose.pose.orientation.x = j[3]
                pose.pose.orientation.y = j[4]
                pose.pose.orientation.z = j[5]
                pose.pose.orientation.w = j[6]
                self.pos_publisher.publish(pose)

                # print(v)
                # vel_msg = Twist()
                # vel_msg.linear.x = v[0]
                # vel_msg.linear.y = v[1]
                # vel_msg.linear.z = v[2]
                # vel_msg.angular.x = 0
                # vel_msg.angular.y = 0
                # vel_msg.angular.z = control_law_AR[5]
                # self.velocity_publisher.publish(vel_msg)

        except CvBridgeError, e:
            print(e)

    def T_callback(self, msg):
        self.pose = msg
        qt_r = self.pose.pose.orientation
        qt_t = self.pose.pose.position

        # r = Quaternion(qt_r.x, qt_r.y, qt_r.z, qt_r.w)
        # t = Quaternion(qt_t.x, qt_t.y, qt_t.z, 0)
        self.dq = DualQuaternion.from_pose(qt_t.x, qt_t.y, qt_t.z, qt_r.x, qt_r.y, qt_r.z, qt_r.w)


    def set_dq(self, dq):
        self.dq_c = dq
        pose = PoseStamped()
        j = self.dq_c.to_pose()
        pose.pose.position.x = j[0]
        pose.pose.position.y = j[1]
        pose.pose.position.z = j[2]
        pose.pose.orientation.x = j[3]
        pose.pose.orientation.y = j[4]
        pose.pose.orientation.z = j[5]
        pose.pose.orientation.w = j[6]


        # print(pose)
        self.pos_publisher.publish(pose)

    def get_dq(self):
        return self.dq_c

    # def run(self):

    #     dq_AR = self.dq

    #     corners_d = self.detector.detect(self.d_image)[0].corners
    #     s_B = np.linalg.pinv(self.K).dot(np.hstack((corners_d, np.ones((4,1)))).T)
    #     lambdaa = .4
    #     Hz = 50
    #     rate = rospy.Rate(Hz)


    #     dt = 1./Hz # control sampling time

    #     while not rospy.is_shutdown():

            
    #         if len(self.result) != 0:        

 
    #             T = np.array([[ np.cos(1.57079) , 0, -np.sin(1.57079) ,  0   ],
    #                           [ 0               , 1, 0                ,  0   ],
    #                           [ np.sin(1.57079) , 0, np.cos(1.57079)  ,  0.03],
    #                           [ 0               , 0, 0                ,  1   ]])
    #             Ls = Jacobian(s_B.T, .77)
    #             # s_A = cam.get_features(corners)
    #             s_A = np.linalg.pinv(self.K).dot(np.hstack((self.corners, np.ones((4,1)))).T)
    #             # q = cam.get_features(corners)
    #             e = s_A - s_B
    #             e_n = e[:2, :]
    #             e_n = e_n.T.reshape(8,1)
                

    #             # V_AA = lambdaa * e.reshape((3,1))

    #             V_AA = -lambdaa * np.linalg.pinv(Ls).dot(e_n)
    #             # V_AA = np.vstack((V_AA, 1))
    #             # V_AA = T.dot(np.array(V_AA))
    #             # print(V_AA)
    #             # V_AA = np.vstack((V_AA, np.zeros((3,1))))
    #             T_AR = get_TT1(np.linalg.pinv(dq_AR.to_matrix()))
    #             T_j = get_TT2(T)
    #             # V_AA = np.vstack((V_AA, 1))
    #             # T_AR = cam.dq.to_matrix()


    #             # calculate the vlocity of the camera wrt the origin frame
    #             control_law_AR = T_AR.dot(T_j.dot(np.array(V_AA)))
    #             v = control_law_AR[:3].flatten()
    #             w = control_law_AR[3:]
    #             # w = np.array([0, 0, 0])
    #             theta = np.linalg.norm(w)
    #             if theta == 0:
    #                 u = [0, 0, 1]
    #             else: 
    #                 u = (w/np.linalg.norm(w)).flatten()

    #             r = Quaternion.from_angle_axis(dt * theta, u)
    #             dq_update = get_dual(r, dt * v)
    #             dq_AR = dq_update * dq_AR
    #             self.set_dq(dq_AR)

def skew(x):
    return np.array([[ 0    , -x[2] ,  x[1]],
                     [ x[2] ,  0    , -x[0]],
                     [-x[1] ,  x[0] ,  0   ]])


def get_TT1(T):
    TT = np.eye(6)
    TT[:3, :3] = T[:3, :3]
    TT[:3, 3:] = np.zeros(3)
    TT[3:, :3] = np.zeros(3)
    TT[3:, 3:] = T[:3, :3]
    return TT

def get_TT2(T):
    TT = np.eye(6)
    TT[:3, :3] = T[:3, :3]
    TT[:3, 3:] = skew(T[:3, 3]).dot(T[:3, :3])
    TT[3:, :3] = np.zeros(3)
    TT[3:, 3:] = T[:3, :3]
    return TT

def get_dual(q_rot, translation):
    q_t = Quaternion(translation[0], translation[1], translation[2], 0)
    q = DualQuaternion(q_rot, 0.5 * q_t * q_rot)
    return q


def Jacobian(p, depth):
    L = np.zeros((2*p.shape[0], 4))
    j = 0
    for i in range(p.shape[0]):
#         x = (p[i, 0] - 512) * 10e-6 * .008 
#         y = (p[i, 1] - 512) * 10e-6 * .008
        x = p[i, 0]
        y = p[i, 1]
        z = depth
        L[j] = np.array([-1/z, 0, x/z,  y])
        L[j+1] = np.array([0, -1/z, y/z, -x])
        j = j + 2
    return L


def main():

    # velocity_publisher = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
    # vel_msg = Twist()
    
    
    
    d_image = ndimage.imread('./features.png')
    d_image = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    cam = Camera(d_image, 2, 0.008, 512, 512)
    # detector = apriltag.Detector()
    # corners_d = detector.detect(d_image)[0].corners
    # s_B = np.linalg.pinv(cam.K).dot(np.hstack((corners_d, np.ones((4,1)))).T)
    # s_B = cam.get_d_features(corners_d)
    # rospy.sleep(3)
    rospy.spin()
    # cam.run()
    # lambdaa = .4
    # Hz = 20
    # rate = rospy.Rate(Hz)
    # # dq_AR = cam.dq
    # # print(dq_AR.q_rot)
    # time = 0 # current time (seconds)
    # tf = 10   # final time  (seconds)
    
    # dt = 1./Hz # control sampling time

    # T = np.array([[ np.cos(1.57079) , 0, -np.sin(1.57079) ,  0   ],
    #               [ 0               , 1, 0                ,  0   ],
    #               [ np.sin(1.57079) , 0, np.cos(1.57079)  ,  0.03],
    #               [ 0               , 0, 0                ,  1   ]])
    # Ls = Jacobian(s_B.T, .77)
    # image_pub = rospy.Publisher('modified_image', Image, queue_size=1)
    # while not rospy.is_shutdown():
    # # while (time < tf):
    #     red   = [0,0,255]
    #     green = [0,255,0]

    #     img = cv2.cvtColor(cam.image, cv2.COLOR_BGR2GRAY)
    #     new_img = cam.image
    #     result = detector.detect(img)
    #     if len(result) != 0:
            

    #         corners = result[0].corners
    #         center = result[0].center
    #         print(corners)
    #         h = result[0].homography
            

    #         try:
    #             image_pub.publish(cam.bridge.cv2_to_imgmsg(cam.image_corner, "bgr8"))
    #         except CvBridgeError as e:
    #             print(e)


    #         # s_A = cam.get_features(corners)
    #         s_A = np.linalg.pinv(cam.K).dot(np.hstack((corners, np.ones((4,1)))).T)
    #         # q = cam.get_features(corners)
    #         e = s_A - s_B
    #         e_n = e[:2, :]
    #         e_n = e_n.T.reshape(8,1)
    #         print(e)
    #         print(s_A)
            

    #         # V_AA = lambdaa * e.reshape((3,1))
    #         V_AA = -lambdaa * np.linalg.pinv(Ls).dot(e_n)
    #         # V_AA = np.vstack((V_AA, 1))
    #         # V_AA = T.dot(np.array(V_AA))
    #         # print(V_AA)
    #         # V_AA = np.vstack((V_AA, np.zeros((3,1))))
    #         T_AR = get_TT1(dq_AR.to_matrix())
    #         T_j = get_TT2(T)
    #         # V_AA = np.vstack((V_AA, 1))
    #         # T_AR = cam.dq.to_matrix()


    #         # calculate the vlocity of the camera wrt the origin frame
    #         control_law_AR = T_AR.dot(T_j.dot(np.array(V_AA)))
    #         v = control_law_AR[:3].flatten()
    #         w = control_law_AR[3:]
    #         # w = np.array([0, 0, 0])
    #         theta = np.linalg.norm(w)
    #         if theta == 0:
    #             u = [0, 0, 1]
    #         else: 
    #             u = (w/np.linalg.norm(w)).flatten()

    #         r = Quaternion.from_angle_axis(dt * theta, u)
    #         dq_update = get_dual(r, dt * v)
    #         dq_AR = dq_update * dq_AR 
    #         cam.set_dq(dq_AR)

    #         # # # dq_AR = cam.get_dq()
    #         # print(v)
    #         # vel_msg.linear.x = V_AA[0]
    #         # vel_msg.linear.y = V_AA[1]
    #         # vel_msg.linear.z = v[2]
    #         # vel_msg.angular.x = 0
    #         # vel_msg.angular.y = 0
    #         # vel_msg.angular.z = 0
    #         # velocity_publisher.publish(vel_msg)


    #     rate.sleep() # every 1/hz second
            # time = time + dt

if __name__ == '__main__':
    main()








