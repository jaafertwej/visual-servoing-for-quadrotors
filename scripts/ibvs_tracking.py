from __future__ import division
import numpy as np
import scipy
from scipy import ndimage
import rospy
import tf

deg2rad = np.pi/180


from sensor_msgs.msg import Image, Imu
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import apriltag
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, Pose
from quaternion import Quaternion
from dualQuaternion import DualQuaternion
from tf.transformations import euler_from_quaternion, concatenate_matrices, rotation_matrix, euler_from_matrix
import matplotlib.pyplot as plt
from math import * 


class Camera():

    def __init__(self, d_image, depth):
        self.d_image = d_image
        self.depth = depth
        self.mass = 1.5
        self.gr = 9.80665
        self.gamma = 1.5
        self.er = np.array([[0],[0],[1]])
        self.K = np.array([[476.7030836014194, 0.0              , 400.5],
                           [0.0              , 476.7030836014194, 400.5],
                           [0.0              , 0.0              , 1.0  ]])

        self.bridge = CvBridge()
        self.detector = apriltag.Detector()
        rospy.init_node('ibvs')
        image_topic = "/hummingbird/camera_nadir/image_raw"
        rospy.Subscriber('/hummingbird/ground_truth/pose', Pose, self.T_callback)
        rospy.sleep(.5)

        rospy.Subscriber(image_topic, Image, self.image_callback)        
        self.vv = []
        self.ee = []
        self.pos_publisher = rospy.Publisher('/hummingbird/command/pose', PoseStamped, queue_size=1)
        self.image_pub = rospy.Publisher('modified_image', Image, queue_size=10)


    def get_features(self, corners, mode = 'c'):
        
        # s = corners[:2, :].reshape(8,1)

        a, xg, yg = self.moments(corners)
        alpha = self.get_angle(corners)
        if mode == 'd':
            self.a = a
            # alpha = 0
            print('desired: {}'.format(degrees(alpha)))
        else:
            print('current: {}'.format(degrees(alpha)))
        an = self.depth * np.sqrt(self.a/a)
        xn = xg * an
        yn = yg * an

        mu = self.get_mu(corners)
        # alpha = 0.5 * np.arctan((2*mu[0])/(mu[1]-mu[2])) + (mu[1]<mu[2])*np.pi/2
        
        sh = np.array([xn, yn, an, alpha])
        # sv = np.array([an, 0])
        
        # return sh, sv
        return sh

    def get_angle(self, corners):

        delta_y = corners[1, 0] - corners[1, 3]
        delta_x = corners[0, 0] - corners[0, 3]

        th = atan2(delta_y,delta_x)
        # if th < 0:
        #     th = (np.pi*2) + th
        return th


    def get_spherical_features(self, corners):
        
        s = corners/np.linalg.norm(corners)
        q = np.sum(s, axis=1)

        return q

    def get_rescaled_feature(self, q, r):
        
        qn = np.linalg.norm(q)
        q0 = q/qn
        F = (r * qn)/np.sqrt(16-qn**2)
        f = F * q0
        return f

    def moments(self, corners, mode = 'f'):

        m00 = 0
        m10 = 0
        m01 = 0
        for i in range(corners.shape[1]):
            m00 = m00 + (corners[0, i]**0 * corners[1, i]**0)
            m10 = m10 + (corners[0, i]**1 * corners[1, i]**0)
            m01 = m01 + (corners[0, i]**0 * corners[1, i]**1)

        n = m00
        xg = m10/n
        yg = m01/n


        if mode == 'f':

            mu = self.get_mu(corners)
            a = mu[1] + mu[2]
            return a, xg, yg

        elif mode == 'mu':
            return xg, yg

        elif mode == 'epsi':

            mu = self.get_mu(corners)
            a = mu[1] + mu[2]
            return m00, xg, yg, a


    def get_mu(self, corners):
        xg, yg = self.moments(corners, 'mu')

        mu = np.zeros(7)
        for i in range(corners.shape[1]):
            mu[0] = mu[0] + (corners[0, i] - xg)**1 * (corners[1, i] - yg)**1  # mu11 
            mu[1] = mu[1] + (corners[0, i] - xg)**2 * (corners[1, i] - yg)**0  # mu20
            mu[2] = mu[2] + (corners[0, i] - xg)**0 * (corners[1, i] - yg)**2  # mu02
            mu[3] = mu[3] + (corners[0, i] - xg)**2 * (corners[1, i] - yg)**1  # mu21
            mu[4] = mu[4] + (corners[0, i] - xg)**1 * (corners[1, i] - yg)**2  # mu12
            mu[5] = mu[5] + (corners[0, i] - xg)**3 * (corners[1, i] - yg)**0  # mu30
            mu[6] = mu[6] + (corners[0, i] - xg)**0 * (corners[1, i] - yg)**3  # mu03


        return mu

    def get_epsi(self, corners):
        m00, xg, yg, a = self.moments(corners, 'epsi')
        mu = self.get_mu(corners)

        epsi = np.zeros(6)

        epsi[4] = yg + (yg * mu[2] + xg * mu[0] + mu[3] + mu[6])/a
        epsi[5] = xg + (xg * mu[1] + yg * mu[0] + mu[4] + mu[5])/a

        n = mu/m00

        epsi[0] = n[0] + xg * (yg - epsi[4])
        epsi[1] = n[2] + yg * (yg - epsi[4])
        epsi[2] = n[1] + xg * (xg - epsi[5])
        epsi[3] = n[0] + yg * (xg - epsi[5])

        return epsi

    def get_alpha(self, corners):
        mu = self.get_mu(corners)
        xg, yg = self.moments(corners, 'mu')
        beta = 4
        gamma = 2

        alpha = np.zeros(2)

        d = (mu[1] + mu[2])**2 + 4 * mu[0]**2
        alpha[0] = (beta * (mu[4]*(mu[1]-mu[2]) + mu[0]*(mu[6]-mu[3]))
                    + gamma*xg*(mu[2]*(mu[1]-mu[2]) - 2*mu[0]**2) +
                      gamma*yg*mu[0]*(mu[1]-mu[2]))/d

        alpha[1] = (beta * (mu[3]*(mu[2]-mu[1]) + mu[0]*(mu[5]-mu[4]))
                    + gamma*xg*mu[0]*(mu[1]-mu[2])
                    + gamma*(mu[1]*(mu[2]-mu[1])-2*mu[0]**2))/d

        return alpha

    def image_callback(self, msg):

        try:
            # Convert your ROS Image message to OpenCV2
            my_dq = self.dq
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.result = self.detector.detect(image2)

            corners_d1 = self.detector.detect(self.d_image)[0].corners.T
            corners_d = np.linalg.inv(self.K).dot(np.vstack((corners_d1, np.ones((1,4)))))

            R1 = rotation_matrix(180*deg2rad, (1, 0, 0))[:3, :3]
            R2 = rotation_matrix(np.pi/2, (0, 1, 0))[:3, :3]
            R3 = rotation_matrix(-np.pi/2, (0, 0, 1))[:3, :3]

            sh_star = self.get_features(corners_d, 'd')
            b = np.array([0, 0, 1])
            q_star = R3.dot(R1.dot(b))
            r_star = corners_d[0, 0] - corners_d[0, 3]
            f_star = self.get_rescaled_feature(q_star, r_star)
            # s_B = np.linalg.pinv(self.K).dot(np.hstack((corners_d, np.ones((4,1)))).T)

            Hz = 10
            

            dt = 1./Hz 

            if len(self.result) != 0:

                corners = self.result[0].corners.T
                for i in range(4):
                    cv2.circle(image, (int(corners[0, i]), int(corners[1, i])), 3, (0,0,255), -1)
                    cv2.circle(image, (int(corners_d1[0, i]), int(corners_d1[1, i])), 3, (0,255,0), -1)


                corners = np.linalg.inv(self.K).dot(np.vstack((corners, np.ones((1,4)))))
                # print(corners*10)
                a, xg, yg = self.moments(corners)

                self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
                sh = self.get_features(corners)
                # q =  self.get_spherical_features(corners)
                # r = corners[0, 0] - corners[0, 3]
                # f =self. get_rescaled_feature(q, r)

                epsi = self.get_epsi(corners)
                alpha = self.get_alpha(corners)

                Lsh1, Lsh2 = Jacobian_horz(sh_star, epsi)
                Lsv1, Lsv2 = Jacobian_vert(sh_star, epsi, alpha)

                Lsh = np.hstack((Lsh1, Lsh2))
                Lsv = np.hstack((Lsv1, Lsv2))
                Ls  = np.linalg.pinv(np.vstack((Lsh, Lsv)))


                eh = sh - sh_star
                # print(eh[3])
                self.ee.append(eh)
                # delta = f - f_star
                # ev = sv - sv_star
                # e = np.hstack((eh, ev)).reshape(4,1)

                # q1 = s - s_star
               
                # k_f = adaptive_gain(4, .8, 20, np.linalg.norm(eh, np.inf))
                k_f = 2
                v11 = np.linalg.pinv(Ls).dot(-k_f * eh)
                control_law_AR = R3.dot(R1.dot(v11[:3]))


                v = control_law_AR[:3].flatten()
                # print(v)
                # for i in range(3):
                #     if v[i] > 2:
                #         v[i] = 2
                #     elif v[i] < -2:
                #         v[i] = -2

                self.vv.append(np.append(v, -v11[3]))
                # print(eh)
                w = np.array([0., 0., -v11[3]]) 

                theta = np.linalg.norm(w)
                if theta == 0:
                    u = [0, 0, 1]
                else: 
                    u = (w/np.linalg.norm(w)).flatten()

                r = Quaternion.from_angle_axis(dt * theta, u)

                dq_update = get_dual(r, dt * v)
                my_dq =  my_dq * dq_update


                pose = PoseStamped()
                T = my_dq.to_matrix()
                t = T[:3, 3]
                r = Quaternion.from_rotation_matrix(T[:3, :3])
                # r = Quaternion(0, 0, 0, 1)

                pose.pose.position.x = t[0]
                pose.pose.position.y = t[1]
                pose.pose.position.z = t[2]

                pose.pose.orientation.x = r.x
                pose.pose.orientation.y = r.y
                pose.pose.orientation.z = r.z
                pose.pose.orientation.w = r.w
                # print(r)
                self.pos_publisher.publish(pose)

        except CvBridgeError, e:
            print(e)

    def T_callback(self, msg):
        self.pose = msg
        qt_r = self.pose.orientation
        qt_t = self.pose.position
        self.dq = DualQuaternion.from_pose(qt_t.x, qt_t.y, qt_t.z, qt_r.x, qt_r.y, qt_r.z, qt_r.w)


def Jacobian(p, depth):
    L = np.zeros((2*p.shape[1], 4))
    j = 0
    for i in range(p.shape[1]):
#         x = (p[i, 0] - 512) * 10e-6 * .008 
#         y = (p[i, 1] - 512) * 10e-6 * .008
        x = p[0, i]
        y = p[0, i]
        z = depth
        L[j] = np.array([-1/z, 0, x/z,  y])
        L[j+1] = np.array([0, -1/z, y/z, -x])
        j = j + 2
    return L

def skew(x):
    return np.array([[ 0    , -x[2] ,  x[1]],
                     [ x[2] ,  0    , -x[0]],
                     [-x[1] ,  x[0] ,  0   ]])

def get_S_cr(R, t):
    TT = np.eye(4)
    TT[:3, :3] = R.T
    TT[:3,  3] = skew(t).dot(R.T)[:3,  2]
    TT[ 3, :3] = np.zeros(3)
    TT[ 3,  3] = R.T[2, 2]
    return TT


def get_V_wr(dq):
    TT = np.eye(4)
    TT[:3, :3] = get_R_rw(dq)
    TT[:3,  3] = np.zeros(3)
    TT[ 3, :3] = np.zeros(3)
    TT[ 3,  3] = np.linalg.pinv(get_G(dq))[2, 2]
    return TT

def get_R_wr(dq):
    q = dq.q_rot
    r, p, y = euler_from_quaternion([q.x, q.y, q.z, q.w])
    Rx = rotation_matrix(r, (1, 0, 0))
    Ry = rotation_matrix(p, (0, 1, 0))
    Rz = rotation_matrix(y, (0, 0, 1))
    return Rx.dot(Ry.dot(Rz))[:3, :3]

def get_G(dq):
    q = dq.q_rot
    r, p, y = euler_from_quaternion([q.x, q.y, q.z, q.w])
    G = np.eye(3)
    G[0, 2] = np.sin(p)
    G[1, 1] = np.cos(r)
    G[1, 2] = np.sin(r) * np.cos(p) * -1
    G[2, 1] = np.sin(r)
    G[2, 2] = np.cos(r) * np.cos(p)
    return G


def Jacobian_horz(sh, epsi):
    Lsh1 = np.array([[-1,  0 , 0],
                     [0 , -1 , 0]])

    # Lsh2 = np.array([[sv[0] * epsi[0]    , -sv[0] * (1+epsi[2]),  sh[1]],
    #                  [sv[0] * (1+epsi[1]), -sv[0] * epsi[3]    , -sh[0]]])


    Lsh2 = np.array([[sh[1]],
                     [-sh[0]]])

    return Lsh1, Lsh2

def Jacobian_vert(sv, epsi, alpha):

    Lsv1 = np.array([[0 ,  0, -1],
                     [0 ,  0,  0]])

    # Lsv2 = np.array([[-sv[0] * epsi[4], sv[0] * epsi[5],  0],
    #                  [ alpha[0]       , alpha[1]       , -1]])
    Lsv2 = np.array([[ 0],
                     [-1]])

    return Lsv1, Lsv2

def get_Ps(Lsh1, Lsh2, Lsv1, Lsv2, R_cr, t_rc):

    Psh1 = -Lsh1.dot(R_cr)
    Psh2 =  Psh1.dot(skew(t_rc)) + Lsh2.dot(R_cr)

    Psv1 = -Lsv1.dot(R_cr)
    Psv2 =  Psv1.dot(skew(t_rc)) + Lsv2.dot(R_cr)

    return Psh1, Psh2, Psv1, Psv2

def get_dual(q_rot, translation):
    q_t = Quaternion(translation[0], translation[1], translation[2], 0)
    q = DualQuaternion(q_rot, 0.5 * q_t * q_rot)
    return q
def adaptive_gain(gain_zero, gain_inf, slope_zero, norm_inf):
    a = gain_zero - gain_inf
    b = slope_zero/a
    c = gain_inf
    lamb = a * np.exp(-b * norm_inf) + c
    return lamb
def main():

    d_image = ndimage.imread('../features.png')
    d_image = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    cam = Camera(d_image, 2.42)


    rospy.spin()

    label = ['vx', 'vy', 'vz', 'wz']
    label2 = ['ex', 'ey', 'ez', r'e$\alpha$']
    color = ['#ed2224', '#344ea2', '#1a8585', '#F99706']
    fig = plt.figure(figsize=(10, 30))
    t1 = np.arange(0.0, len(cam.vv), 1)
    fig.add_subplot(2,1,1)
    plt.ylim([-2, 2])
    for i in range(4):
        plt.plot(t1, np.array(cam.ee)[:, i], label= label2[i])

    plt.legend(loc='upper right')

    fig.add_subplot(2,1,2)
    plt.ylim([-3, 3])
    for i in range(4):
        plt.plot(t1, np.array(cam.vv)[:, i], color = color[i], label= label[i])

    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()

