#include "mujoco.h"
#include "glfw3.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <Eigen/Dense>
#include <iostream>

//-------------------------------- global -------------------------------------
GLFWwindow *window = NULL;

mjModel *m = NULL;
mjData *d = NULL;

mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

std::mutex mtx;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

void init(void)
{
    // check for mjkey.txt in ~/.mujoco and activate it
    char mjkey_path[4096 + 1024];

    char *home = getenv("HOME");
    if ( home )
        snprintf(mjkey_path, sizeof mjkey_path, "%.4096s/.mujoco/mjkey.txt", home);

    mj_activate(mjkey_path);

    char error[1000] = "";
    m = mj_loadXML("assets/cassie_leg_planar_fixed.xml", NULL, error, 1000);
    if ( !m )
        mju_error_s("%.1000s", error);

    // make data
    d = mj_makeData(m);

    // initialize OpenGL
    if (!glfwInit())
        mju_error("could not initialize GLFW");

    GLFWvidmode vmode = *glfwGetVideoMode(glfwGetPrimaryMonitor());

    window = glfwCreateWindow((2 * vmode.width) / 3, (2 * vmode.height) / 3, "IK", NULL, NULL);

    if( !window )
    {
        glfwTerminate();
        mju_error("could not create window");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
}

void cassie_ik(double lx, double ly, double lz,
               double rx, double ry, double rz)
{
    using namespace Eigen;

    mjtNum left_x[3]  = {lx, ly, lz};
    //mjtNum right_x[3] = {rx, ry, rz};

    int left_foot_id  = mj_name2id(m, mjOBJ_BODY, "left-foot");

    int left_heel_spring_id = mj_name2id(m, mjOBJ_JOINT, "left-heel-spring");
    int left_shin_id = mj_name2id(m, mjOBJ_JOINT, "left-shin");

    // int right_foot_id = mj_name2id(m, mjOBJ_BODY, "right-foot");
    // int pelvis_id = mj_name2id(m, mjOBJ_BODY, "cassie-pelvis");

    // Generalized coordinate column vector
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> q_pos(d->qpos, m->nq, 1);

    // End effector position Jacobian
    Matrix<double, Dynamic, Dynamic, RowMajor> J_p(3, m->nv);

    // End effector rotation Jacobian
    Matrix<double, Dynamic, Dynamic, RowMajor> J_r(3, m->nv);

    // Floating base Jacobian
    Matrix<double, Dynamic, Dynamic, RowMajor> J_f(3, m->nv);

    // Foot desired positions
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> left_x_des(left_x, 3, 1);
    // Map<Matrix<double, Dynamic, Dynamic, RowMajor>> right_x_des(right_x, 3, 1);

    // Foot positions
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> left_x_pos(&d->xpos[3 * left_foot_id], 3, 1);
    // Map<Matrix<double, Dynamic, Dynamic, RowMajor>> right_x_pos(&d->xpos[3 * right_foot_id], 3, 1);

    // Step direction
    Matrix<double, Dynamic, Dynamic, RowMajor> dq(m->nq, 1);

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> efc_J(d->efc_J, m->njmax, m->nv);

    //std::cout << efc_J << std::endl;
    // std::cout << G << std::endl; 

    // left foot IK
    for (int i = 0; i < 1000; i++)
    {
        // prepare jacobians
        mj_kinematics(m, d);
        mj_comPos(m, d);
        mj_makeConstraint(m, d);

        // number of joint limit and equality constraints
        int njl_eq = d->nefc - (d->nf + d->ncon);

        // Equality and joint limit constraint violation Jacobian 
        // Equality and joint limits are consecutive in the constraint jacobian
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>> G(d->efc_J, njl_eq, m->nv);

        // get end effector jacobian
        mj_jacBody(m, d, J_p.data(), J_r.data(), left_foot_id);

        // Zero out jacobian columns relating to spring degrees of freedom
        // plus 2 because ball joint has 2 extra dof
        G.col(left_heel_spring_id + 2).setZero();
        G.col(left_shin_id + 2).setZero();

        J_p.col(left_heel_spring_id + 2).setZero();
        J_p.col(left_shin_id + 2).setZero();

        MatrixXd Ginv = G.completeOrthogonalDecomposition().pseudoInverse();
        MatrixXd I = MatrixXd::Identity(Ginv.rows(), Ginv.rows());

        // Null space projector
        MatrixXd N = (I - G.transpose() * Ginv.transpose()).transpose();

        dq = -N * J_p.transpose() * (left_x_pos - left_x_des);

        // velocity might have different dimension than position due to quaternions, 
        // so we must integrate it
        mj_integratePos(m, q_pos.data(), dq.data(), 0.01);

        // zero out floating base dof
        //dq.block(0, 0, 7, 1) = MatrixXd::Zero(7, 1);

        //mj_normalizeQuat(m, d->qpos);

        // print out quaternion
        // std::cout << dq.transpose() << std::endl;
        // std::cout << left_shin_id << " " << left_heel_spring_id << std::endl;

    }

    // efc_J:
    // neq
    // nefc - neq

    // Map<Matrix<int, Dynamic, Dynamic, RowMajor>> efc_type(d->efc_type, 1, d->nefc);
    // std::cout << efc_type << std::endl;

    //std::cout << "NEFC: " << d->nefc << std::endl << efc_J << std::endl;

    std::cout << left_x_pos.transpose() << " | " << left_x_des.transpose()  << std::endl;
    // std::cin >> a;
}


void ik(double x, double y, double z)
{
    int bodyid = mj_name2id(m, mjOBJ_BODY, "end0");

    // MuJoCo matrix definitions
    mjtNum point[3] = {x, y, z};

    using namespace Eigen;

    // End effector Jacobians
    Matrix<double, Dynamic, Dynamic, RowMajor> J_p(3, m->nv);
    Matrix<double, Dynamic, Dynamic, RowMajor> J_r(3, m->nv);

    // Constraint Jacobian
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> G(d->efc_J, m->neq * 3, m->nv);

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> g_err(d->efc_J, 3, 1);

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> x_des(point, 3, 1);

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> x_pos(&d->xpos[3 * bodyid], 3, 1);
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> q_pos(d->qpos, m->nv, 1);

    Matrix<double, Dynamic, Dynamic, RowMajor> dq(1, m->nv);

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> efc_J(d->efc_J, m->njmax, m->nv);

    for (int i = 0; i < 100; i++)
    {
        // prepare jacobians
        mj_kinematics(m, d);
        mj_comPos(m, d);
        mj_makeConstraint(m, d);

        std::cout << m->neq << std::endl;
        int a;
        std::cin >> a;

        MatrixXd Ginv = G.completeOrthogonalDecomposition().pseudoInverse();
        MatrixXd I = MatrixXd::Identity(Ginv.rows(), Ginv.rows());

        MatrixXd N = (I - G.transpose() * Ginv.transpose()).transpose();

        // get jacobian
        mj_jacBody(m, d, J_p.data(), J_r.data(), bodyid);

        dq = N * J_p.transpose() * (x_pos - x_des);

        q_pos -= dq;
    }
}

void simulate(void)
{
    int wait_time = 1;
    while ( true )
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        
        mtx.lock();

        //mj_step(m, d);

        // mjtNum simstart = d->time;
        // while( d->time - simstart < 1.0/60.0 )
        //     mj_step(m, d);

        // ik(0.2, 0, 0.1);

        cassie_ik(-0.02, 0.13477171, .6,
                   0,    0,          0);

        mtx.unlock();
    }
}

void render(GLFWwindow *window)
{
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
}

int main(void)
{
    init();

    std::thread simthread(simulate);

    while( !glfwWindowShouldClose(window) )
    {
        // start exclusive access (block simulation thread)
        mtx.lock();

        // handle events (calls all callbacks)
        glfwPollEvents();

        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // end exclusive access (allow simulation thread to run)
        mtx.unlock();

        // render while simulation is running
        render(window);
    }

    simthread.join();

    // Free data
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    mj_deleteData(d); 
    mj_deleteModel(m);
    mj_deactivate();

    return 0;
}