#include <GL/freeglut_std.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include "meshvisualizer.h"

MeshVisualizer::MeshVisualizer(const string &title, const BasicMesh &mesh)
  : QGLWidget(QGLFormat(QGL::SampleBuffers | QGL::AlphaChannel | QGL::DepthBuffer)),
    mesh(mesh), image_tex(-1), face_alpha(0.5),
    draw_faces(true), draw_edges(false)
{
  setWindowTitle(title.c_str());
}

void MeshVisualizer::initializeGL() {
  glShadeModel(GL_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_TEXTURE_2D);

  if(!image.isNull()) {
    CreateTexture();
  }
}

void MeshVisualizer::CreateTexture() {
  cout << "Creating opengl texture ..." << endl;
#if 1
  if( image_tex >= 0 )
    glDeleteTextures(1, &image_tex);

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &image_tex);
  glBindTexture(GL_TEXTURE_2D, image_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image.bits());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#else
  image_tex = bindTexture(pixmap);
#endif
  cout << "texture id = " << image_tex << endl;
  cout << "done." << endl;
}

void MeshVisualizer::paintGL() {
  glClearColor(1, 1, 1, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  if( mesh.NumFaces() > 0 ) {
    // Draw image
#if 1
    if( !image.isNull() ) {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0001, 10.0);
      glViewport(0, 0, width(), height());

      glColor4f(1, 1, 1, 0.0);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, image_tex);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(-1.0f, -1.0f, -5.0f);
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(1.0f, -1.0f, -5.0f);
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, -5.0f);
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, -5.0f);
      }
      glEnd();
      glPopMatrix();
      glDisable(GL_TEXTURE_2D);
    }
#endif

    // Setup Camera's model-view-projection matrix
    // Can equivalently build a camera matrix directly, and use glLoadMatrixd instead
    if( 0 ) {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(45.0,
                     static_cast<float>(width()) / static_cast<float>(height()),
                     2.0, 100.00);
      glViewport(0, 0, width(), height());

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      gluLookAt(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    } else {
      glMatrixMode(GL_PROJECTION);
      glm::dmat4 Mproj = glm::perspective(45.0,
                                          static_cast<double>(width()) / static_cast<double>(height()),
                                          1.0, 10.0);

      glLoadMatrixd(&Mproj[0][0]);
      glViewport(0, 0, width(), height());

      glm::dmat4 Rmat = glm::eulerAngleYXZ(mesh_rotation[0],
                                           mesh_rotation[1],
                                           mesh_rotation[2]);

      glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                       glm::dvec3(mesh_translation[0],
                                                  mesh_translation[1],
                                                  mesh_translation[2]));

      glm::dmat4 MV = Tmat * Rmat;
      glMatrixMode(GL_MODELVIEW);
      glLoadMatrixd(&MV[0][0]);
    }

    glPushMatrix();

    EnableLighting();

    if( draw_faces ) {
      /// Draw faces
      glColor4d(.75, .75, .75, face_alpha);
      glBegin(GL_TRIANGLES);
      for (int i = 0; i < mesh.NumFaces(); ++i) {
        auto face_i = mesh.face(i);
        auto v0 = mesh.vertex(face_i[0]);
        auto v1 = mesh.vertex(face_i[1]);
        auto v2 = mesh.vertex(face_i[2]);
        auto n = mesh.normal(i);
        glNormal3dv(n.data());glVertex3dv(v0.data());
        glNormal3dv(n.data());glVertex3dv(v1.data());
        glNormal3dv(n.data());glVertex3dv(v2.data());
      }
      glEnd();
    }

    if( draw_edges ) {
      /// Draw edges
      glColor3f(.25, .25, .25);
      glLineWidth(2.5);
      for (int i = 0; i < mesh.NumFaces(); ++i) {
        auto face_i = mesh.face(i);
        auto v0 = mesh.vertex(face_i[0]);
        auto v1 = mesh.vertex(face_i[1]);
        auto v2 = mesh.vertex(face_i[2]);
        glBegin(GL_LINE_LOOP);
        glVertex3dv(v0.data());
        glVertex3dv(v1.data());
        glVertex3dv(v2.data());
        glEnd();
      }
    }

    // Draw landmarks
    {
      glColor3f(.75, .25, .25);
      GLfloat mat_diffuse[] = {0.875, 0.375, 0.375, 1.0};
      GLfloat mat_specular[] = {0.875, 0.875, 0.875, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
      glPointSize(3.0);
      glBegin(GL_POINTS);
      for (auto &landmark : landmarks) {
        auto v = mesh.vertex(landmark);
        const double delta_z = 1e-2;
#if 1
        glPushMatrix();
        glTranslated(v[0], v[1], v[2] + delta_z);
        glutSolidSphere(0.015, 32, 32);
        glPopMatrix();
#else
        glVertex3d(v[0], v[1], v[2] + delta_z);
#endif
      }
      glEnd();
    }

    glPopMatrix();

    // Draw constraints
    glDisable(GL_CULL_FACE);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width(), 0, height(), 0.0001, 1000.0);
    glViewport(0, 0, width(), height());

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    glColor3f(.25, .75, .25);
    GLfloat mat_diffuse[] = {0.375, 0.875, 0.375, 1.0};
    GLfloat mat_specular[] = {0.875, 0.875, 0.875, 1.0};
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for(auto& constraint : constraints) {
#if 1
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glTranslated(constraint.data.x, constraint.data.y, 2.0);
      glutSolidSphere(2.5, 32, 32);
      glPopMatrix();
#else
      glVertex3d(constraint.data.x, constraint.data.y, 3.0);
#endif
    }
    glEnd();

    DisableLighting();
  } else {
    cout << "no mesh" << endl;
    glColor3f(1, 0, 0);
    glPushMatrix();
    glLineWidth(2.5);
    glColor3f(0.25, 0.75, 0.25);
    double square_size = 1.0;
    glBegin(GL_LINE_LOOP);
    glVertex3d(-square_size, -square_size, 0.0);
    glVertex3d( square_size, -square_size, 0.0);
    glVertex3d( square_size,  square_size, 0.0);
    glVertex3d(-square_size,  square_size, 0.0);
    glEnd();
    glPopMatrix();
  }
}

void MeshVisualizer::resizeGL(int w, int h) {
  QGLWidget::resizeGL(w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, static_cast<float>(w)/ static_cast<float>(h), 0.0001, 1000.00);
  glViewport(0, 0, w, h);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void MeshVisualizer::EnableLighting()
{
  GLfloat light_position[] = {10.0, 4.0, 10.0, 1.0};
  GLfloat mat_specular[] = {0.5, 0.5, 0.5, 1.0};
  GLfloat mat_diffuse[] = {0.375, 0.375, 0.375, 1.0};
  GLfloat mat_shininess[] = {25.0};
  GLfloat light_ambient[] = {0.05, 0.05, 0.05, 1.0};
  GLfloat white_light[] = {1.0, 1.0, 1.0, 1.0};

  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
  glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

  light_position[0] = -10.0;
  glLightfv(GL_LIGHT1, GL_POSITION, light_position);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, white_light);
  glLightfv(GL_LIGHT1, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
}

void MeshVisualizer::DisableLighting()
{
  glDisable(GL_LIGHT0);
  glDisable(GL_LIGHT1);
  glDisable(GL_LIGHTING);
}

void MeshVisualizer::BindConstraints(const vector<Constraint2D> &constraints_in) {
  constraints = constraints_in;
}

void MeshVisualizer::BindImage(const QImage& img) {
  image = convertToGLFormat(img);
  CreateTexture();
}

void MeshVisualizer::BindLandmarks(const vector<int> &landmarks_in) {
  landmarks = landmarks_in;
}

void MeshVisualizer::SetMeshRotationTranslation(const Vector3d &R,
                                                const Vector3d &T) {
  mesh_rotation = R;
  mesh_translation = T;
}

void MeshVisualizer::SetCameraParameters(const CameraParameters &cam_params) {
  camera_params = cam_params;
}
